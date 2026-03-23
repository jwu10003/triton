#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOOPINVARIANTCODEMOTION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-licm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class LoopInvariantCodeMotionPass
    : public impl::TritonLoopInvariantCodeMotionBase<
          LoopInvariantCodeMotionPass> {

  DenseMap<LoopLikeOpInterface, bool> isLoopMemoryEffectFreeOrOnlyRead;

  bool isMemoryEffectFreeOrOnlyRead(Operation *op) {
    std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
        getEffectsRecursively(op);
    if (!effects)
      return false;
    return llvm::all_of(*effects,
                        [&](const MemoryEffects::EffectInstance &effect) {
                          return isa<MemoryEffects::Read>(effect.getEffect());
                        });
  }

  // Returns true if op is a "load-like" operation: it implements the memory
  // effects interface and has only read effects (no write/alloc/free).
  // This covers tt.load, amdg.buffer_load, and any other dialect-specific
  // load operations.
  bool isLoadLikeOp(Operation *op) {
    auto memEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memEffectOp)
      return false;
    SmallVector<MemoryEffects::EffectInstance> effects;
    memEffectOp.getEffects(effects);
    if (effects.empty())
      return false;
    return llvm::all_of(effects,
                        [&](const MemoryEffects::EffectInstance &effect) {
                          return isa<MemoryEffects::Read>(effect.getEffect());
                        });
  }

  // Check if a type contains a Triton pointer type, either as a scalar
  // !tt.ptr<T> or a tensor of pointers tensor<Nx!tt.ptr<T>>.
  bool isPointerLikeType(Type type) {
    if (isa<triton::PointerType>(type))
      return true;
    if (auto tensorType = dyn_cast<RankedTensorType>(type))
      return isa<triton::PointerType>(tensorType.getElementType());
    return false;
  }

  // Trace a value through addptr/splat/broadcast chains to find the root
  // function argument (base pointer). Returns nullptr if the root cannot be
  // determined or is not a function argument.
  Value getRootBasePointer(Value ptr) {
    while (ptr) {
      if (auto blockArg = dyn_cast<BlockArgument>(ptr)) {
        // Check if it's a function argument (entry block of a FuncOp).
        if (blockArg.getOwner()->isEntryBlock())
          return ptr;
        return nullptr;
      }
      Operation *defOp = ptr.getDefiningOp();
      if (!defOp)
        return nullptr;
      if (auto addptrOp = dyn_cast<AddPtrOp>(defOp)) {
        ptr = addptrOp.getPtr();
      } else if (auto splatOp = dyn_cast<SplatOp>(defOp)) {
        ptr = splatOp.getSrc();
      } else if (auto broadcastOp = dyn_cast<BroadcastOp>(defOp)) {
        ptr = broadcastOp.getSrc();
      } else {
        return nullptr;
      }
    }
    return nullptr;
  }

  // Find the pointer operand of an operation (load or store).
  // Handles both scalar pointers (!tt.ptr<T>) and tensor of pointers
  // (tensor<Nx!tt.ptr<T>>).
  Value getPointerOperand(Operation *op) {
    for (Value operand : op->getOperands()) {
      if (isPointerLikeType(operand.getType()))
        return operand;
    }
    return nullptr;
  }

  // Collect all root base pointers for write effects in the loop.
  // Returns std::nullopt if any write effect cannot be traced to a function
  // argument (conservative: assume potential aliasing with everything).
  std::optional<SmallVector<Value>>
  getLoopWriteBasePointers(LoopLikeOpInterface loopLike) {
    SmallVector<Value> writeBasePtrs;
    bool failed = false;
    for (Region *region : loopLike.getLoopRegions()) {
      region->walk([&](Operation *op) {
        if (failed)
          return;
        auto memEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
        if (!memEffectOp)
          return;
        SmallVector<MemoryEffects::EffectInstance> effects;
        memEffectOp.getEffects(effects);
        for (auto &effect : effects) {
          if (!isa<MemoryEffects::Write>(effect.getEffect()))
            continue;
          Value ptr = getPointerOperand(op);
          if (!ptr) {
            // Write op without a recognizable pointer operand (e.g.,
            // tt.print, tt.assert). These don't alias with memory loads,
            // so skip them.
            continue;
          }
          Value root = getRootBasePointer(ptr);
          if (!root) {
            // Cannot trace write pointer to a function argument.
            // Be conservative: cannot prove non-aliasing.
            failed = true;
            return;
          }
          writeBasePtrs.push_back(root);
        }
      });
    }
    if (failed)
      return std::nullopt;
    return writeBasePtrs;
  }

  // Check if a load op's base pointer is provably non-aliasing with all
  // write base pointers in the loop.
  bool isLoadNonAliasingWithLoopWrites(
      Operation *op,
      const std::optional<SmallVector<Value>> &writeBasePtrs) {
    // If we couldn't determine write base pointers, be conservative.
    if (!writeBasePtrs)
      return false;
    if (writeBasePtrs->empty())
      return true;
    // Find the pointer operand of the load op.
    Value loadPtr = getPointerOperand(op);
    if (!loadPtr)
      return false;
    Value loadRoot = getRootBasePointer(loadPtr);
    if (!loadRoot)
      return false;
    // If the load's root base pointer differs from all write base pointers,
    // there is no aliasing (different kernel arguments point to separate
    // allocations).
    return llvm::all_of(*writeBasePtrs, [&](Value writeRoot) {
      return loadRoot != writeRoot;
    });
  }

  void runOnOperation() override {
    // Walk through all loops in a function in innermost-loop-first order.
    // This way, we first LICM from the inner loop, and place the ops in the
    // outer loop, which in turn can be further LICM'ed.
    getOperation()->walk([&](LoopLikeOpInterface loopLike) {
      // Write base pointers for the loop, computed lazily.
      // outer optional: not yet computed; inner optional: computed but may have
      // failed (nullopt means couldn't trace all write pointers).
      std::optional<std::optional<SmallVector<Value>>> loopWriteBasePtrs;

      moveLoopInvariantCode(
          loopLike.getLoopRegions(),
          // isDefinedOutsideOfRegion
          [&](Value value, Region *region) {
            return loopLike.isDefinedOutsideOfLoop(value);
          },
          // shouldMoveOutOfRegion
          [&](Operation *op, Region *region) {
            if (!isLoadLikeOp(op))
              return isSpeculatable(op) && isMemoryEffectFree(op);
            // For load-like ops (tt.load, amdg.buffer_load, etc.):
            // First check if the loop is memory-effect-free or read-only.
            if (!isLoopMemoryEffectFreeOrOnlyRead.contains(loopLike))
              isLoopMemoryEffectFreeOrOnlyRead[loopLike] =
                  isMemoryEffectFreeOrOnlyRead(loopLike);
            if (isLoopMemoryEffectFreeOrOnlyRead[loopLike])
              return true;
            // The loop has write effects. Check if the load's base pointer
            // is provably non-aliasing with all write base pointers.
            if (!loopWriteBasePtrs.has_value())
              loopWriteBasePtrs = getLoopWriteBasePointers(loopLike);
            return isLoadNonAliasingWithLoopWrites(op, *loopWriteBasePtrs);
          },
          // moveOutOfRegion
          [&](Operation *op, Region *) {
            // Create the new mask for tt.load op to guard against zero-trip
            // loops.
            if (auto loadOp = dyn_cast<LoadOp>(op)) {
              IRRewriter rewriter(loopLike);
              Location loc = loopLike->getLoc();
              Value cond;
              if (auto forOp = dyn_cast<scf::ForOp>(loopLike.getOperation())) {
                cond = arith::CmpIOp::create(
                    rewriter, loc, arith::CmpIPredicate::slt,
                    forOp.getLowerBound(), forOp.getUpperBound());
              } else if (auto whileOp =
                             dyn_cast<scf::WhileOp>(loopLike.getOperation())) {
                // TODO: Support Load Op hoisting for while loop.
                return;
              } else {
                return;
              }
              Value newMask = getPredMask(rewriter, loadOp.getPtr().getType(),
                                          loadOp.getMask(), cond);
              loadOp.getMaskMutable().assign(newMask);
            }
            loopLike.moveOutOfLoop(op);
          });
    });
  }
};

} // namespace mlir::triton
