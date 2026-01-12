import os
os.environ["HIP_VISIBLE_DEVICES"] = "4,5"

import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

GROUP_SIZE = 32
PACK_B = 2

torch.manual_seed(0)


'''
################################################################################
# Gluon baseline
# M:15   N:57344 K:8192 BLOCK_SIZE_M:16   BLOCK_SIZE_N:64   BLOCK_SIZE_K:128  num_warps:4    num_stages:2    kpack:1    nonkdim:0    duration:0.77668  tflops:18.145   bandwidth:323.85
################################################################################

@gluon.jit
def w4a16_kernel(
        a_ptr, b_ptr, bs_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_bsn, stride_bsk,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
        BLOCK_SIZE_K: gl.constexpr,
        PAD_M: gl.constexpr,
        PAD_N: gl.constexpr,
        PAD_K: gl.constexpr,
        a_cache: gl.constexpr,
        b_cache: gl.constexpr,
        bs_cache: gl.constexpr
):
    """Kernel for computing the matmul C = A x (B x scale).
    A has shape (M, K) with bfloat16 data type
    B has shape (N, K) with mxfp4 data type
    Pack B as (N, K // 2) with uint8 data type
    BS has shape (N, K // 32) with fp8 data type
    C has shape (M, N) with bfloat16 data type
    """

    PACK_B : gl.constexpr = 2
    GROUP_SIZE : gl.constexpr = 32

    blocked_a: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    blocked_b: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )

    blocked_bs: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[1, 4],
    )

    mfma_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )

    mfma_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    blocked_bf16: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 2, 1],
        threads_per_warp=[4, 1, 16],
        warps_per_cta=[1, 1, 4],
        order=[1, 0, 2],
    )

    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[0, 1]
    )
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )
    smem_b = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_K, BLOCK_SIZE_N], layout=shared_b
    )

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)

    if PAD_M:
        offs_am = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_a))) % M
    else:
        offs_am = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_a))
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_a))

    if PAD_N:
        offs_bn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))) % N
    else:
        offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))
    offs_bk = gl.arange(0, BLOCK_SIZE_K // PACK_B, layout=gl.SliceLayout(1, blocked_b))

    if PAD_N:
        offs_bsn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_bs))) % N
    else:
        offs_bsn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_bs))
    offs_bsk = gl.arange(0, BLOCK_SIZE_K // GROUP_SIZE, layout=gl.SliceLayout(1, blocked_bs))

    a_ptr_offs = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_ptr_offs = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    bs_ptr_offs = offs_bsk[:, None] * stride_bsk + offs_bsn[None, :] * stride_bsn

    ############################################################################
    # accumulator
    ############################################################################

    accumulator = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout)

    ############################################################################
    # main loop
    ############################################################################

    num_k_iter = gl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_iter):
        bs = gl.amd.cdna3.buffer_load(bs_ptr, bs_ptr_offs)
        b = gl.amd.cdna3.buffer_load(b_ptr, b_ptr_offs)
        a = gl.amd.cdna3.buffer_load(a_ptr, a_ptr_offs)

        # Convert bs to bfloat16
        bs_u8 = bs.to(gl.uint8, bitcast=True)
        bs_u16 = bs_u8.to(gl.uint16)
        bs_u16 = bs_u16 << 7
        bs_bf16 = bs_u16.to(gl.bfloat16, bitcast=True)
        # reshape
        bs_bf16 = gl.reshape(bs_bf16, (BLOCK_SIZE_K // GROUP_SIZE, 1, BLOCK_SIZE_N))

        # Convert b to bfloat16
        b_bf16 = gl.fp4_to_fp(b, gl.bfloat16, 0)
        # reshape
        b_bf16 = gl.reshape(b_bf16, (BLOCK_SIZE_K // GROUP_SIZE, GROUP_SIZE, BLOCK_SIZE_N))

        bs_bf16 = gl.convert_layout(bs_bf16, b_bf16.type.layout)
        b = bs_bf16 * b_bf16
        b = gl.reshape(b, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        smem_a.store(a)
        smem_b.store(b)

        a_converted = smem_a.load(layout=mfma_lhs_layout)
        b_converted = smem_b.load(layout=mfma_rhs_layout)

        # We accumulate along the K dimension.
        accumulator = gl.amd.cdna3.mfma(a_converted, b_converted, accumulator)

        # Advance the ptrs to the next K block.
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += (BLOCK_SIZE_K // PACK_B) * stride_bk
        bs_ptr += (BLOCK_SIZE_K // GROUP_SIZE) * stride_bsk

    ############################################################################
    # end
    ############################################################################
    
    # Convert accumulator to bfloat16
    c = accumulator.to(gl.bfloat16)

    if PAD_M:
        offs_cm = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout))) % M
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout))
    if PAD_N:
        offs_cn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))) % N
    else:
        offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))
    gl.amd.cdna3.buffer_store(c, c_ptr, stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])
'''

'''
################################################################################
# Use prefetch
# M:15   N:57344 K:8192 BLOCK_SIZE_M:16   BLOCK_SIZE_N:64   BLOCK_SIZE_K:128  num_warps:4    num_stages:2    kpack:1    nonkdim:0    duration:0.74265  tflops:18.977   bandwidth:338.69
################################################################################

@gluon.jit
def w4a16_kernel(
        a_ptr, b_ptr, bs_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_bsn, stride_bsk,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
        BLOCK_SIZE_K: gl.constexpr,
        PAD_M: gl.constexpr,
        PAD_N: gl.constexpr,
        PAD_K: gl.constexpr,
        a_cache: gl.constexpr,
        b_cache: gl.constexpr,
        bs_cache: gl.constexpr
):
    """Kernel for computing the matmul C = A x (B x scale).
    A has shape (M, K) with bfloat16 data type
    B has shape (N, K) with mxfp4 data type
    Pack B as (N, K // 2) with uint8 data type
    BS has shape (N, K // 32) with fp8 data type
    C has shape (M, N) with bfloat16 data type
    """

    PACK_B : gl.constexpr = 2
    GROUP_SIZE : gl.constexpr = 32

    blocked_a: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    blocked_b: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )

    blocked_bs: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=False,
        warps_per_cta=[1, 4],
    )

    mfma_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )

    mfma_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[0, 1]
    )
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )
    smem_b = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_K, BLOCK_SIZE_N], layout=shared_b
    )

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)

    if PAD_M:
        offs_am = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_a))) % M
    else:
        offs_am = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_a))
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_a))

    if PAD_N:
        offs_bn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))) % N
    else:
        offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))
    offs_bk = gl.arange(0, BLOCK_SIZE_K // PACK_B, layout=gl.SliceLayout(1, blocked_b))

    if PAD_N:
        offs_bsn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_bs))) % N
    else:
        offs_bsn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_bs))
    offs_bsk = gl.arange(0, BLOCK_SIZE_K // GROUP_SIZE, layout=gl.SliceLayout(1, blocked_bs))

    a_ptr_offs = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_ptr_offs = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    bs_ptr_offs = offs_bsk[:, None] * stride_bsk + offs_bsn[None, :] * stride_bsn

    ############################################################################
    # accumulator
    ############################################################################

    accumulator = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout)

    ############################################################################
    # prologue
    ############################################################################

    bs = gl.amd.cdna3.buffer_load(bs_ptr, bs_ptr_offs)
    b = gl.amd.cdna3.buffer_load(b_ptr, b_ptr_offs)
    a = gl.amd.cdna3.buffer_load(a_ptr, a_ptr_offs)

    a_ptr += BLOCK_SIZE_K * stride_ak
    b_ptr += (BLOCK_SIZE_K // PACK_B) * stride_bk
    bs_ptr += (BLOCK_SIZE_K // GROUP_SIZE) * stride_bsk

    # Convert bs to bfloat16
    bs_u8 = bs.to(gl.uint8, bitcast=True)
    bs_u16 = bs_u8.to(gl.uint16)
    bs_u16 = bs_u16 << 7
    bs_bf16 = bs_u16.to(gl.bfloat16, bitcast=True)
    # reshape
    bs_bf16 = gl.reshape(bs_bf16, (BLOCK_SIZE_K // GROUP_SIZE, 1, BLOCK_SIZE_N))

    # Convert b to bfloat16
    b_bf16 = gl.fp4_to_fp(b, gl.bfloat16, 0)
    # reshape
    b_bf16 = gl.reshape(b_bf16, (BLOCK_SIZE_K // GROUP_SIZE, GROUP_SIZE, BLOCK_SIZE_N))

    bs_bf16 = gl.convert_layout(bs_bf16, b_bf16.type.layout)
    b = bs_bf16 * b_bf16
    b = gl.reshape(b, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    smem_a.store(a)
    smem_b.store(b)

    ############################################################################
    # main loop
    ############################################################################

    num_k_iter = gl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_iter - 1):
        bs = gl.amd.cdna3.buffer_load(bs_ptr, bs_ptr_offs)
        b = gl.amd.cdna3.buffer_load(b_ptr, b_ptr_offs)
        a = gl.amd.cdna3.buffer_load(a_ptr, a_ptr_offs)

        a_converted = smem_a.load(layout=mfma_lhs_layout)
        b_converted = smem_b.load(layout=mfma_rhs_layout)

        # Convert bs to bfloat16
        bs_u8 = bs.to(gl.uint8, bitcast=True)
        bs_u16 = bs_u8.to(gl.uint16)
        bs_u16 = bs_u16 << 7
        bs_bf16 = bs_u16.to(gl.bfloat16, bitcast=True)
        # reshape
        bs_bf16 = gl.reshape(bs_bf16, (BLOCK_SIZE_K // GROUP_SIZE, 1, BLOCK_SIZE_N))

        # Convert b to bfloat16
        b_bf16 = gl.fp4_to_fp(b, gl.bfloat16, 0)
        # reshape
        b_bf16 = gl.reshape(b_bf16, (BLOCK_SIZE_K // GROUP_SIZE, GROUP_SIZE, BLOCK_SIZE_N))

        bs_bf16 = gl.convert_layout(bs_bf16, b_bf16.type.layout)
        b = bs_bf16 * b_bf16
        b = gl.reshape(b, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        smem_a.store(a)
        smem_b.store(b)

        # We accumulate along the K dimension.
        accumulator = gl.amd.cdna3.mfma(a_converted, b_converted, accumulator)

        # Advance the ptrs to the next K block.
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += (BLOCK_SIZE_K // PACK_B) * stride_bk
        bs_ptr += (BLOCK_SIZE_K // GROUP_SIZE) * stride_bsk

    ############################################################################
    # epilogue
    ############################################################################

    a_converted = smem_a.load(layout=mfma_lhs_layout)
    b_converted = smem_b.load(layout=mfma_rhs_layout)

    accumulator = gl.amd.cdna3.mfma(a_converted, b_converted, accumulator)

    ############################################################################
    # end
    ############################################################################
    
    # Convert accumulator to bfloat16
    c = accumulator.to(gl.bfloat16)

    if PAD_M:
        offs_cm = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout))) % M
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout))
    if PAD_N:
        offs_cn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))) % N
    else:
        offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))
    gl.amd.cdna3.buffer_store(c, c_ptr, stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])
'''

'''
################################################################################
# Opt bf16 mul
# M:15   N:57344 K:8192 BLOCK_SIZE_M:16   BLOCK_SIZE_N:64   BLOCK_SIZE_K:128  num_warps:4    num_stages:2    kpack:1    nonkdim:0    duration:0.45900  tflops:30.703   bandwidth:547.99
################################################################################

@gluon.jit
def w4a16_kernel(
        a_ptr, b_ptr, bs_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_bsn, stride_bsk,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
        BLOCK_SIZE_K: gl.constexpr,
        PAD_M: gl.constexpr,
        PAD_N: gl.constexpr,
        PAD_K: gl.constexpr,
        a_cache: gl.constexpr,
        b_cache: gl.constexpr,
        bs_cache: gl.constexpr
):
    """Kernel for computing the matmul C = A x (B x scale).
    A has shape (M, K) with bfloat16 data type
    B has shape (N, K) with mxfp4 data type
    Pack B as (N, K // 2) with uint8 data type
    BS has shape (N, K // 32) with fp8 data type
    C has shape (M, N) with bfloat16 data type
    """

    PACK_B : gl.constexpr = 2
    GROUP_SIZE : gl.constexpr = 32

    blocked_a: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    blocked_b: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )

    blocked_bs: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=False,
        warps_per_cta=[1, 4],
    )

    mfma_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )

    mfma_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[0, 1]
    )
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )
    smem_b = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_K, BLOCK_SIZE_N], layout=shared_b
    )

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)

    if PAD_M:
        offs_am = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_a))) % M
    else:
        offs_am = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_a))
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_a))

    if PAD_N:
        offs_bn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))) % N
    else:
        offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))
    offs_bk = gl.arange(0, BLOCK_SIZE_K // PACK_B, layout=gl.SliceLayout(1, blocked_b))

    if PAD_N:
        offs_bsn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_bs))) % N
    else:
        offs_bsn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_bs))
    offs_bsk = gl.arange(0, BLOCK_SIZE_K // GROUP_SIZE, layout=gl.SliceLayout(1, blocked_bs))

    a_ptr_offs = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_ptr_offs = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    bs_ptr_offs = offs_bsk[:, None] * stride_bsk + offs_bsn[None, :] * stride_bsn

    ############################################################################
    # accumulator
    ############################################################################

    accumulator = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout)

    ############################################################################
    # prologue
    ############################################################################

    bs = gl.amd.cdna3.buffer_load(bs_ptr, bs_ptr_offs)
    b = gl.amd.cdna3.buffer_load(b_ptr, b_ptr_offs)
    a = gl.amd.cdna3.buffer_load(a_ptr, a_ptr_offs)

    a_ptr += BLOCK_SIZE_K * stride_ak
    b_ptr += (BLOCK_SIZE_K // PACK_B) * stride_bk
    bs_ptr += (BLOCK_SIZE_K // GROUP_SIZE) * stride_bsk

    # Convert bs to f32
    bs_u8 = bs.to(gl.uint8, bitcast=True)
    bs_u16 = bs_u8.to(gl.uint32)
    bs_u32 = bs_u16 << 23
    bs_f32 = bs_u32.to(gl.float32, bitcast=True)
    # reshape
    bs_f32 = gl.reshape(bs_f32, (BLOCK_SIZE_K // GROUP_SIZE, 1, BLOCK_SIZE_N))

    # Convert b to f32
    b_bf16 = gl.fp4_to_fp(b, gl.bfloat16, 0)
    b_f32 = b_bf16.to(gl.float32)
    # reshape
    b_f32 = gl.reshape(b_f32, (BLOCK_SIZE_K // GROUP_SIZE, GROUP_SIZE, BLOCK_SIZE_N))

    bs_f32 = gl.convert_layout(bs_f32, b_f32.type.layout)
    b_f32 = bs_f32 * b_f32
    b_u32 = b_f32.to(gl.uint32, bitcast=True)
    b_u32 = b_u32 >> 16
    b_u16 = b_u32.to(gl.uint16)
    b_bf16 = b_u16.to(gl.bfloat16, bitcast=True)
    b = gl.reshape(b_bf16, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    smem_a.store(a)
    smem_b.store(b)

    ############################################################################
    # main loop
    ############################################################################

    num_k_iter = gl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_iter - 1):
        bs = gl.amd.cdna3.buffer_load(bs_ptr, bs_ptr_offs)
        b = gl.amd.cdna3.buffer_load(b_ptr, b_ptr_offs)
        a = gl.amd.cdna3.buffer_load(a_ptr, a_ptr_offs)

        a_converted = smem_a.load(layout=mfma_lhs_layout)
        b_converted = smem_b.load(layout=mfma_rhs_layout)

        # Convert bs to f32
        bs_u8 = bs.to(gl.uint8, bitcast=True)
        bs_u16 = bs_u8.to(gl.uint32)
        bs_u32 = bs_u16 << 23
        bs_f32 = bs_u32.to(gl.float32, bitcast=True)
        # reshape
        bs_f32 = gl.reshape(bs_f32, (BLOCK_SIZE_K // GROUP_SIZE, 1, BLOCK_SIZE_N))

        # Convert b to f32
        b_bf16 = gl.fp4_to_fp(b, gl.bfloat16, 0)
        b_f32 = b_bf16.to(gl.float32)
        # reshape
        b_f32 = gl.reshape(b_f32, (BLOCK_SIZE_K // GROUP_SIZE, GROUP_SIZE, BLOCK_SIZE_N))

        bs_f32 = gl.convert_layout(bs_f32, b_f32.type.layout)
        b_f32 = bs_f32 * b_f32
        b_u32 = b_f32.to(gl.uint32, bitcast=True)
        b_u32 = b_u32 >> 16
        b_u16 = b_u32.to(gl.uint16)
        b_bf16 = b_u16.to(gl.bfloat16, bitcast=True)
        b = gl.reshape(b_bf16, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        smem_a.store(a)
        smem_b.store(b)

        # We accumulate along the K dimension.
        accumulator = gl.amd.cdna3.mfma(a_converted, b_converted, accumulator)

        # Advance the ptrs to the next K block.
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += (BLOCK_SIZE_K // PACK_B) * stride_bk
        bs_ptr += (BLOCK_SIZE_K // GROUP_SIZE) * stride_bsk

    ############################################################################
    # epilogue
    ############################################################################

    a_converted = smem_a.load(layout=mfma_lhs_layout)
    b_converted = smem_b.load(layout=mfma_rhs_layout)

    accumulator = gl.amd.cdna3.mfma(a_converted, b_converted, accumulator)

    ############################################################################
    # end
    ############################################################################
    
    # Convert accumulator to bfloat16
    c = accumulator.to(gl.bfloat16)

    if PAD_M:
        offs_cm = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout))) % M
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout))
    if PAD_N:
        offs_cn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))) % N
    else:
        offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))
    gl.amd.cdna3.buffer_store(c, c_ptr, stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])
'''

'''
################################################################################
# Opt bf16 mul, hack fp4_to_fp direct to f32
# M:15   N:57344 K:8192 BLOCK_SIZE_M:16   BLOCK_SIZE_N:64   BLOCK_SIZE_K:128  num_warps:4    num_stages:2    kpack:1    nonkdim:0    duration:0.38871  tflops:36.255   bandwidth:647.08
################################################################################

@gluon.jit
def w4a16_kernel(
        a_ptr, b_ptr, bs_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_bsn, stride_bsk,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
        BLOCK_SIZE_K: gl.constexpr,
        PAD_M: gl.constexpr,
        PAD_N: gl.constexpr,
        PAD_K: gl.constexpr,
        a_cache: gl.constexpr,
        b_cache: gl.constexpr,
        bs_cache: gl.constexpr
):
    """Kernel for computing the matmul C = A x (B x scale).
    A has shape (M, K) with bfloat16 data type
    B has shape (N, K) with mxfp4 data type
    Pack B as (N, K // 2) with uint8 data type
    BS has shape (N, K // 32) with fp8 data type
    C has shape (M, N) with bfloat16 data type
    """

    PACK_B : gl.constexpr = 2
    GROUP_SIZE : gl.constexpr = 32

    blocked_a: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    blocked_b: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )

    blocked_bs: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=False,
        warps_per_cta=[1, 4],
    )

    mfma_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )

    mfma_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[0, 1]
    )
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )
    smem_b = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_K, BLOCK_SIZE_N], layout=shared_b
    )

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)

    if PAD_M:
        offs_am = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_a))) % M
    else:
        offs_am = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_a))
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_a))

    if PAD_N:
        offs_bn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))) % N
    else:
        offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))
    offs_bk = gl.arange(0, BLOCK_SIZE_K // PACK_B, layout=gl.SliceLayout(1, blocked_b))

    if PAD_N:
        offs_bsn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_bs))) % N
    else:
        offs_bsn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_bs))
    offs_bsk = gl.arange(0, BLOCK_SIZE_K // GROUP_SIZE, layout=gl.SliceLayout(1, blocked_bs))

    a_ptr_offs = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_ptr_offs = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    bs_ptr_offs = offs_bsk[:, None] * stride_bsk + offs_bsn[None, :] * stride_bsn

    ############################################################################
    # accumulator
    ############################################################################

    accumulator = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout)

    ############################################################################
    # prologue
    ############################################################################

    bs = gl.amd.cdna3.buffer_load(bs_ptr, bs_ptr_offs)
    b = gl.amd.cdna3.buffer_load(b_ptr, b_ptr_offs)
    a = gl.amd.cdna3.buffer_load(a_ptr, a_ptr_offs)

    a_ptr += BLOCK_SIZE_K * stride_ak
    b_ptr += (BLOCK_SIZE_K // PACK_B) * stride_bk
    bs_ptr += (BLOCK_SIZE_K // GROUP_SIZE) * stride_bsk

    # Convert bs to f32
    bs_u8 = bs.to(gl.uint8, bitcast=True)
    bs_u16 = bs_u8.to(gl.uint32)
    bs_u32 = bs_u16 << 23
    bs_f32 = bs_u32.to(gl.float32, bitcast=True)
    # reshape
    bs_f32 = gl.reshape(bs_f32, (BLOCK_SIZE_K // GROUP_SIZE, 1, BLOCK_SIZE_N))

    # Convert b to f32
    b_f32 = gl.fp4_to_fp(b, gl.float32, 0)
    # reshape
    b_f32 = gl.reshape(b_f32, (BLOCK_SIZE_K // GROUP_SIZE, GROUP_SIZE, BLOCK_SIZE_N))

    bs_f32 = gl.convert_layout(bs_f32, b_f32.type.layout)
    b_f32 = bs_f32 * b_f32
    b_u32 = b_f32.to(gl.uint32, bitcast=True)
    b_u32 = b_u32 >> 16
    b_u16 = b_u32.to(gl.uint16)
    b_bf16 = b_u16.to(gl.bfloat16, bitcast=True)
    b = gl.reshape(b_bf16, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    smem_a.store(a)
    smem_b.store(b)

    ############################################################################
    # main loop
    ############################################################################

    num_k_iter = gl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_iter - 1):
        bs = gl.amd.cdna3.buffer_load(bs_ptr, bs_ptr_offs)
        b = gl.amd.cdna3.buffer_load(b_ptr, b_ptr_offs)
        a = gl.amd.cdna3.buffer_load(a_ptr, a_ptr_offs)

        a_converted = smem_a.load(layout=mfma_lhs_layout)
        b_converted = smem_b.load(layout=mfma_rhs_layout)

        # Convert bs to f32
        bs_u8 = bs.to(gl.uint8, bitcast=True)
        bs_u16 = bs_u8.to(gl.uint32)
        bs_u32 = bs_u16 << 23
        bs_f32 = bs_u32.to(gl.float32, bitcast=True)
        # reshape
        bs_f32 = gl.reshape(bs_f32, (BLOCK_SIZE_K // GROUP_SIZE, 1, BLOCK_SIZE_N))

        # Convert b to f32
        b_f32 = gl.fp4_to_fp(b, gl.float32, 0)
        # reshape
        b_f32 = gl.reshape(b_f32, (BLOCK_SIZE_K // GROUP_SIZE, GROUP_SIZE, BLOCK_SIZE_N))

        bs_f32 = gl.convert_layout(bs_f32, b_f32.type.layout)
        b_f32 = bs_f32 * b_f32
        b_u32 = b_f32.to(gl.uint32, bitcast=True)
        b_u32 = b_u32 >> 16
        b_u16 = b_u32.to(gl.uint16)
        b_bf16 = b_u16.to(gl.bfloat16, bitcast=True)
        b = gl.reshape(b_bf16, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        smem_a.store(a)
        smem_b.store(b)

        # We accumulate along the K dimension.
        accumulator = gl.amd.cdna3.mfma(a_converted, b_converted, accumulator)

        # Advance the ptrs to the next K block.
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += (BLOCK_SIZE_K // PACK_B) * stride_bk
        bs_ptr += (BLOCK_SIZE_K // GROUP_SIZE) * stride_bsk

    ############################################################################
    # epilogue
    ############################################################################

    a_converted = smem_a.load(layout=mfma_lhs_layout)
    b_converted = smem_b.load(layout=mfma_rhs_layout)

    accumulator = gl.amd.cdna3.mfma(a_converted, b_converted, accumulator)

    ############################################################################
    # end
    ############################################################################
    
    # Convert accumulator to bfloat16
    c = accumulator.to(gl.bfloat16)

    if PAD_M:
        offs_cm = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout))) % M
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout))
    if PAD_N:
        offs_cn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))) % N
    else:
        offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))
    gl.amd.cdna3.buffer_store(c, c_ptr, stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])
'''

'''
################################################################################
# Opt st b w/ origin data type, then upcast to bfloat16
# M:15   N:57344 K:8192 BLOCK_SIZE_M:16   BLOCK_SIZE_N:64   BLOCK_SIZE_K:128  num_warps:4    num_stages:2    kpack:1    nonkdim:0    duration:0.33406  tflops:42.187   bandwidth:752.94
################################################################################

@gluon.jit
def w4a16_kernel(
        a_ptr, b_ptr, bs_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_bsn, stride_bsk,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
        BLOCK_SIZE_K: gl.constexpr,
        PAD_M: gl.constexpr,
        PAD_N: gl.constexpr,
        PAD_K: gl.constexpr,
        a_cache: gl.constexpr,
        b_cache: gl.constexpr,
        bs_cache: gl.constexpr
):
    """Kernel for computing the matmul C = A x (B x scale).
    A has shape (M, K) with bfloat16 data type
    B has shape (N, K) with mxfp4 data type
    Pack B as (N, K // 2) with uint8 data type
    BS has shape (N, K // 32) with fp8 data type
    C has shape (M, N) with bfloat16 data type
    """

    PACK_B : gl.constexpr = 2
    GROUP_SIZE : gl.constexpr = 32

    blocked_a: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    blocked_b: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )

    blocked_bs: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[1, 4],
    )

    mfma_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )

    mfma_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    mfma_b_fp4_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )

    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[0, 1]
    )
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )

    smem_b_fp4 = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [BLOCK_SIZE_K // PACK_B, BLOCK_SIZE_N], layout=shared_b
    )

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)

    if PAD_M:
        offs_am = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_a))) % M
    else:
        offs_am = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_a))
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_a))

    if PAD_N:
        offs_bn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))) % N
    else:
        offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))
    offs_bk = gl.arange(0, BLOCK_SIZE_K // PACK_B, layout=gl.SliceLayout(1, blocked_b))

    if PAD_N:
        offs_bsn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_bs))) % N
    else:
        offs_bsn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_bs))
    offs_bsk = gl.arange(0, BLOCK_SIZE_K // GROUP_SIZE, layout=gl.SliceLayout(1, blocked_bs))

    a_ptr_offs = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_ptr_offs = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    bs_ptr_offs = offs_bsk[:, None] * stride_bsk + offs_bsn[None, :] * stride_bsn

    ############################################################################
    # accumulator
    ############################################################################

    accumulator = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout)

    ############################################################################
    # main loop
    ############################################################################

    num_k_iter = gl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_iter):
        bs = gl.amd.cdna3.buffer_load(bs_ptr, bs_ptr_offs)
        b_fp4 = gl.amd.cdna3.buffer_load(b_ptr, b_ptr_offs)
        a = gl.amd.cdna3.buffer_load(a_ptr, a_ptr_offs)

        smem_a.store(a)
        smem_b_fp4.store(b_fp4)

        # Convert bs to f32
        bs_u8 = bs.to(gl.uint8, bitcast=True)
        bs_u16 = bs_u8.to(gl.uint32)
        bs_u32 = bs_u16 << 23
        bs_f32 = bs_u32.to(gl.float32, bitcast=True)
        # reshape
        bs_f32 = gl.reshape(bs_f32, (BLOCK_SIZE_K // GROUP_SIZE, 1, BLOCK_SIZE_N))

        b_fp4 = smem_b_fp4.load(layout=mfma_b_fp4_layout)

        # Convert b to f32
        b_f32 = gl.fp4_to_fp(b_fp4, gl.float32, 0)
        # reshape
        b_f32 = gl.reshape(b_f32, (BLOCK_SIZE_K // GROUP_SIZE, GROUP_SIZE, BLOCK_SIZE_N))

        bs_f32 = gl.convert_layout(bs_f32, b_f32.type.layout)
        b_f32 = bs_f32 * b_f32
        b_u32 = b_f32.to(gl.uint32, bitcast=True)
        b_u32 = b_u32 >> 16
        b_u16 = b_u32.to(gl.uint16)
        b_bf16 = b_u16.to(gl.bfloat16, bitcast=True)
        b = gl.reshape(b_bf16, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        a_converted = smem_a.load(layout=mfma_a_layout)
        b_converted = gl.convert_layout(b, mfma_b_layout)
        
        # We accumulate along the K dimension.
        accumulator = gl.amd.cdna3.mfma(a_converted, b_converted, accumulator)

        # Advance the ptrs to the next K block.
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += (BLOCK_SIZE_K // PACK_B) * stride_bk
        bs_ptr += (BLOCK_SIZE_K // GROUP_SIZE) * stride_bsk

    ############################################################################
    # end
    ############################################################################
    
    # Convert accumulator to bfloat16
    c = accumulator.to(gl.bfloat16)

    if PAD_M:
        offs_cm = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout))) % M
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout))
    if PAD_N:
        offs_cn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))) % N
    else:
        offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))
    gl.amd.cdna3.buffer_store(c, c_ptr, stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])
'''


################################################################################
# Opt shared memory & mha layout
# M:15   N:57344 K:8192 BLOCK_SIZE_M:16   BLOCK_SIZE_N:64   BLOCK_SIZE_K:128  num_warps:4    num_stages:2    kpack:1    nonkdim:0    duration:0.31478  tflops:44.770   bandwidth:799.05
# Disable SLP
# M:15   N:57344 K:8192 BLOCK_SIZE_M:16   BLOCK_SIZE_N:64   BLOCK_SIZE_K:128  num_warps:4    num_stages:2    kpack:1    nonkdim:0    duration:0.27943  tflops:50.435   bandwidth:900.16
################################################################################

@gluon.jit
def w4a16_kernel(
        a_ptr, b_ptr, bs_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_bsn, stride_bsk,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
        BLOCK_SIZE_K: gl.constexpr,
        PAD_M: gl.constexpr,
        PAD_N: gl.constexpr,
        PAD_K: gl.constexpr,
        a_cache: gl.constexpr,
        b_cache: gl.constexpr,
        bs_cache: gl.constexpr
):
    """Kernel for computing the matmul C = A x (B x scale).
    A has shape (M, K) with bfloat16 data type
    B has shape (N, K) with mxfp4 data type
    Pack B as (N, K // 2) with uint8 data type
    BS has shape (N, K // 32) with fp8 data type
    C has shape (M, N) with bfloat16 data type
    """

    PACK_B : gl.constexpr = 2
    GROUP_SIZE : gl.constexpr = 32

    blocked_a: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    blocked_b: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )

    blocked_bs: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[1, 4],
    )

    mfma_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )

    mfma_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )

    mfma_b_fp4_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=4
    )

    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=2, max_phase=8, order=[0, 1]
    )
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )

    smem_b_fp4 = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [BLOCK_SIZE_K // PACK_B, BLOCK_SIZE_N], layout=shared_b
    )

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)

    if PAD_M:
        offs_am = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_a))) % M
    else:
        offs_am = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_a))
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_a))

    if PAD_N:
        offs_bn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))) % N
    else:
        offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))
    offs_bk = gl.arange(0, BLOCK_SIZE_K // PACK_B, layout=gl.SliceLayout(1, blocked_b))

    if PAD_N:
        offs_bsn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_bs))) % N
    else:
        offs_bsn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_bs))
    offs_bsk = gl.arange(0, BLOCK_SIZE_K // GROUP_SIZE, layout=gl.SliceLayout(1, blocked_bs))

    a_ptr_offs = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_ptr_offs = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    bs_ptr_offs = offs_bsk[:, None] * stride_bsk + offs_bsn[None, :] * stride_bsn

    ############################################################################
    # accumulator
    ############################################################################

    accumulator = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout)


    ############################################################################
    # prologue
    ############################################################################
    a = gl.amd.cdna3.buffer_load(a_ptr, a_ptr_offs)
    b_fp4 = gl.amd.cdna3.buffer_load(b_ptr, b_ptr_offs)

    a_ptr += BLOCK_SIZE_K * stride_ak
    b_ptr += (BLOCK_SIZE_K // PACK_B) * stride_bk

    smem_a.store(a)
    smem_b_fp4.store(b_fp4)

    ############################################################################
    # main loop
    ############################################################################

    num_k_iter = gl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_iter - 1):
        bs = gl.amd.cdna3.buffer_load(bs_ptr, bs_ptr_offs)
        b_fp4 = gl.amd.cdna3.buffer_load(b_ptr, b_ptr_offs)
        a = gl.amd.cdna3.buffer_load(a_ptr, a_ptr_offs)

        # load b
        b_fp4_converted = smem_b_fp4.load(layout=mfma_b_fp4_layout)
        # load a
        a_converted = smem_a.load(layout=mfma_a_layout)

        # Convert bs to f32
        bs_u8 = bs.to(gl.uint8, bitcast=True)
        bs_u16 = bs_u8.to(gl.uint32)
        bs_u32 = bs_u16 << 23
        bs_f32 = bs_u32.to(gl.float32, bitcast=True)
        # reshape
        bs_f32 = gl.reshape(bs_f32, (BLOCK_SIZE_K // GROUP_SIZE, 1, BLOCK_SIZE_N))


        # Convert b to f32
        b_f32 = gl.fp4_to_fp(b_fp4_converted, gl.float32, 0)
        # reshape
        b_f32 = gl.reshape(b_f32, (BLOCK_SIZE_K // GROUP_SIZE, GROUP_SIZE, BLOCK_SIZE_N))

        bs_f32 = gl.convert_layout(bs_f32, b_f32.type.layout)
        b_f32 = bs_f32 * b_f32
        b_u32 = b_f32.to(gl.uint32, bitcast=True)
        b_u32 = b_u32 >> 16
        b_u16 = b_u32.to(gl.uint16)
        b_bf16 = b_u16.to(gl.bfloat16, bitcast=True)
        b = gl.reshape(b_bf16, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        b_converted = gl.convert_layout(b, mfma_b_layout)

        smem_b_fp4.store(b_fp4)
        smem_a.store(a)
        
        # We accumulate along the K dimension.
        accumulator = gl.amd.cdna3.mfma(a_converted, b_converted, accumulator)

        # Advance the ptrs to the next K block.
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += (BLOCK_SIZE_K // PACK_B) * stride_bk
        bs_ptr += (BLOCK_SIZE_K // GROUP_SIZE) * stride_bsk

    ############################################################################
    # epilogue
    ############################################################################

    bs = gl.amd.cdna3.buffer_load(bs_ptr, bs_ptr_offs)

    # load b
    b_fp4_converted = smem_b_fp4.load(layout=mfma_b_fp4_layout)
    # load a
    a_converted = smem_a.load(layout=mfma_a_layout)

    # Convert bs to f32
    bs_u8 = bs.to(gl.uint8, bitcast=True)
    bs_u16 = bs_u8.to(gl.uint32)
    bs_u32 = bs_u16 << 23
    bs_f32 = bs_u32.to(gl.float32, bitcast=True)
    # reshape
    bs_f32 = gl.reshape(bs_f32, (BLOCK_SIZE_K // GROUP_SIZE, 1, BLOCK_SIZE_N))

    # Convert b to f32
    b_f32 = gl.fp4_to_fp(b_fp4_converted, gl.float32, 0)
    # reshape
    b_f32 = gl.reshape(b_f32, (BLOCK_SIZE_K // GROUP_SIZE, GROUP_SIZE, BLOCK_SIZE_N))

    # b * bs
    bs_f32 = gl.convert_layout(bs_f32, b_f32.type.layout)
    b_f32 = bs_f32 * b_f32
    b_u32 = b_f32.to(gl.uint32, bitcast=True)
    b_u32 = b_u32 >> 16
    b_u16 = b_u32.to(gl.uint16)
    b_bf16 = b_u16.to(gl.bfloat16, bitcast=True)
    b = gl.reshape(b_bf16, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    b_converted = gl.convert_layout(b, mfma_b_layout)

    # We accumulate along the K dimension.
    accumulator = gl.amd.cdna3.mfma(a_converted, b_converted, accumulator)

    ############################################################################
    # end
    ############################################################################
    
    # Convert accumulator to bfloat16
    c = accumulator.to(gl.bfloat16)

    if PAD_M:
        offs_cm = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout))) % M
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout))
    if PAD_N:
        offs_cn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))) % N
    else:
        offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))
    gl.amd.cdna3.buffer_store(c, c_ptr, stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])



def w4a16(a, b, bs,
          c = None,
          BLOCK_SIZE_M = 64,
          BLOCK_SIZE_N = 64,
          BLOCK_SIZE_K = 64,
          num_warps=4,
          num_stages=2,
          kpack=1,
          matrix_instr_nonkdim=0,
          a_cache="",
          b_cache="",
          bs_cache=""):
    assert a.shape[1] == b.shape[1] * PACK_B, "Incompatible dimensions"
    assert a.shape[1] == bs.shape[1] * GROUP_SIZE, "Incompatible dimensions"
    assert b.shape[0] == bs.shape[0], "Incompatible dimensions"

    assert a.is_contiguous(), "a must be contiguous"
    assert b.is_contiguous(), "b must be contiguous"
    assert bs.is_contiguous(), "bs must be contiguous"

    M, K = a.shape
    N, _ = b.shape

    # Allocates output.
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    w4a16_kernel[grid](
        a, b, bs, c,                #
        M, N, K,                    #
        a.stride(0), a.stride(1),   #
        b.stride(0), b.stride(1),   #
        bs.stride(0), bs.stride(1), #
        c.stride(0), c.stride(1),   #
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_K = BLOCK_SIZE_K,
        PAD_M = M % BLOCK_SIZE_M != 0,
        PAD_N = N % BLOCK_SIZE_N != 0,
        PAD_K = K % BLOCK_SIZE_K != 0,
        a_cache = a_cache,
        b_cache = b_cache,
        bs_cache = bs_cache,
        num_warps = num_warps,
        num_stages = num_stages,
        kpack = kpack,
        matrix_instr_nonkdim = matrix_instr_nonkdim
    )
    return c

# %%
# Unit Test

def generate_test_data(M, N, K):
    # A operand
    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")

    # B operand
    b_mxfp4 = MXFP4Tensor(size=(N, K), device="cuda").random()
    b = b_mxfp4.to_packed_tensor(dim=1)
    b_scale_ref = MXScaleTensor(torch.rand([N, K // GROUP_SIZE], device="cuda") + 1e-8)
    b_scale = b_scale_ref.data

    # ref
    b_scale_ref = b_scale_ref.to(torch.float32).repeat_interleave(GROUP_SIZE, dim=1).contiguous()[:N, :K]
    b_ref = b_mxfp4.to(torch.float32) * b_scale_ref

    return a, b, b_scale, b_ref

def function_test(M, N, K):
    print("test w4a16 with M={} N={} K={}".format(M, N, K))
    a, b, b_scale, b_ref = generate_test_data(512, 512, 512)
    triton_out = w4a16(a, b, b_scale)
    torch_out = torch.matmul(a.to(torch.float32), b_ref.T).to(torch.bfloat16)

    # Check
    torch.testing.assert_close(triton_out, torch_out, atol=5e-2, rtol=5e-2)
    print("✅ Triton and Torch match")

# %%
# Performance Test

class W4A16PerfBenchmark:
    INVOCATIONS = 2
    TEST_TIMES = 5
    REPEAT = 10

    def __init__(self, M, N, K, 
                 BLOCK_SIZE_M,
                 BLOCK_SIZE_N,
                 BLOCK_SIZE_K,
                 num_warps,
                 num_stages,
                 kpack,
                 matrix_instr_nonkdim,
                 a_cache="",
                 b_cache="",
                 bs_cache=""):
        self.M = M
        self.N = N
        self.K = K
        self.BLOCK_SIZE_M = BLOCK_SIZE_M
        self.BLOCK_SIZE_N = BLOCK_SIZE_N
        self.BLOCK_SIZE_K = BLOCK_SIZE_K
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.kpack = kpack
        self.matrix_instr_nonkdim = matrix_instr_nonkdim
        self.a_cache = a_cache
        self.b_cache = b_cache
        self.bs_cache = bs_cache
        self.a, self.b, self.b_scale, self.b_ref = generate_test_data(M, N, K)
        self.c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

        self._setup_cuda_graph()

    def _run_once(self):
        w4a16(self.a, self.b, self.b_scale,
              c = self.c,
              BLOCK_SIZE_M = self.BLOCK_SIZE_M,
              BLOCK_SIZE_N = self.BLOCK_SIZE_N,
              BLOCK_SIZE_K = self.BLOCK_SIZE_K,
              a_cache = self.a_cache,
              b_cache = self.b_cache,
              bs_cache = self.bs_cache,
              num_warps = self.num_warps,
              num_stages = self.num_stages,
              kpack = self.kpack,
              matrix_instr_nonkdim = self.matrix_instr_nonkdim)

    def _setup_cuda_graph(self):
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for _ in range(self.INVOCATIONS):
                self._run_once()

    def run_benchmark(self):
        if self.graph is None:
            raise RuntimeError("CUDA graph not set up. Call setup_cuda_graph() first.")
        self.graph.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(self.TEST_TIMES):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(self.REPEAT):
                self.graph.replay()
            end_event.record()
            torch.cuda.synchronize()
            iteration_time = start_event.elapsed_time(end_event)
            times.append(iteration_time / self.REPEAT)

        # ms
        time = (sum(times) - max(times) - min(times)) / (self.TEST_TIMES - 2) / self.INVOCATIONS
        self._print(time)

    def _mem(self):
        return self.a.numel() * self.a.element_size() + \
               self.b.numel() * self.b.element_size() + \
               self.b_scale.numel() * self.b_scale.element_size() + \
               self.c.numel() * self.c.element_size()

    def _flops(self):
        return 2.0 * self.M * self.N * self.K

    def _print(self, ms):
        flops = self._flops()
        mem = self._mem()

        tflops = flops / (ms * 1e-3) / 1e12 # TFLOPS
        bandwidth = mem / (ms * 1e-3) / 1e9 # GB/s
        print(f"M:{self.M:<4} N:{self.N:<4} K:{self.K:<4} BLOCK_SIZE_M:{self.BLOCK_SIZE_M:<4} BLOCK_SIZE_N:{self.BLOCK_SIZE_N:<4} BLOCK_SIZE_K:{self.BLOCK_SIZE_K:<4} num_warps:{self.num_warps:<4} num_stages:{self.num_stages:<4} kpack:{self.kpack:<4} nonkdim:{self.matrix_instr_nonkdim:<4} duration:{ms:<7.5f}  tflops:{tflops:<7.3f}  bandwidth:{bandwidth:<7.2f}")

def perf_test(M, N, K):
    # cache = ["", ".ca", ".cg", ".cv"]
    # for a_cache in cache:
    #     for b_cache in cache:
    #         for bs_cache in cache:
    #             print("a_cache:{}, b_cache:{}, bs_cache".format(a_cache, b_cache, bs_cache))
    #             bench = W4A16PerfBenchmark(M, N, K, 16, 64, 128, 4, 2, 1, 0, a_cache=a_cache, b_cache=b_cache, bs_cache=bs_cache)
    #             bench.run_benchmark()
    bench = W4A16PerfBenchmark(M, N, K, 16, 64, 128, 4, 2, 1, 0)
    bench.run_benchmark()

if __name__ == "__main__":
    function_test(512, 512, 512)
    perf_test(15, 57344, 8192)