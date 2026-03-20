# 降低寄存器压力 (Reduce Register Pressure) 深度指南

> 面向 LLM 高性能 Kernel 开发的寄存器压力分析、优化策略与实战模式
> 覆盖寄存器文件架构、Occupancy 关系、Spilling 机制、编译器控制、Warp Specialization、TMEM、CUDA 13.0 Shared Memory Spilling

---

## 目录

1. [寄存器压力概述](#1-寄存器压力概述)
2. [寄存器文件硬件架构](#2-寄存器文件硬件架构)
3. [寄存器与 Occupancy 的关系](#3-寄存器与-occupancy-的关系)
4. [寄存器 Spilling 机制](#4-寄存器-spilling-机制)
5. [编译器控制：`__launch_bounds__` 与 `--maxrregcount`](#5-编译器控制launch_bounds-与---maxrregcount)
6. [代码级优化策略](#6-代码级优化策略)
7. [MMA 指令的寄存器开销](#7-mma-指令的寄存器开销)
8. [Hopper: Warp Specialization 与 `setmaxnreg`](#8-hopper-warp-specialization-与-setmaxnreg)
9. [Blackwell: Tensor Memory (TMEM) 解放寄存器](#9-blackwell-tensor-memory-tmem-解放寄存器)
10. [CUDA 13.0: Shared Memory Register Spilling](#10-cuda-130-shared-memory-register-spilling)
11. [寄存器 Bank Conflict 与 Operand Reuse](#11-寄存器-bank-conflict-与-operand-reuse)
12. [LLM Kernel 实战：寄存器优化模式](#12-llm-kernel-实战寄存器优化模式)
13. [诊断与分析工具](#13-诊断与分析工具)
14. [优化检查清单](#14-优化检查清单)

---

## 1. 寄存器压力概述

### 1.1 什么是寄存器压力

**寄存器压力 (Register Pressure)** 是指一个 kernel 所需的寄存器数量超出硬件能够高效分配的范围，导致以下后果之一或兼有：

```
寄存器需求过高
├── 后果 1: Occupancy 下降 (SM 上能驻留的 Warp 数减少)
│           → 延迟隐藏能力下降 → 性能降低
│
├── 后果 2: 寄存器 Spilling (编译器将变量溢出到 Local Memory)
│           → 额外的 load/store 指令 → 性能降低
│
└── 后果 3: 两者同时发生 (最严重)
```

### 1.2 为什么寄存器压力是 LLM Kernel 的核心挑战

| LLM Kernel 类型 | 寄存器压力来源 | 严重程度 |
|----------------|-------------|---------|
| **GEMM (Tensor Core)** | MMA 累加器 + 输入 Fragment + 双缓冲 | 极高 (100–200+ regs/thread) |
| **FlashAttention** | GEMM 累加器 + Online Softmax 状态 (max, sum) + 多阶段流水线 | 极高 |
| **Fused Kernel (SwiGLU + Norm + Residual)** | 多个输入/输出向量 + 中间结果 | 中高 (60–100 regs) |
| **RMSNorm / LayerNorm** | 向量化加载的临时寄存器 + reduction 中间值 | 中等 (40–64 regs) |
| **Quantization / Dequantization** | 打包/解包的临时变量 + scale 因子 | 中等 |

### 1.3 延迟层级：寄存器为何如此重要

```
延迟层级 (近似时钟周期):
  寄存器 (Register File)     ~1 cycle
  Shared Memory              ~20–30 cycles    (20–30×)
  L1 Cache 命中              ~28–35 cycles    (28–35×)
  L2 Cache 命中              ~150–200 cycles  (150–200×)
  HBM (Global / Local Memory) ~400–800 cycles (400–800×)
```

寄存器是 GPU 上唯一能在**单周期**内提供数据的存储层次。任何从寄存器 "溢出" 到更慢层次的数据都会显著增加指令延迟。

---

## 2. 寄存器文件硬件架构

### 2.1 各架构寄存器文件规格

| 架构 | CC | 寄存器/SM (32-bit) | RF 大小/SM | 最大 Warp/SM | 最大 Reg/线程 |
|------|:--:|:-----------------:|:---------:|:----------:|:-----------:|
| Pascal | 6.0 | 65,536 | 256 KB | 64 | 255 |
| Volta | 7.0 | 65,536 | 256 KB | 64 | 255 |
| Turing | 7.5 | 65,536 | 256 KB | 32 | 255 |
| Ampere (GA100) | 8.0 | 65,536 | 256 KB | 64 | 255 |
| Ampere (GA10x) | 8.6 | 65,536 | 256 KB | 48 | 255 |
| Ada | 8.9 | 65,536 | 256 KB | 48 | 255 |
| Hopper | 9.0 | 65,536 | 256 KB | 64 | 255 |
| Blackwell (DC) | 10.0 | 65,536 | 256 KB | 64 | 255 |
| Blackwell (消费级) | 12.0 | 65,536 | 256 KB | 48 | 255 |

> **十年不变：** 寄存器文件大小自 Kepler 以来始终为 65,536 × 32-bit = 256 KB/SM。但 Tensor Core 吞吐每代翻倍，使得 "喂饱 Tensor Core" 所需的寄存器预算持续增长，寄存器压力日益严峻。

### 2.2 寄存器分配粒度

寄存器分配涉及**两级舍入**，导致实际利用率低于理论值：

```
Level 1: 寄存器按 Warp 分配, 向上舍入到 256 的倍数 (CC 7.0+)
  例: 每线程 33 regs → 每 Warp 33 × 32 = 1056 regs → 向上舍入到 1280 regs
      (浪费 224 / 1280 = 17.5%)

Level 2: Warp 数向下舍入到 4 的倍数 (Warp Allocation Granularity)
  例: 65536 / 1280 = 51.2 → 向下到 48 Warps
      (浪费 3.2 / 51.2 = 6.25%)

两级舍入导致平均 ~12% 的寄存器文件利用率损失
```

**关键公式：**

```
实际可用 Warp 数 = floor(floor(65536 / ceil(regs_per_thread × 32 / 256) × 256) / 4) × 4
```

### 2.3 64-bit 值的寄存器开销

```
类型          | 占用寄存器数 (32-bit regs)
-------------|-------------------------
int32 / float | 1
int64 / double| 2                        ← 常见的隐性压力来源
指针 (64-bit) | 2                        ← GPU 上所有指针都是 64-bit
float4        | 4                        ← 向量化加载的临时开销
```

> **陷阱：** 使用 `int64` 做循环变量或数组索引会无声地翻倍寄存器消耗。threadIdx.x 和 blockIdx.x 是 32-bit，但与 64-bit 字面量运算会隐式提升。

---

## 3. 寄存器与 Occupancy 的关系

### 3.1 Occupancy 基础

**Occupancy** = SM 上活跃 Warp 数 / SM 支持的最大 Warp 数

影响 Occupancy 的三大资源限制：

```
Occupancy 瓶颈
├── 1. 寄存器 — 每线程 regs × 线程数 ≤ 65,536
├── 2. Shared Memory — 每 Block smem ≤ SM 可用 smem
└── 3. Thread/Warp Slots — 每 SM 最大 Block 数 / Warp 数
→ Occupancy = min(三个限制) / max_warps
```

### 3.2 寄存器导致的 Occupancy 阶梯效应

由于分配粒度为 256 regs/Warp，寄存器-Occupancy 关系呈**阶梯状**：

```
CC 8.0 (A100): 65,536 regs/SM, 64 max warps

Regs/Thread | Regs/Warp (原始) | 舍入到 256 | 可用 Warp | Occupancy
     ≤ 32   |      1024       |    1024    |    64     |   100%
     33      |      1056       |    1280    |    48     |    75%    ← 1 个寄存器 = -25%!
     40      |      1280       |    1280    |    48     |    75%
     41      |      1312       |    1536    |    40     |   62.5%
     48      |      1536       |    1536    |    40     |   62.5%
     49      |      1568       |    1792    |    36     |   56.3%
     64      |      2048       |    2048    |    32     |    50%
     80      |      2560       |    2560    |    24     |   37.5%
     96      |      3072       |    3072    |    20     |   31.3%
    128      |      4096       |    4096    |    16     |    25%
    192      |      6144       |    6144    |     8     |   12.5%
    255      |      8160       |    8192    |     8     |   12.5%
```

**关键洞察：** 从 32 到 33 regs/thread，Occupancy 从 100% 断崖下降到 75%。但从 33 到 40 regs 没有任何变化。**优化寄存器使用时，需要瞄准舍入边界，而非盲目减少 1–2 个寄存器。**

### 3.3 Occupancy 并非越高越好

```
Occupancy 与性能的真实关系:

低 Occupancy (< 25%):
  → 几乎总是性能问题
  → 不足以隐藏内存延迟
  → Warp Scheduler 经常空闲

中 Occupancy (25–50%):
  → 对计算密集 Kernel (GEMM) 通常足够
  → 更多寄存器/线程 → 更大的 tile → 更高的算术强度
  → 减少 Shared Memory / L1 Cache 争用

高 Occupancy (50–100%):
  → 对内存密集 Kernel (elementwise) 有利
  → 但更多活跃 Warp → 更少寄存器/线程 → 可能导致 spilling
  → Cache 压力增大 (更多线程竞争 L1/L2)
```

> **经验法则：** GEMM 类 kernel 在 25–50% occupancy 达到峰值性能；Elementwise kernel 在 50–75% 达到峰值。盲目追求 100% occupancy 通常适得其反。

---

## 4. 寄存器 Spilling 机制

### 4.1 什么是 Spilling

当 kernel 所需寄存器超过编译器的预算 (由 `__launch_bounds__` 或 `--maxrregcount` 决定)，编译器将部分变量 **溢出 (spill)** 到 Local Memory：

```
正常变量生命周期:     寄存器 → 计算 → 寄存器       (~1 cycle)
Spilled 变量生命周期: 寄存器 → STL (存到 Local) → ... → LDL (从 Local 读回) → 计算

STL: Store to Local memory (SASS 指令)
LDL: Load from Local memory (SASS 指令)
```

### 4.2 Local Memory 的物理位置

```
Local Memory 不是独立的物理内存!

物理位置: GPU DRAM (与 Global Memory 相同的 HBM)
缓存路径:
  CC 7.0+: L1 (unified with Shared Memory) → L2 → DRAM
  CC 5.x/6.x: L2 → DRAM (不经过 L1)

实际延迟 (取决于缓存命中):
  L1 命中: ~28–35 cycles (Spill 常常命中, 因为栈式访问模式)
  L2 命中: ~150–200 cycles
  DRAM:    ~400–800 cycles
```

> **Spill 并非世界末日。** 由于 spill 访问具有良好的空间局部性 (栈式后进先出)，L1 命中率通常很高，有效延迟约 30 cycles 而非 DRAM 的 400+ cycles。但每次 spill 仍然增加 2 条额外指令 (STL + LDL)，增加指令流水线压力并占用 L1 cache 容量。

### 4.3 Spill 的性能影响

```
Spill 的三重代价:

1. 延迟代价: ~30 cycles (L1 命中) vs ~1 cycle (寄存器) → 30× 慢
2. 指令代价: 每个 spilled 变量增加 STL + LDL 共 2 条指令
3. Cache 代价: Spilled 数据占用 L1/L2 cache 行, 可能驱逐有用的 Global Memory 数据

实测案例 (NVIDIA 官方 benchmark, Fermi):
  带 spill (44 bytes/thread): kernel_time = X
  无 spill (46 regs/thread):  kernel_time = X / 1.22 (提速 22%)
  Spill 指令仅占 4.6% 的总指令数, 但性能差 22%
```

### 4.4 判断 Spill 是否可接受

```
可接受的 Spill:
  - spill 量 < 16 bytes/thread
  - spill 指令占总指令 < 5%
  - 换取的 Occupancy 提升 > 25%

不可接受的 Spill:
  - spill 量 > 64 bytes/thread
  - spill 指令在 hot loop 内
  - spill 导致 L1 cache 命中率下降 (驱逐有用数据)
```

---

## 5. 编译器控制：`__launch_bounds__` 与 `--maxrregcount`

### 5.1 `__launch_bounds__`

**作用：** 告知编译器 kernel 的启动配置，使其计算出最优的寄存器预算。

```cpp
__global__ void __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)
myKernel(...) {
    // ...
}

// 参数说明:
// maxThreadsPerBlock (必需): kernel 启动时每 block 的最大线程数
// minBlocksPerSM (可选): 期望每 SM 驻留的最少 block 数
```

**编译器行为：**

```
给定: __launch_bounds__(256, 2), CC 8.0 (65536 regs, 64 max warps)

Step 1: 计算目标 warp 数
  2 blocks × 256 threads = 512 threads = 16 warps

Step 2: 计算寄存器预算
  65536 regs / 16 warps / 32 threads = 128 regs/thread

Step 3: 编译器目标
  生成的代码尽量不超过 128 regs/thread
  如果 kernel 自然需要 80 regs → 不 spill, 保持 80 regs
  如果 kernel 自然需要 150 regs → spill 到 128 regs
```

**不同 `minBlocksPerSM` 的效果：**

```cpp
// 宽松: 至少 1 block/SM → 预算 = 65536/8/32 = 256 → 最多 255 regs (实际上限)
__launch_bounds__(256, 1)

// 中等: 至少 2 blocks/SM → 预算 = 128 regs/thread
__launch_bounds__(256, 2)

// 激进: 至少 4 blocks/SM → 预算 = 64 regs/thread → 可能大量 spilling
__launch_bounds__(256, 4)
```

### 5.2 `--maxrregcount`

**作用：** 全局 (per-file) 寄存器限制。

```bash
# 限制所有 kernel 最多使用 64 个寄存器
nvcc --maxrregcount=64 kernel.cu

# 或通过 ptxas 选项
nvcc -Xptxas=-maxrregcount=64 kernel.cu
```

### 5.3 两者对比与推荐

| 特性 | `__launch_bounds__` | `--maxrregcount` |
|------|:------------------:|:----------------:|
| 作用范围 | 单个 kernel | 整个编译单元 (文件) |
| Occupancy 感知 | ✅ (根据 block 大小计算) | ❌ (硬性上限) |
| 灵活性 | 高 (每个 kernel 独立) | 低 (一刀切) |
| 推荐场景 | 生产代码 | 快速实验/调试 |
| 不指定时的默认行为 | 假设最大 block size (通常 1024) | 无限制 (最多 255) |

**最佳实践：**

```cpp
// ✅ 推荐: 每个 kernel 使用 __launch_bounds__
__global__ void __launch_bounds__(256, 2) gemm_kernel(...) { ... }
__global__ void __launch_bounds__(128, 4) rmsnorm_kernel(...) { ... }

// ⚠️ 不推荐: 全局限制可能对某些 kernel 过松、对另一些过紧
// nvcc --maxrregcount=64 all_kernels.cu
```

### 5.4 不指定 Launch Bounds 的隐患

```
不指定 __launch_bounds__ 时:

ptxas 假设 maxThreadsPerBlock = 1024 (最大值)
→ 假设每 SM 可能有 2048/1024 = 2 blocks (最保守)
→ 编译器可能过度优化寄存器使用, 导致不必要的 spilling

或者:
→ 编译器完全不限制寄存器 → kernel 用 200+ regs
→ 运行时 occupancy 极低

两种情况都不理想。始终指定 __launch_bounds__。
```

### 5.5 编译器警告

```bash
# CUDA 12.0+: 启用缺少 launch_bounds 警告
nvcc --Wmissing-launch-bounds kernel.cu
```

---

## 6. 代码级优化策略

### 6.1 缩小变量生命周期 (Live Range Reduction)

这是**最重要**的寄存器优化策略。编译器分配物理寄存器时，同时存活 (live) 的变量越多，所需寄存器越多。

```cpp
// ❌ 宽生命周期: a, b, c, d 同时存活
__global__ void bad(float* data, int N) {
    float a = data[threadIdx.x];           // a 存活开始
    float b = data[threadIdx.x + N];       // b 存活开始
    float c = data[threadIdx.x + 2*N];     // c 存活开始
    float d = data[threadIdx.x + 3*N];     // d 存活开始
    // 此时 4 个寄存器同时活跃
    float result = a + b + c + d;           // 全部使用
    data[threadIdx.x] = result;
}

// ✅ 窄生命周期: 逐步累加, 每次仅 2 个变量存活
__global__ void good(float* data, int N) {
    float acc = data[threadIdx.x];          // acc 存活
    acc += data[threadIdx.x + N];           // 临时值立即消费
    acc += data[threadIdx.x + 2*N];
    acc += data[threadIdx.x + 3*N];
    data[threadIdx.x] = acc;
}
```

### 6.2 Recomputation vs. Storage (重算替代存储)

如果一个值的计算代价低于将其保存在寄存器中的代价 (占用寄存器的时间内其他变量无法使用该寄存器)，则选择重算：

```cpp
// ❌ 存储中间结果: offset 占用 1 个寄存器整个 kernel 生命周期
__global__ void bad(float* data, int stride, int base) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x + base;  // 保存
    // ... 100 行其他代码 ...
    float val = data[offset];  // 使用
    // ... 100 行其他代码 ...
    data[offset] = val * 2.0f;  // 再次使用
}

// ✅ 重算: offset 不占用寄存器, 每次使用时重算 (3 条 ALU 指令, 延迟 ~3 cycles)
__global__ void good(float* data, int stride, int base) {
    // ... 100 行其他代码 ...
    int offset = blockIdx.x * blockDim.x + threadIdx.x + base;
    float val = data[offset];
    // ... 100 行其他代码 ...
    offset = blockIdx.x * blockDim.x + threadIdx.x + base;  // 重算
    data[offset] = val * 2.0f;
}
```

### 6.3 避免 64-bit 隐式提升

```cpp
// ❌ 隐式 64-bit: 字面量 1000000L 是 long long → 整个表达式提升到 int64
int idx = threadIdx.x + blockIdx.x * blockDim.x + 1000000L;
// 编译器可能为 idx 分配 2 个寄存器 (int64)

// ✅ 保持 32-bit
int idx = threadIdx.x + blockIdx.x * blockDim.x + 1000000;
// idx 只需 1 个寄存器

// ❌ 指针运算中的 64-bit 开销
float* ptr = base + (long long)threadIdx.x * stride;
// 64-bit 乘法和加法各需要 2 个寄存器

// ✅ 先做 32-bit 偏移计算, 最后才转指针
int offset = threadIdx.x * stride;  // 32-bit
float val = base[offset];           // 编译器内部做 64-bit 指针加法
```

### 6.4 Shared Memory 代替寄存器

将不频繁访问但生命周期长的变量移到 Shared Memory：

```cpp
// ❌ 累加器占寄存器: 16 个 float = 16 regs
float acc[16];  // 全在寄存器
for (int k = 0; k < K; k++) {
    for (int i = 0; i < 16; i++) {
        acc[i] += ...;
    }
}

// ✅ 累加器在 Shared Memory: 释放 16 个寄存器
__shared__ float acc_smem[BLOCK_SIZE][16];
for (int k = 0; k < K; k++) {
    for (int i = 0; i < 16; i++) {
        acc_smem[threadIdx.x][i] += ...;  // ~20–30 cycles vs ~1 cycle
    }
}
```

> **权衡：** Shared Memory 延迟 (20–30 cycles) 远低于 Local Memory spill (30–200 cycles)，但高于寄存器 (1 cycle)。适用于访问频率低或可被计算延迟隐藏的变量。

### 6.5 循环展开控制

`#pragma unroll` 展开循环会增加每个循环体实例的寄存器需求 (更多同时存活的变量)：

```cpp
// ❌ 完全展开: 所有迭代的变量同时存活
#pragma unroll
for (int i = 0; i < 16; i++) {
    tmp[i] = input[i] * weight[i];  // 16 个 tmp 同时活跃
}

// ✅ 部分展开: 平衡 ILP 与寄存器压力
#pragma unroll 4
for (int i = 0; i < 16; i++) {
    tmp = input[i] * weight[i];  // 每次仅 4 个 tmp 活跃
    output[i] = tmp;
}

// ✅ 禁止展开 (寄存器极度紧张时)
#pragma unroll 1
for (int i = 0; i < 16; i++) { ... }
```

### 6.6 `__noinline__` 控制函数内联

内联函数会将被调用函数的寄存器需求 "合并" 到调用者中：

```cpp
// ❌ 内联后: caller 的寄存器 + callee 的寄存器 → 压力叠加
__device__ __forceinline__ float heavy_compute(float x) {
    // 使用 30 个临时变量...
    return result;
}

// ✅ __noinline__: callee 的寄存器独立于 caller
__device__ __noinline__ float heavy_compute(float x) {
    // 使用 30 个临时变量, 但不影响 caller 的寄存器预算
    return result;
}
```

> **注意：** `__noinline__` 引入函数调用开销 (栈帧、跳转)。仅当内联导致寄存器 spill 时才值得考虑。

### 6.7 volatile 技巧 (谨慎使用)

声明变量为 `volatile` 强制编译器将其存储到 Local Memory 而非寄存器：

```cpp
// volatile 强制变量离开寄存器
volatile int temp_idx = compute_index();
// 编译器生成 STL + LDL 而非寄存器保持
```

> **警告：** `volatile` 阻止编译器进行多种优化 (公共子表达式消除、指令重排、load scheduling)。仅在其他方法都失败时尝试，且必须 profiling 验证效果。报告的案例中 (RTX 3090, CUDA 11.1)，volatile 使寄存器从 96 降到 80，kernel 提速 50%。但也有案例适得其反。

---

## 7. MMA 指令的寄存器开销

### 7.1 各 MMA 指令的寄存器需求

Tensor Core MMA 指令的操作数和累加器必须驻留在寄存器中 (Blackwell TMEM 除外)，这是 GEMM kernel 寄存器压力的最大来源。

#### mma.sync (Ampere/Turing)

| 指令形状 | A 操作数 (regs/thread) | B 操作数 | C/D 累加器 (FP32) | C/D 累加器 (FP16) | 总计 (FP32 accum) |
|---------|:---------------------:|:--------:|:-----------------:|:-----------------:|:-----------------:|
| m16n8k16 (FP16/BF16) | 4 | 2 | 4 | 2 | **10** |
| m16n8k16 (INT8) | 2 | 1 | 4 | — | **7** |
| m16n8k32 (INT8/FP8) | 4 | 2 | 4 | — | **10** |
| m16n8k8 (FP16/BF16) | 2 | 1 | 4 | 2 | **7** |

#### Warp 级 Tile 的寄存器消耗

一个 Warp 通常会使用多个 MMA atom 拼成更大的 tile。以 FP16 m16n8k16 拼成常见的 Warp-level tile 为例：

```
Warp Tile: 16×16 (2 个 m16n8k16 横向拼接)
  A: 4 regs × 1 = 4 regs
  B: 2 regs × 2 = 4 regs
  C/D (FP32): 4 regs × 2 = 8 regs
  小计: 16 regs/thread

Warp Tile: 32×32 (2×4 个 m16n8k16)
  A: 4 regs × 2 = 8 regs
  B: 2 regs × 4 = 8 regs
  C/D (FP32): 4 regs × 8 = 32 regs
  小计: 48 regs/thread         ← 仅累加器就占 32 regs!

Warp Tile: 64×64 (4×8 个 m16n8k16)
  A: 4 regs × 4 = 16 regs
  B: 2 regs × 8 = 16 regs
  C/D (FP32): 4 regs × 32 = 128 regs
  小计: 160 regs/thread        ← 已接近 255 上限!
```

> **核心矛盾：** 更大的 Warp Tile → 更高的算术强度 (数据复用率) → 更高的 Tensor Core 利用率，但寄存器消耗按 tile 面积的平方增长。

#### 双缓冲的寄存器开销

CUTLASS 使用双缓冲 (double buffering) 重叠加载与计算：同时持有当前迭代的 A/B fragment 和下一迭代的 A/B fragment。

```
双缓冲: A/B fragment 寄存器 × 2
  Warp Tile 16×16: (4 + 4) × 2 = 16 regs (双缓冲) + 8 regs (accum) = 24 regs
  Warp Tile 32×32: (8 + 8) × 2 = 32 regs (双缓冲) + 32 regs (accum) = 64 regs
```

### 7.2 WGMMA (Hopper, SM 9.0)

Hopper 的 WGMMA 在 Warpgroup (128 线程 = 4 Warps) 级别执行 MMA：

```
WGMMA 累加器寄存器 (每线程):
  FP16 累加:   ~90 regs/thread
  FP32 累加:  ~168 regs/thread   ← 配合 register reallocation 才可行

不使用 setmaxnreg (无寄存器重分配):
  编译器报告: 2784 bytes stack frame, 4764 bytes spill stores
  → 灾难级 spilling!
```

### 7.3 tcgen05 (Blackwell, SM 10.0)

Blackwell 的 tcgen05 使用 TMEM 存放累加器，从根本上解决了 MMA 累加器的寄存器压力问题。详见 [Section 9](#9-blackwell-tensor-memory-tmem-解放寄存器)。

---

## 8. Hopper: Warp Specialization 与 `setmaxnreg`

### 8.1 问题：Producer 与 Consumer 的寄存器不平衡

Hopper 架构引入了**Warp Specialization** 范式——将 Warp 分为 Producer (负责数据加载) 和 Consumer (负责 MMA 计算) 两种角色：

```
传统 (Homogeneous):
  所有 Warp 做相同的工作 → 寄存器需求相同 → 公平分配

Warp Specialization:
  Producer Warp: 使用 TMA 加载数据 → 仅需 ~40 regs/thread
  Consumer Warp: 执行 WGMMA + epilogue → 需要 ~200 regs/thread

问题: 固定分配寄存器 → Producer 浪费, Consumer 不够用
```

### 8.2 `setmaxnreg` 动态寄存器重分配

Hopper 引入 PTX 指令 `setmaxnreg`，允许在 kernel 运行时动态调整每个 Warpgroup 的最大寄存器数：

```
PTX 指令:
  setmaxnreg.inc.sync.aligned.u32 N;   // 增加到 N regs/thread
  setmaxnreg.dec.sync.aligned.u32 N;   // 减少到 N regs/thread

约束:
  - N 必须是 8 的倍数
  - N 的范围: 24 ≤ N ≤ 256
  - 必须由 Warpgroup 中所有线程同步执行
  - 减少的寄存器立即归还给 SM 供其他 Warpgroup 使用
```

### 8.3 CUTLASS 中的使用模式

```cpp
// CUTLASS Hopper GEMM: Warp Specialization 模式
// 1 个 Producer Warpgroup + 2 个 Consumer Warpgroup

__global__ void __launch_bounds__(384, 1) gemm_ws_kernel(...) {
    int warp_group_idx = threadIdx.x / 128;

    if (warp_group_idx == 0) {
        // === Producer Warpgroup ===
        // 释放寄存器给 Consumer (从初始分配减少到 40)
        cutlass::arch::warpgroup_reg_dealloc<40>();
        // → PTX: setmaxnreg.dec.sync.aligned.u32 40;

        // TMA 加载循环 (仅需少量寄存器)
        while (has_work) {
            tma_load_tile(...);
            arrive_at_barrier(...);
        }
    } else {
        // === Consumer Warpgroup ===
        // 获取额外寄存器 (从初始分配增加到 232)
        cutlass::arch::warpgroup_reg_alloc<232>();
        // → PTX: setmaxnreg.inc.sync.aligned.u32 232;

        // WGMMA 计算循环 (需要大量寄存器存放累加器)
        while (has_work) {
            wait_on_barrier(...);
            wgmma_compute(...);  // 累加器在寄存器中
        }
    }
}
```

### 8.4 寄存器重分配的数学

```
假设: 384 线程 (3 Warpgroup), 65536 regs/SM

均匀分配:
  65536 / (384/32) = 65536 / 12 = 5461 regs/warp ≈ 170 regs/thread
  → Consumer 需 200 → spill!

重分配后:
  Producer (1 WG, 4 warps): 40 regs/thread × 32 × 4 = 5120 regs
  Consumer (2 WG, 8 warps): (65536 - 5120) / 8 / 32 ≈ 236 regs/thread
  → Consumer 有 236 regs, 远超需求 → 无 spill!

节省: Producer 释放 (170-40) × 128 = 16640 个寄存器 → 给 Consumer 使用
```

### 8.5 `setmaxnreg` 的限制

- 仅 Hopper (SM 9.0) 及以上支持
- 必须配合 `__launch_bounds__` 使用——`minBlocksPerSM` 通常设为 1
- 增加寄存器 (`inc`) 是阻塞操作——直到 SM 有足够空闲寄存器才返回
- 减少寄存器 (`dec`) 不能低于 kernel 当前实际使用的寄存器数
- 与 `__launch_bounds__` 的组合可能导致编译器对初始寄存器分配的错误判断

---

## 9. Blackwell: Tensor Memory (TMEM) 解放寄存器

### 9.1 问题的根源

GEMM kernel 中，MMA 累加器是寄存器压力的最大单一来源：

```
m64n128 tile, FP32 累加器:
  每线程累加器 = 64 × 128 / 128 线程 = 64 个 FP32 寄存器
  占 255 上限的 25%

m64n256 tile, FP32 累加器:
  每线程累加器 = 64 × 256 / 128 = 128 个 FP32 寄存器
  占 255 上限的 50%!
```

### 9.2 TMEM 硬件解决方案

Blackwell SM 10.0 引入 **Tensor Memory (TMEM)**——SM 上的专用 256 KB 存储区域，用于存放 MMA 累加器：

```
每 SM TMEM 容量: 256 KB
  = 512 列 × 128 行 × 32 bit
  = 与寄存器文件 (256 KB) 等大!

TMEM 接口:
  tcgen05.alloc   → 在 TMEM 中分配空间 (类似 malloc)
  tcgen05.dealloc → 释放 TMEM 空间
  tcgen05.ld      → TMEM → Register (后处理时读回)
  tcgen05.st      → Register → TMEM
  tcgen05.cp      → Shared Memory → TMEM

tcgen05.mma:
  A, B → 来自 Shared Memory
  C, D → 驻留在 TMEM
  → 累加器完全不占用通用寄存器!
```

### 9.3 TMEM 对寄存器压力的影响

```
Hopper GEMM (m64n256, FP32 accum):
  累加器:    128 regs/thread  ← 在寄存器中
  A Fragment: ~16 regs/thread
  其他:       ~40 regs/thread
  总计:       ~184 regs/thread → 需要 setmaxnreg 才能运行

Blackwell GEMM (m256n256, FP32 accum):
  累加器:    0 regs/thread    ← 在 TMEM 中!
  A/B:       来自 Shared Memory (不占寄存器)
  其他:      ~40 regs/thread
  总计:      ~40 regs/thread  → Occupancy 大幅提升

→ TMEM 将 GEMM kernel 的寄存器需求从 ~200 降到 ~40
→ 从根本上解决了 Tensor Core GEMM 的寄存器压力问题
```

> **限制：** TMEM 仅存在于 SM 10.0 (B200/B100 数据中心 GPU)。SM 12.0 (RTX 5090) **没有 TMEM 硬件**。

---

## 10. CUDA 13.0: Shared Memory Register Spilling

### 10.1 概述

CUDA 13.0 引入了一项新优化：允许编译器将寄存器 spill 到 **Shared Memory** 而非 Local Memory (DRAM)。这将 spill 延迟从 L1/L2 级别 (~30–200 cycles) 降低到 Shared Memory 级别 (~20–30 cycles)。

### 10.2 启用方式

通过 PTX 内联汇编 pragma 启用：

```cpp
__global__ void __launch_bounds__(256, 2) my_kernel(...) {
    // 在 kernel 入口处启用 shared memory spilling
    asm volatile("pragma \"enable_smem_spilling\";" ::: "memory");

    // ... kernel 代码 ...
}
```

### 10.3 工作原理

```
不启用 (传统):
  编译器 spill → STL (Store to Local Memory) → DRAM/L1/L2
                → LDL (Load from Local Memory) → DRAM/L1/L2

启用后:
  编译器 spill → STS (Store to Shared Memory) → On-chip SRAM (~20 cycles)
                → LDS (Load from Shared Memory) → On-chip SRAM (~20 cycles)
  如果 Shared Memory 不够 → 回退到 Local Memory (保证正确性)
```

### 10.4 使用条件

- 必须指定 `__launch_bounds__` (编译器需要知道 shared memory 容量来分配 spill 空间)
- 不能与动态分配的 shared memory (`extern __shared__`) 同时使用
- 仅在函数作用域内有效
- 编译器自动决定哪些变量 spill 到 shared memory

### 10.5 实测效果

NVIDIA 官方 benchmark 结果：

```
优化前:
  Used 255 registers
  X bytes spill stores, Y bytes spill loads
  Duration: 8.35 us

优化后 (enable_smem_spilling):
  Used 255 registers
  0 bytes spill stores, 0 bytes spill loads    ← spill 完全消除!
  46080 bytes smem (含 spill 空间)
  Duration: 7.71 us

提升:
  - Kernel 时长: -7.76%
  - Elapsed cycles: -8%
  - Spill 指令: 完全消除
```

### 10.6 与其他策略的关系

```
优化优先级 (从优到劣):

1. 消除寄存器压力 (代码优化) → 无 spill, 最优
2. TMEM (Blackwell SM 10.0)  → 累加器不占寄存器
3. setmaxnreg (Hopper SM 9.0) → 寄存器动态重分配
4. Shared Memory Spilling (CUDA 13.0) → 20–30 cycle spill
5. Local Memory Spilling (传统) → 30–800 cycle spill

→ Shared Memory Spilling 是在无法消除 spill 时的有效缓解手段
```

---

## 11. 寄存器 Bank Conflict 与 Operand Reuse

### 11.1 寄存器文件的 Bank 结构

GPU 寄存器文件被组织为多个 **Bank**，类似 Shared Memory 的 bank 结构。当一条指令的多个源操作数位于同一 bank 时，产生 **Register Bank Conflict**，需要额外 1 个周期序列化访问。

```
寄存器文件 Bank 结构 (简化):

Bank 0: R0, R4, R8,  R12, ...
Bank 1: R1, R5, R9,  R13, ...
Bank 2: R2, R6, R10, R14, ...
Bank 3: R3, R7, R11, R15, ...

指令: FFMA R0, R4, R8
  R0 → Bank 0
  R4 → Bank 0  ← 冲突!
  R8 → Bank 0  ← 冲突!
  → 3-way bank conflict → 需要额外 2 个周期

指令: FFMA R0, R1, R2
  R0 → Bank 0
  R1 → Bank 1
  R2 → Bank 2
  → 无冲突 → 1 个周期完成操作数收集
```

> **注意：** 编译器 (ptxas) 会尝试将变量分配到不同 bank 以避免冲突。手写 PTX 或内联汇编时需注意寄存器编号的 bank 分布。

### 11.2 Operand Collector 与 Reuse Buffer

GPU 通过 **Operand Collector** (操作数收集器) 架构缓解 bank conflict：

```
Register File
     │
     ├─→ Arbiter (仲裁器, 序列化冲突的 bank 访问)
     │
     ├─→ Operand Collector Buffer (缓冲已读取的操作数)
     │       │
     │       └─→ Operand Reuse Buffer (缓存最近使用的操作数)
     │
     └─→ ALU / Tensor Core
```

**Operand Reuse Buffer** 机制：

```
SASS 中的 .reuse 标记:

  FFMA R4, R0, R1, R2;        // 正常: 从 RF 读取 R0, R1, R2
  FFMA R5, R0.reuse, R3, R2.reuse;  // R0 和 R2 从 reuse buffer 读取
                                      // 避免 RF 再次读取 → 减少 bank conflict

.reuse 的效果:
  1. 减少寄存器文件的读取带宽压力
  2. 缓解 bank conflict (从 buffer 读取绕过 bank)
  3. 降低能耗 (RF 读取能耗 >> buffer 读取)
```

### 11.3 对 Kernel 开发者的影响

```
编译器 (ptxas) 自动处理:
  - 选择物理寄存器编号以减少 bank conflict
  - 插入 .reuse 标记以利用 operand reuse buffer
  - 指令调度以避免连续的 bank conflict

开发者可以做的:
  1. 减少同一指令中使用相邻声明的变量 (可能被分配到同一 bank)
  2. 使用 cuobjdump -sass 检查 .reuse 标记是否存在
  3. 手写 PTX 时注意寄存器编号 mod 4 的分布
  4. 信任编译器——绝大多数情况下 ptxas 的 bank 分配已经很好
```

---

## 12. LLM Kernel 实战：寄存器优化模式

### 12.1 GEMM Tile Size 选择的寄存器权衡

```
                ┌──────────────────────────────────┐
                │          Tile Size 决策           │
                │                                  │
                │   更大 Tile                       │
                │   ├── ✅ 更高算术强度              │
                │   ├── ✅ 更少 Global Memory 访问   │
                │   ├── ❌ 更多累加器寄存器           │
                │   ├── ❌ 更多 Shared Memory        │
                │   └── ❌ 更低 Occupancy            │
                │                                  │
                │   更小 Tile                       │
                │   ├── ❌ 更低算术强度              │
                │   ├── ❌ 更多 Global Memory 访问   │
                │   ├── ✅ 更少寄存器                │
                │   ├── ✅ 更少 Shared Memory        │
                │   └── ✅ 更高 Occupancy            │
                └──────────────────────────────────┘

实践中的甜蜜点 (Ampere A100, FP16 GEMM):
  CTA Tile 128×128×32, Warp Tile 32×32:
    ~64 regs/thread, ~50% occupancy → 接近 cuBLAS 性能

  CTA Tile 256×128×32, Warp Tile 64×32:
    ~128 regs/thread, ~25% occupancy → 大矩阵最优
```

### 12.2 FlashAttention 的寄存器优化

FlashAttention 将 GEMM + Softmax 融合，避免中间结果写回 HBM，但代价是寄存器同时持有：

```
FlashAttention 寄存器预算:
  1. QK^T GEMM 累加器 (S tile):     ~32 regs
  2. Online Softmax 状态:
     - row_max (m):                  ~4 regs
     - row_sum (l):                  ~4 regs
     - rescaling factors:            ~4 regs
  3. PV GEMM 累加器 (O tile):       ~32 regs
  4. Q Fragment (从 SMEM 加载):      ~8 regs
  5. K/V Fragment (从 SMEM 加载):    ~8 regs
  6. 临时变量、指针、循环变量:        ~16 regs
  ────────────────────────────────────────
  总计:                              ~108 regs/thread

优化策略:
  - 使用 FP16 累加器 (regs 减半, 精度略低) → ~76 regs
  - 减小 head_dim tile → 累加器减小
  - Hopper: 使用 setmaxnreg 给 Consumer 更多寄存器
```

### 12.3 Fused Elementwise Kernel 的寄存器管理

```cpp
// ============= Fused: Residual + SwiGLU + RMSNorm =============
// 错误示范: 一次性加载所有输入

// ❌ 所有向量同时活跃 → 大量寄存器
__global__ void bad_fused_kernel(
    half* output, const half* residual, const half* gate,
    const half* up, const half* weight, float eps, int D
) {
    // 一次性加载所有输入 (4 × uint4 = 64B = 16 个寄存器)
    uint4 v_res  = load_vec(residual, ...);
    uint4 v_gate = load_vec(gate, ...);
    uint4 v_up   = load_vec(up, ...);
    uint4 v_w    = load_vec(weight, ...);  // 4 个 uint4 同时活跃!
    // ... 计算 ...
}

// ✅ 分阶段处理: 每次仅 1–2 个向量活跃
__global__ void good_fused_kernel(...) {
    float sum_sq = 0.0f;

    // Phase 1: Residual Add + SwiGLU → 写入 Shared Memory
    for (int i = threadIdx.x; i < D/8; i += blockDim.x) {
        uint4 v_res  = load_vec(residual, i);
        uint4 v_gate = load_vec(gate, i);
        uint4 v_up   = load_vec(up, i);

        // Residual add (v_res 立即消费)
        half2* h2_res = reinterpret_cast<half2*>(&v_res);
        // ... residual add ...

        // SwiGLU: silu(gate) * up (gate, up 立即消费)
        // ... compute, 存入 shared memory ...

        // 累加 sum_sq (用于后续 RMSNorm)
        // v_gate, v_up 已不再需要 → 寄存器可复用
    }
    __syncthreads();

    // Phase 2: RMSNorm (从 Shared Memory 读取)
    // 此时无需 gate, up 的寄存器
    for (int i = threadIdx.x; i < D/8; i += blockDim.x) {
        uint4 v = load_from_smem(i);
        uint4 v_w = load_vec(weight, i);
        // ... normalize and store ...
    }
}
```

### 12.4 量化 Kernel 的寄存器优化

```cpp
// FP8/FP4 dequantize + GEMV: 避免同时展开过多解包结果

// ❌ 完全展开 → 16 个 float 同时活跃 (16 regs)
uint4 packed = load_fp8(...);  // 16 个 FP8
float vals[16];
#pragma unroll
for (int i = 0; i < 16; i++) {
    vals[i] = fp8_to_float(packed_byte(packed, i));
}
float dot = 0;
for (int i = 0; i < 16; i++) {
    dot += vals[i] * x[i];
}

// ✅ 流式处理: 每次仅解包 4 个
uint4 packed = load_fp8(...);
float dot = 0;
__nv_fp8x4_e4m3* groups = reinterpret_cast<__nv_fp8x4_e4m3*>(&packed);
#pragma unroll
for (int g = 0; g < 4; g++) {
    float4 f4 = static_cast<float4>(groups[g]);  // 4 个 float 活跃
    float4 x4 = load_x_float4(g);
    dot += f4.x * x4.x + f4.y * x4.y + f4.z * x4.z + f4.w * x4.w;
    // f4, x4 立即消费 → 寄存器复用
}
```

---

## 13. 诊断与分析工具

### 13.1 编译时诊断

```bash
# 显示每个 kernel 的寄存器使用和 spill 信息
nvcc --ptxas-options=-v kernel.cu 2>&1
# 输出示例:
# ptxas info: Compiling entry function 'my_kernel'
# ptxas info: Used 96 registers, 8192 bytes smem, 380 bytes cmem[0]
# ptxas info: 48 bytes stack frame, 48 bytes spill stores, 48 bytes spill loads

# 各字段含义:
# Used N registers        → 每线程物理寄存器数 (核心指标)
# K bytes smem            → 静态 Shared Memory
# X bytes stack frame     → 函数调用栈帧 (非 0 说明有 __noinline__ 或递归)
# Y bytes spill stores    → 寄存器溢出到 Local Memory 的存储量
# Z bytes spill loads     → 从 Local Memory 读回的量
# spill stores = spill loads = 0 → 无 spilling ✅
```

### 13.2 SASS 级检查

```bash
# 生成 SASS 查看 spill 指令
nvcc -cubin kernel.cu && cuobjdump -sass kernel.cubin | grep -E "STL|LDL|STS|LDS"

# STL = Store to Local (spill store)
# LDL = Load from Local (spill load)
# 如果大量 STL/LDL 出现在 hot loop 内 → 需要优化

# 检查 .reuse 标记
cuobjdump -sass kernel.cubin | grep ".reuse"
# .reuse 标记多 → 编译器在积极利用 operand reuse buffer
```

### 13.3 Nsight Compute 运行时分析

```bash
# 寄存器相关的关键 metrics
ncu --metrics \
    launch__registers_per_thread,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum \
    ./my_kernel

# launch__registers_per_thread         → 运行时实际寄存器/线程
# sm__warps_active.avg.pct_of_peak...  → 实际 occupancy
# l1tex__...local_op_ld/st             → Local Memory 访问 (spill 指标)

# Occupancy 分析 (完整)
ncu --section Occupancy ./my_kernel
# 显示: 理论/实际 occupancy, 限制因素 (registers/smem/block)
```

### 13.4 Occupancy Calculator API

```cpp
#include <cuda_runtime.h>

// 运行时查询 kernel 的 occupancy
int blockSize = 256;
int minGridSize, optimalBlockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize,
                                    my_kernel, 0, 0);
// optimalBlockSize: 编译器推荐的 block 大小 (平衡寄存器和 occupancy)

// 计算给定 block 大小的 occupancy
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, my_kernel,
                                               blockSize, sharedMemSize);
float occupancy = (float)(numBlocks * blockSize / 32) /
                  deviceProp.maxWarpsPerMultiprocessor;
```

---

## 14. 优化检查清单

### 14.1 开发阶段

- [ ] 所有性能关键 kernel 标注 `__launch_bounds__(maxThreads, minBlocks)`
- [ ] 编译加 `--ptxas-options=-v` 检查寄存器使用和 spill 量
- [ ] 寄存器使用对齐到 Warp 分配粒度边界 (避免浪费)
- [ ] 避免 64-bit 隐式提升 (检查循环变量、数组索引)
- [ ] 变量声明靠近首次使用 (缩小 live range)
- [ ] 低频使用的值考虑重算而非存储
- [ ] `#pragma unroll` 数字要平衡 ILP 和寄存器压力

### 14.2 GEMM / MMA Kernel

- [ ] 清楚 MMA 累加器的寄存器开销 (参考 Section 7)
- [ ] Tile 大小选择考虑寄存器和 Shared Memory 的联合约束
- [ ] 双缓冲的 Fragment 寄存器开销纳入预算
- [ ] Hopper: 使用 `setmaxnreg` 重分配 Producer/Consumer 寄存器
- [ ] Blackwell SM 10.0: 使用 TMEM 存放累加器

### 14.3 Spill 诊断与处理

```
优先级从高到低:

1. 消除 spill (代码优化)
   - 缩小变量生命周期
   - 重算替代存储
   - 减小向量化宽度 (float4 → float2)
   - 调整 #pragma unroll 参数

2. 如果无法消除:
   - CUDA 13.0: 启用 enable_smem_spilling
   - 确保 spill 不在 hot loop 内
   - 验证 spill 的 L1 cache 命中率 (ncu)

3. 如果 spill 换来显著 occupancy 提升:
   - 可能是值得的 (特别是 memory-bound kernel)
   - 需要 profiling 验证
```

### 14.4 常见陷阱

```
🚩 陷阱 1: 盲目追求 0 spill
   有时少量 spill + 高 occupancy 优于 0 spill + 低 occupancy
   → 以实测性能为准, 非寄存器数

🚩 陷阱 2: 不指定 __launch_bounds__
   编译器假设最大 block size → 可能过度 spill 或寄存器过多
   → 始终指定

🚩 陷阱 3: 64-bit 数组索引
   size_t / long long 用于索引 → 每个索引 2 个寄存器
   → 确认 32-bit int 足够时使用 int

🚩 陷阱 4: 过度展开循环
   #pragma unroll 无参数 → 完全展开 → 寄存器爆炸
   → 始终给定展开因子: #pragma unroll 4

🚩 陷阱 5: 忽略寄存器分配粒度的阶梯效应
   从 48 到 47 regs 节省 1 个寄存器 → 无 occupancy 变化
   从 41 到 40 regs 节省 1 个寄存器 → occupancy 从 62.5% 到 75% (+12.5%)
   → 瞄准粒度边界优化
```

### 14.5 决策流程图

```
                    ┌─────────────────────┐
                    │ 检查 ptxas -v 输出   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Spill > 0 bytes?     │
                    └──┬───────────────┬──┘
                       │ Yes           │ No
               ┌───────▼──────┐  ┌────▼─────────────┐
               │ Spill 在     │  │ Occupancy 足够?    │
               │ hot loop 内? │  │ (memory-bound: >50%│
               └──┬────────┬──┘  │  compute-bound:>25%│
                  │Yes     │No   └──┬──────────────┬──┘
           ┌──────▼─────┐ │        │ Yes          │ No
           │ 必须优化!   │ │   ┌────▼────┐   ┌────▼────────┐
           │ 代码级优化  │ │   │ 性能OK   │   │ 降低 regs    │
           │ 参考 §6    │ │   │ 完成 ✅  │   │ __launch_   │
           └────────────┘ │   └─────────┘   │ bounds 调整  │
                          │                  │ 代码级优化    │
                   ┌──────▼──────┐           └──────────────┘
                   │ CUDA 13.0?  │
                   └──┬───────┬──┘
                      │Yes    │No
               ┌──────▼────┐ ┌▼────────────────┐
               │ 尝试      │ │ 接受 spill      │
               │ smem_spill│ │ (如果 L1 命中高  │
               └───────────┘ │  且 occupancy 好)│
                             └─────────────────┘
```

---

## 参考资源

- [Modal GPU Glossary: Register Pressure](https://modal.com/gpu-glossary/perf/register-pressure)
- [Modal GPU Glossary: Occupancy](https://modal.com/gpu-glossary/perf/occupancy)
- [Modal GPU Glossary: Registers](https://modal.com/gpu-glossary/device-software/registers)
- [NVIDIA Blog: How to Improve CUDA Kernel Performance with Shared Memory Register Spilling](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/)
- [NVIDIA Training: Local Memory and Register Spilling (PDF)](https://developer.download.nvidia.com/CUDA/training/register_spilling.pdf)
- [NVIDIA Training: Warps and Occupancy (PDF)](https://developer.download.nvidia.com/CUDA/training/cuda_webinars_WarpsAndOccupancy.pdf)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [NVIDIA Blog: CUDA Pro Tip — Occupancy API Simplifies Launch Configuration](https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/)
- [Lei Mao: CUDA Occupancy Calculation](https://leimao.github.io/blog/CUDA-Occupancy-Calculation/)
- [siboehm: How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- [Colfax Research: CUTLASS Tutorial — Efficient GEMM Kernel Designs with Pipelining](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)
- [Colfax Research: CUTLASS Tutorial — WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
- [Colfax Research: CUTLASS Tutorial — GEMM with Tensor Memory on Blackwell](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
- [NVIDIA Forums: Tricks to Fight Register Pressure](https://forums.developer.nvidia.com/t/tricks-to-fight-register-pressure-or-how-i-got-down-from-29-to-15-registers/16678)
- [NVIDIA Forums: About Register Bank Conflict](https://forums.developer.nvidia.com/t/about-register-bank-conflict/47853)
- [RegDem: Increasing GPU Performance via Shared Memory Register Spilling (arXiv)](https://arxiv.org/pdf/1907.02894)
- [CuAsmRL: Optimizing GPU SASS Schedules via Deep Reinforcement Learning (arXiv)](https://arxiv.org/html/2501.08071v1)
- [CMU Advanced CUDA: Warp Specialization (PDF)](https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/06-warp-specialization.pdf)
- [CUDA Occupancy Calculator (Web)](https://xmartlabs.github.io/cuda-calculator/)

---

*本文档作为 LLM Kernel Agent 的寄存器压力优化技能参考。与 `vectorization.md`（向量化与寄存器权衡）、`mma-wmma.md`（MMA Fragment 寄存器布局）、`tensor-memory-accelerator.md`（TMA 与 TMEM）配合使用。*
