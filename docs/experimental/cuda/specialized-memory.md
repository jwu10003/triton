# GPU 专用存储深度指南

> 面向 LLM 高性能 Kernel 开发的 GPU 存储层次完整解析与实战优化
> 覆盖 Registers、Shared Memory、Local Memory、Constant Memory、Texture/Read-Only Cache、Global Memory、Pinned Memory、Unified Memory、TMEM (Blackwell)

---

## 目录

1. [GPU 存储层次总览](#1-gpu-存储层次总览)
2. [Registers (寄存器)](#2-registers-寄存器)
3. [Shared Memory](#3-shared-memory)
4. [Local Memory (Register Spilling)](#4-local-memory-register-spilling)
5. [Constant Memory](#5-constant-memory)
6. [Texture Memory / Read-Only Cache](#6-texture-memory--read-only-cache)
7. [Global Memory (HBM)](#7-global-memory-hbm)
8. [Pinned Memory (Page-Locked Host Memory)](#8-pinned-memory-page-locked-host-memory)
9. [Unified Memory (Managed Memory)](#9-unified-memory-managed-memory)
10. [Tensor Memory — TMEM (Blackwell)](#10-tensor-memory--tmem-blackwell)
11. [异步数据传输与拷贝](#11-异步数据传输与拷贝)
12. [LLM Kernel 存储优化实战](#12-llm-kernel-存储优化实战)
13. [存储选型决策指南](#13-存储选型决策指南)
14. [诊断与分析](#14-诊断与分析)
15. [存储优化检查清单](#15-存储优化检查清单)

---

## 1. GPU 存储层次总览

### 1.1 存储层次结构

```
                    ┌───────────────────────────────┐
                    │     Registers (~1 cycle)       │  ← 最快, 线程私有
                    │     256 KB/SM, 65536×32-bit    │
                    └───────────┬───────────────────-┘
                                │
                    ┌───────────▼────────────────────┐
                    │  TMEM (Blackwell, ~1 cycle)     │  ← 线程私有, Tensor Core 专用
                    │  256 KB/SM, tcgen05 指令使用    │
                    └───────────┬────────────────────┘
                                │
          ┌─────────────────────▼──────────────────────────┐
          │     L1 / Shared Memory / Texture Cache         │  ← SM 内共享
          │     (~20–35 cycles, 128–256 KB/SM)             │
          │  ┌─────────────────┬───────────────────────┐   │
          │  │ Shared Memory   │ L1 Data / Texture     │   │
          │  │ (可编程,        │ (硬件管理,            │   │
          │  │  Block 可见)    │  SM 内一致)           │   │
          │  └─────────────────┴───────────────────────┘   │
          │  + Constant Cache (8 KB/SM, 只读, ~1 cyc 广播) │
          └─────────────────────┬──────────────────────────┘
                                │
                    ┌───────────▼────────────────────┐
                    │     L2 Cache (~150–273 cycles)  │  ← 全 SM 共享
                    │     6–126 MB, 全局一致          │
                    └───────────┬────────────────────┘
                                │
                    ┌───────────▼────────────────────┐
                    │  HBM / Global Memory            │  ← 最大容量
                    │  (~400–800 cycles)              │
                    │  32–180 GB, 0.9–8.0 TB/s       │
                    └───────────┬────────────────────┘
                                │
                    ┌───────────▼────────────────────┐
                    │     Host Memory (CPU DRAM)      │  ← 通过 PCIe/NVLink
                    │     (~10,000–50,000 cycles)     │
                    │     Pinned / Unified / Pageable  │
                    └────────────────────────────────┘
```

### 1.2 各存储类型核心特征对比

| 存储类型 | 位置 | 延迟 | 可见范围 | 容量/SM | 可编程性 | 读/写 |
|---------|------|------|---------|---------|---------|-------|
| **Registers** | SM (RF) | ~1 cycle | 线程私有 | 256 KB | 编译器自动 | R/W |
| **TMEM** | SM (Blackwell) | ~1 cycle | 线程私有 | 256 KB | 显式 (tcgen05) | R/W |
| **Shared Memory** | SM (SRAM) | ~20–30 cycles | Block 内 | 0–228 KB | 完全可编程 | R/W |
| **L1 Cache** | SM (SRAM) | ~28–35 cycles | SM 内 | 与 SMEM 共享池 | Cache hints | R (+ store bypass) |
| **Constant Cache** | SM | ~1 cycle (广播) | 全局只读 | 8 KB | `__constant__` | R |
| **Texture Cache** | SM (与 L1 统一) | ~28–35 cycles | 全局只读 | 与 L1 共享 | `__ldg()` | R |
| **L2 Cache** | 全局 | ~150–273 cycles | 全 SM | 6–126 MB | 持久化 API | R/W |
| **Local Memory** | HBM (经 L1/L2) | ~400–800 cycles | 线程私有 | 无限 (从 Global) | 编译器自动 | R/W |
| **Global Memory** | HBM | ~400–800 cycles | 全局 | 32–180 GB | 显式 | R/W |
| **Pinned Memory** | Host DRAM | ~10K+ cycles | Host+Device | 系统 RAM | `cudaMallocHost` | R/W |
| **Unified Memory** | Host+Device | 变化大 | Host+Device | 系统 RAM | `cudaMallocManaged` | R/W |

### 1.3 各架构存储规格对比

| 规格 | Volta V100 | Ampere A100 | Hopper H100 | Blackwell B200 | RTX 5090 |
|------|:----------:|:-----------:|:-----------:|:--------------:|:--------:|
| **CC** | 7.0 | 8.0 | 9.0 | 10.0 | 12.0 |
| **RF (每 SM)** | 256 KB | 256 KB | 256 KB | 256 KB | 256 KB |
| **TMEM (每 SM)** | — | — | — | 256 KB | — |
| **L1/Tex/SMEM 池 (每 SM)** | 128 KB | 192 KB | 256 KB | 256 KB | 128 KB |
| **最大 SMEM (每 SM)** | 96 KB | 164 KB | 228 KB | 228 KB | 128 KB |
| **最大 SMEM (每 Block)** | 96 KB | 163 KB | 227 KB | 227 KB | 99 KB |
| **Constant Cache (每 SM)** | 8 KB | 8 KB | 8 KB | 8 KB | 8 KB |
| **L2 Cache (全局)** | ~6 MB | 40 MB | 50 MB | 126 MB | 96 MB |
| **显存容量** | 32 GB HBM2 | 80 GB HBM2e | 80 GB HBM3 | 180 GB HBM3e | 32 GB GDDR7 |
| **显存带宽** | 900 GB/s | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s | 1.79 TB/s |
| **SM 数** | 80 | 108 | 132 | 148 | 170 |

---

## 2. Registers (寄存器)

> 详细分析见 [reduce-register-pressure.md](reduce-register-pressure.md)

### 2.1 物理架构

每 SM 拥有 65,536 个 32-bit 寄存器 (256 KB)，是 GPU 上最快的存储层次 (~1 cycle 延迟)。

```
SM Register File (256 KB):
┌──────────────────────────────────────────────────┐
│  65,536 × 32-bit registers                       │
│  ┌─────────┬─────────┬─────────┬─────────┐      │
│  │ Bank 0  │ Bank 1  │ Bank 2  │ Bank 3  │      │  ← 4 Bank 组织
│  │ 16K reg │ 16K reg │ 16K reg │ 16K reg │      │     (Operand Collector)
│  └─────────┴─────────┴─────────┴─────────┘      │
│                                                   │
│  按 Warp 分配: 静态编译时确定, 运行时不回收       │
│  最大 255 regs/thread (CUDA 限制)                 │
│  分配粒度: 256 regs/Warp (CC 7.0+)               │
└──────────────────────────────────────────────────┘
```

### 2.2 分配与 Occupancy

寄存器分配涉及**两级舍入**:

```
Level 1: 每 Warp 寄存器数 → 向上舍入到 256 的倍数
  例: 33 regs/thread × 32 threads = 1056 → ceil 到 1280

Level 2: SM 上 Warp 数 → 向下舍入到 4 的倍数
  例: 65536 / 1280 = 51.2 → floor 到 48 Warps → 75% Occupancy
```

**关键阶梯效应：**

| Regs/Thread | 可用 Warps (CC 8.0) | Occupancy |
|:-----------:|:-------------------:|:---------:|
| ≤ 32 | 64 | 100% |
| 33 | 48 | 75% |
| 41 | 40 | 62.5% |
| 49 | 36 | 56.3% |
| 64 | 32 | 50% |
| 128 | 16 | 25% |

### 2.3 寄存器在 LLM Kernel 中的使用

| Kernel 类型 | 典型寄存器用量 | 主要消耗者 |
|------------|:------------:|-----------|
| GEMM (Tensor Core) | 100–200+ | MMA 累加器 + 输入 Fragment |
| FlashAttention | 120–200 | GEMM 累加器 + Softmax 状态 |
| Fused Elementwise | 60–100 | 多个输入向量 + 中间结果 |
| RMSNorm / LayerNorm | 40–64 | 向量化加载临时 + reduction |
| Quantization | 40–80 | 打包/解包 + scale 因子 |

### 2.4 编译器控制

```cpp
// __launch_bounds__ — 每个 Kernel 独立指定
__global__ void __launch_bounds__(256, 2)  // maxThreadsPerBlock, minBlocksPerSM
myKernel(...) { ... }

// --maxrregcount — 编译单元级全局限制 (nvcc flag)
// nvcc --maxrregcount=128 my_kernel.cu
```

> **建议：** 优先用 `__launch_bounds__`，因为 `--maxrregcount` 会影响编译单元内所有 kernel。

---

## 3. Shared Memory

### 3.1 硬件架构

Shared Memory 是 SM 上的可编程片上 SRAM，自 Volta 起与 L1 Data Cache 共享同一块 SRAM 池：

```
SM 片上 SRAM 池:
┌─────────────────────────────────────────┐
│                                         │
│    ┌──────────────────────────────┐     │
│    │   Shared Memory (0–228 KB)   │     │  ← 完全可编程
│    ├──────────────────────────────┤     │     Block 内所有线程可见
│    │   L1 Data / Texture Cache    │     │  ← 硬件管理
│    │   (剩余空间)                 │     │
│    └──────────────────────────────┘     │
│                                         │
│    总池大小: V100=128, A100=192,        │
│             H100/B200=256 KB            │
└─────────────────────────────────────────┘
```

**可配置的 Carveout 比例：**

| 架构 | 池大小 | 最大 SMEM/SM | 最大 SMEM/Block | 可选大小 (KB) |
|------|:------:|:-----------:|:--------------:|-------------|
| Volta V100 | 128 KB | 96 KB | 96 KB | 0, 8, 16, 32, 64, 96 |
| Ampere A100 | 192 KB | 164 KB | 163 KB | 0, 8, 16, 32, 64, 100, 132, 164 |
| Hopper H100 | 256 KB | 228 KB | 227 KB | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |
| Blackwell B200 | 256 KB | 228 KB | 227 KB | 同 Hopper |
| RTX 5090 | 128 KB | 128 KB | 99 KB | 0, 8, 16, 32, 64, 100, 128 |

> 每 Block 最大 SMEM 比每 SM 少 1 KB, 因为硬件保留了 1 KB 用于 driver 内部用途。

### 3.2 Bank 组织

Shared Memory 被划分为 **32 个 Bank**，每 Bank 宽度 **4 字节**，每周期可独立服务一次 4 字节读写。

```
Bank:    0     1     2     3    ...   30    31
        ┌─────┬─────┬─────┬─────┬───┬─────┬─────┐
Row 0:  │ 0-3 │ 4-7 │8-11 │12-15│...│120- │124- │  ← 128 Bytes/Row
        │     │     │     │     │   │ 123 │ 127 │
        ├─────┼─────┼─────┼─────┼───┼─────┼─────┤
Row 1:  │128- │132- │136- │140- │...│248- │252- │
        │ 131 │ 135 │ 139 │ 143 │   │ 251 │ 255 │
        └─────┴─────┴─────┴─────┴───┴─────┴─────┘

bank_id = (byte_address / 4) % 32
```

**Bank Conflict 规则：**
- 同一 Warp 内不同线程访问同一 Bank 的**不同地址** → N-way Conflict → 串行化
- 同一 Bank 的**同一地址** → 广播 (Broadcast) → 无冲突
- 不同 Warp → 不会冲突

> **详细的 Bank Conflict 分析与解决方案见 [conflict-free-accesses.md](conflict-free-accesses.md)**

### 3.3 声明与分配

#### 3.3.1 静态 Shared Memory (≤ 48 KB)

```cpp
__global__ void kernel() {
    __shared__ float smem[1024];           // 4 KB
    __shared__ half  tile_a[128][64];      // 16 KB
    __shared__ half  tile_b[64][128];      // 16 KB
    // 总计 36 KB, 无需额外配置
}
```

#### 3.3.2 动态 Shared Memory (> 48 KB 需 opt-in)

```cpp
extern __shared__ char smem_buf[];  // 无大小, 由 launch 时指定

__global__ void kernel() {
    // 通过指针算术分割动态 SMEM
    half* tile_a = (half*)smem_buf;                    // offset 0
    half* tile_b = (half*)(smem_buf + 128*64*2);       // offset 16 KB
    float* accum = (float*)(smem_buf + 128*64*2*2);    // offset 32 KB
}

// Host 端: 必须显式设置最大动态 SMEM
cudaFuncSetAttribute(kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);

kernel<<<grid, block, dynamicSmemBytes>>>();
```

> **注意：** 静态声明限制为 48 KB 是为了向后兼容。超过此限制必须使用动态分配 + `cudaFuncSetAttribute`。

#### 3.3.3 配置 Carveout 偏好

```cpp
// 方法 1: 百分比 Carveout (推荐)
cudaFuncSetAttribute(kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);  // 最大化 SMEM

// 方法 2: 枚举偏好 (传统 API)
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared);
// 可选: PreferNone / PreferShared / PreferL1 / PreferEqual
```

### 3.4 Shared Memory 使用模式

#### 3.4.1 数据复用 (GEMM Tiling)

```
Global Memory → Load → Shared Memory → 多次读取 → Registers → Tensor Core

                  ┌──────────────────────────┐
                  │   Global Memory (HBM)    │
                  │   K×M, K×N 矩阵          │
                  └──────────┬───────────────┘
                             │  一次加载
                             ▼
                  ┌──────────────────────────┐
                  │  Shared Memory (SM 内)    │
                  │  A_tile[BM][BK]          │  ← K 次复用
                  │  B_tile[BK][BN]          │
                  └──────────┬───────────────┘
                             │  多次读取
                             ▼
                  ┌──────────────────────────┐
                  │  Registers (线程私有)     │
                  │  Fragment / Accumulator   │
                  └──────────────────────────┘
```

**数据复用率分析 (GEMM):**
- Tile [128×32] 的 A: 被 128/Warp_M 个 Warp 共享 → 复用 N/BN 次
- 无 SMEM: 每线程从 HBM 读 → BW = O(MNK)
- 有 SMEM: 从 HBM 读入 SMEM 一次 → BW = O(MNK × (1/BM + 1/BN))
  - 对于方阵 BM=BN=B: BW = O(2MNK/B), 复用率提升 B/2 倍

#### 3.4.2 线程间通信 (Reduction)

```cpp
// Block-level parallel reduction via Shared Memory
__shared__ float smem[256];
smem[tid] = partial_sum;
__syncthreads();

for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
}
float block_sum = smem[0];  // 所有线程可见
```

#### 3.4.3 数据布局变换 (转置)

```cpp
// 全局 Row-Major → SMEM 转置 → 全局 Column-Major
// 避免直接写 Global 时的 Non-Coalesced 访问
__shared__ float tile[32][33];  // +1 padding 消除 Bank Conflict

int x = blockIdx.x * 32 + threadIdx.x;
int y = blockIdx.y * 32 + threadIdx.y;
tile[threadIdx.y][threadIdx.x] = input[y * N + x];  // Coalesced 读
__syncthreads();

x = blockIdx.y * 32 + threadIdx.x;  // 交换 x, y
y = blockIdx.x * 32 + threadIdx.y;
output[y * M + x] = tile[threadIdx.x][threadIdx.y]; // Coalesced 写, 转置访问 SMEM
```

### 3.5 Double Buffering (流水线)

通过两组 SMEM Buffer 重叠数据加载与计算：

```
Stage:  ────── Iteration i ──────  ── Iteration i+1 ──
Buffer A: [Load tile i+1]           [Compute tile i+1]
Buffer B: [Compute tile i]          [Load tile i+2]

时间线:
  ┌──────────┬──────────┬──────────┬──────────┐
  │ Load B0  │ Load B1  │ Load B0  │ Load B1  │
  │          │ Comp B0  │ Comp B1  │ Comp B0  │
  └──────────┴──────────┴──────────┴──────────┘
       stage 0    stage 1    stage 2    stage 3
```

```cpp
// 双缓冲骨架
__shared__ half smem_a[2][BM][BK];  // 2 个 buffer
__shared__ half smem_b[2][BK][BN];

int buf = 0;
// 预加载第一个 tile 到 buf=0
load_tile(smem_a[0], smem_b[0], k=0);
__syncthreads();

for (int k = 0; k < K; k += BK) {
    // 异步加载下一个 tile 到 buf^1
    if (k + BK < K)
        load_tile(smem_a[buf^1], smem_b[buf^1], k+BK);
    // 用当前 buf 计算
    compute(smem_a[buf], smem_b[buf], accumulator);
    buf ^= 1;
    __syncthreads();
}
```

### 3.6 Multistage Pipeline (Ampere+)

Ampere 引入 `cp.async` 支持多于 2 个 stage 的异步流水线:

```
3-Stage Pipeline (Ampere):
  ┌──────────┬──────────┬──────────┬──────────┬──────────┐
  │ Load S0  │ Load S1  │ Load S2  │ Load S0  │ ...      │
  │          │          │ Comp S0  │ Comp S1  │ Comp S2  │
  └──────────┴──────────┴──────────┴──────────┴──────────┘
```

```cpp
// Ampere cp.async: Global → Shared Memory, 绕过寄存器
// cp.async.ca.shared.global [smem_ptr], [gmem_ptr], 16;
asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
    :: "r"(smem_addr), "l"(gmem_ptr));

// 提交一组 cp.async
asm volatile("cp.async.commit_group;\n");

// 等待: N 表示允许 N 组未完成
asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
```

**Hopper TMA (cp.async.bulk)** 进一步简化:
- 硬件计算地址 → 单条指令批量传输
- 支持 2D/3D tile 描述符
- 硬件 Swizzle 消除 Bank Conflict
- 与 mbarrier 原生集成

```cpp
// TMA Load: 一条指令加载整个 tile
asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    " [%0], [%1, {%2, %3}], [%4];\n"
    :: "r"(smem_addr), "l"(tensorMap), "r"(x), "r"(y), "r"(mbar_addr));
```

> **详细的 TMA 分析见 [tensor-memory-accelerator.md](tensor-memory-accelerator.md)**

### 3.7 Distributed Shared Memory (Hopper+)

Hopper 引入 **Thread Block Cluster** 和 **Distributed Shared Memory (DSMEM)**:

```
Thread Block Cluster (可移植最多 8 Blocks, 非可移植最多 16; 通常 2–4):
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Block 0    │  │   Block 1    │  │   Block 2    │
│   SM 0       │  │   SM 1       │  │   SM 2       │
│  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │
│  │ SMEM 0 │←─┼──┼──│ SMEM 1 │←─┼──┼──│ SMEM 2 │  │
│  └────────┘  │  │  └────────┘  │  │  └────────┘  │
└──────────────┘  └──────────────┘  └──────────────┘
        ↕               ↕               ↕
        └───────── SM-to-SM Network ─────┘
```

- **直接访问相邻 SM 的 SMEM**: 无需经过 L2 或 HBM
- **延迟**: SM 间 ~50–80 cycles (低于 L2 ~273 cycles)
- **TMA Multicast**: 单次 TMA 加载同时写入 Cluster 中多个 SM 的 SMEM

```cpp
// 声明 Cluster (CUDA 12.0+)
cudaLaunchAttribute attr;
attr.id = cudaLaunchAttributeClusterDimension;
attr.val.clusterDim = {2, 1, 1};  // 2 个 Block 组成一个 Cluster

// Kernel 内访问远程 SMEM
// 通过 cluster.map_shared_rank() 获取远程 SMEM 地址
```

### 3.8 Shared Memory 使用要点

| 维度 | 要点 |
|------|------|
| **同步** | `__syncthreads()` 确保 Block 内所有线程完成 SMEM 写入后再读取 |
| **Bank Conflict** | 32 Bank × 4B, 用 Padding / Swizzle 消除 |
| **Carveout** | SMEM 增大 → L1 减少, 需权衡 |
| **容量限制** | 过大 SMEM → Occupancy 降低 (Block/SM 数受限) |
| **异步加载** | Ampere cp.async / Hopper TMA 绕过寄存器直达 SMEM |
| **对齐** | SMEM 基地址自动 128B 对齐; 手动分割时注意对齐 |

---

## 4. Local Memory (Register Spilling)

### 4.1 什么是 Local Memory

Local Memory 是编译器在寄存器不足时，将线程私有变量 "溢出" (Spill) 到的存储空间。虽然名为 "Local"，但**物理上位于 HBM (Global Memory)**，延迟与 Global Memory 相同:

```
变量声明 (线程私有)
        │
        ▼
    编译器判断
   ┌─────────┐
   │能放寄存器?│
   └──┬───┬──┘
      │   │
     Yes  No
      │    │
      ▼    ▼
  Registers  Local Memory (HBM)
  (~1 cyc)   (~400–800 cyc)
              经 L1 + L2 缓存
```

### 4.2 哪些变量会被溢出

| 溢出原因 | 典型场景 |
|---------|---------|
| **寄存器用量超限** | Kernel 使用 > 255 regs/thread 或超出 `__launch_bounds__` 限制 |
| **大数组** | `float arr[256]` — 编译器无法索引到寄存器 |
| **动态索引数组** | `arr[variable_index]` — 编译器无法在编译时解析索引 |
| **函数调用** | 调用栈 (ABI 保存的寄存器) 溢出到 Local Memory |

### 4.3 Local Memory 的缓存行为

```
Thread → Register File
              │ spill
              ▼
    ┌─────────────────┐
    │  L1 Data Cache   │  ← Spill 数据被 L1 缓存
    ├─────────────────┤      命中率通常很高 (线程局部性好)
    │  L2 Cache        │
    ├─────────────────┤
    │  HBM (实际存储)   │  ← 物理位置
    └─────────────────┘

延迟: L1 命中 ~28–35 cycles (vs Register ~1 cycle)
      L1 miss → L2 命中 ~200 cycles
      L2 miss → HBM ~600 cycles
```

> **关键点：** Spill 数据通常 L1 命中率很高 (因为访问模式局部性好)，实际延迟约 30 cycles 而非 600 cycles。但仍比寄存器慢 30×。

### 4.4 检测与量化 Spilling

```bash
# 编译时查看 spill 量
nvcc --ptxas-options=-v kernel.cu
# 输出: Used 128 registers, 48 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads

# Nsight Compute metrics
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_local_op_ld,\
              l1tex__t_bytes_pipe_lsu_mem_local_op_st \
    ./my_app
```

### 4.5 CUDA 13.0: Shared Memory Spilling

CUDA 13.0 引入将 Spill 重定向到 Shared Memory 而非 Local Memory (HBM):

```
传统 Spilling:       CUDA 13.0 Spilling:
Register → HBM      Register → Shared Memory
(~30–600 cyc)        (~20–30 cyc)
```

```cpp
// 启用 SMEM Spilling (CUDA 13.0+, Hopper/Blackwell)
__global__ void __launch_bounds__(256, 2) my_kernel(...) {
    // 在 kernel 入口处启用 shared memory spilling
    asm volatile("pragma \"enable_smem_spilling\";" ::: "memory");

    // ... kernel 代码 ...
    // 编译器自动将部分 spill 重定向到 SMEM
    // 前提: SMEM 有足够剩余空间
}
```

**限制：**
- 需要 SMEM 有剩余空间, 与程序员显式使用的 SMEM 竞争
- 仅对 Hopper (CC 9.0) 和 Blackwell (CC 10.0+) 可用
- 编译器自动决定哪些 spill 放 SMEM, 哪些仍放 Local Memory

---

## 5. Constant Memory

### 5.1 硬件架构

```
┌─────────────────────────────────────────────┐
│              Constant Memory (64 KB)         │  ← 全局, Device 端只读
│              位于 Device DRAM               │
└────────────────────┬────────────────────────┘
                     │ 缓存
          ┌──────────▼──────────┐
          │  Constant Cache     │  ← 每 SM 8 KB
          │  (L1-like 延迟)     │     Warp 级广播
          └─────────────────────┘
```

### 5.2 声明与使用

```cpp
// 声明 (文件作用域)
__constant__ float scale_factors[1024];   // 最多 64 KB 总计
__constant__ int lookup_table[256];

// Host 端写入
cudaMemcpyToSymbol(scale_factors, h_scales, sizeof(float) * 1024);

// Device 端读取 (隐式, 直接使用变量名)
__global__ void kernel() {
    float s = scale_factors[idx];  // 自动通过 Constant Cache 读取
}
```

### 5.3 广播 vs 串行化

Constant Cache 的性能高度依赖 **Warp 内访问的均匀性**:

```
Case 1: 全 Warp 统一访问 (同一地址)
  32 个线程 → scale_factors[0] → 1 次 Cache 读取 + 广播 → ~1 cycle
  效率: 极高, 等效于寄存器

Case 2: 全 Warp 访问不同地址
  T0 → scale_factors[0]
  T1 → scale_factors[1]
  T2 → scale_factors[2]
  ...
  T31 → scale_factors[31]
  → 32 次串行化 Cache 读取 → ~32 cycles
  效率: 极低, 比 Global Memory 可能更慢!

Case 3: 部分统一 (K 个不同地址)
  → K 次串行化读取 → ~K cycles
```

### 5.4 适用与不适用场景

| 场景 | 是否适合 | 原因 |
|------|:-------:|------|
| 全 Warp 统一的超参数 (lr, eps) | 极佳 | 广播, ~1 cycle |
| 量化 scale 表 (per-tensor) | 好 | 同一 Block 内通常统一访问 |
| 小型 LUT (< 8 KB) | 好 | 命中 Constant Cache |
| Per-channel scale (每线程不同) | 差 | 32-way 串行化 |
| 大表 (> 64 KB) | 不可用 | 硬件限制 64 KB |
| 运行时需要修改的数据 | 不可用 | Device 端只读 |

### 5.5 替代方案: `__ldg()` 或 `const __restrict__`

对于需要线程级不同地址读取的只读数据, 使用 Read-Only Cache 更高效:

```cpp
// 替代 Constant Memory — 走 Texture/Read-Only Cache 路径
__global__ void kernel(const float* __restrict__ scales) {
    float s = __ldg(&scales[tid]);  // 每线程不同地址, 仍通过 Read-Only Cache
    // 或: const __restrict__ 修饰让编译器自动用 LDG
}
```

---

## 6. Texture Memory / Read-Only Cache

### 6.1 架构演进

```
Kepler 之前:                            Maxwell 以来:
┌────────────────┐                     ┌────────────────────────┐
│ L1 Data Cache  │ (独立)              │ 统一 L1/Tex/SMEM 池    │
├────────────────┤                     │ ┌────────────────────┐ │
│ Texture Cache  │ (独立)              │ │ L1 Data Cache      │ │
├────────────────┤                     │ │ = Texture Cache     │ │
│ Shared Memory  │ (独立)              │ │ (同一硬件)          │ │
└────────────────┘                     │ ├────────────────────┤ │
                                       │ │ Shared Memory      │ │
                                       │ └────────────────────┘ │
                                       └────────────────────────┘
```

自 Maxwell (CC 5.0) 起, Texture Cache 与 L1 Data Cache **合并为同一硬件**。传统 Texture 管线仍可用，但 `__ldg()` 提供更简洁的只读 Cache 访问路径。

### 6.2 `__ldg()` — Read-Only Data Cache 路径

```cpp
// __ldg() 通过只读纹理缓存路径加载 (CC 3.5+)
float val = __ldg(&input[idx]);

// 编译为 PTX: ld.global.nc (non-coherent, 走 Read-Only Cache)
// vs 普通 ld.global.ca (走 L1 Data Cache, 可能被 store 驱逐)
```

**`__ldg()` vs 普通 Load:**

| 特性 | 普通 `ld.global.ca` | `__ldg()` / `ld.global.nc` |
|------|:------------------:|:-------------------------:|
| Cache 路径 | L1 Data Cache | Read-Only Cache (= Texture Cache) |
| Store 驱逐 | 可能被同 SM 的 store 驱逐 | 不被 store 驱逐 |
| 适用 | 通用读写数据 | 只读数据 |
| 一致性 | Coherent | Non-Coherent (Kernel 内不变) |

### 6.3 编译器自动使用 Read-Only Cache

```cpp
// 方法 1: const __restrict__ 让编译器自动选择 LDG
__global__ void kernel(const float* __restrict__ input, float* output) {
    output[tid] = input[tid] * 2.0f;  // 编译器可能自动使用 LDG
}

// 方法 2: 显式 __ldg (更可靠)
__global__ void kernel(const float* input, float* output) {
    output[tid] = __ldg(&input[tid]) * 2.0f;  // 强制使用 Read-Only Cache
}
```

> **现代最佳实践:** 在 Maxwell+ 架构上, `const __restrict__` 通常足够让编译器优化。显式 `__ldg()` 主要用于确保关键路径的缓存行为。

### 6.4 传统 Texture 对象

传统 Texture API 对 LLM Kernel 价值有限，但仍在某些特殊场景有用:

```cpp
// 创建 Texture 对象
cudaTextureObject_t tex;
cudaResourceDesc resDesc = {};
resDesc.resType = cudaResourceTypeLinear;
resDesc.res.linear.devPtr = d_data;
resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
resDesc.res.linear.sizeInBytes = N * sizeof(float);

cudaTextureDesc texDesc = {};
texDesc.readMode = cudaReadModeElementType;
cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

// Kernel 中使用
__global__ void kernel(cudaTextureObject_t tex) {
    float val = tex1Dfetch<float>(tex, idx);
}
```

**Texture 独有特性 (LLM 中少用):**
- 硬件插值 (线性/双线性) — 图形渲染, 非 LLM 计算
- 2D/3D 空间局部性缓存 — 图像卷积, 非 LLM
- 归一化坐标 + 边界处理 (clamp/wrap) — 图像处理

### 6.5 LLM 中 Read-Only Cache 的实际用途

| 用途 | 方法 | 效果 |
|------|------|------|
| 只读权重/参数加载 | `const __restrict__` 或 `__ldg()` | 避免被同 SM store 驱逐 |
| KV Cache 读取 (Decode) | `__ldg()` | Decode 阶段 KV 只读 |
| Embedding Lookup | `__ldg()` | 随机访问大型 embedding 表 |
| 量化 scale/zero-point 表 | `__ldg()` | 小表, 多线程共享读取 |

---

## 7. Global Memory (HBM)

### 7.1 HBM 物理架构

```
GPU Chip:
┌──────────────────────────────────────────┐
│                                          │
│   ┌──────────────────────────┐           │
│   │      SM Array            │           │
│   │  (计算核心)               │           │
│   └─────────┬────────────────┘           │
│             │ L2 Crossbar                │
│   ┌─────────▼────────────────┐           │
│   │      L2 Cache            │           │
│   │  (6–126 MB)              │           │
│   └─────────┬────────────────┘           │
│             │ Memory Controller          │
│   ┌─────────▼────────────────┐           │
│   │  HBM Stack Interface     │           │
│   └─────────┬────────────────┘           │
└─────────────┼────────────────────────────┘
              │ TSV (Through-Silicon Via)
    ┌─────────▼───────────────────┐
    │     HBM2e / HBM3e Stacks   │
    │     8-Hi / 12-Hi            │
    │                             │
    │  V100: 4×HBM2, 32 GB       │
    │  A100: 5×HBM2e, 80 GB      │
    │  H100: 5×HBM3, 80 GB       │
    │  B200: 8×HBM3e, 180 GB     │
    └─────────────────────────────┘
```

### 7.2 分配 API

| API | 特性 | 用途 |
|-----|------|------|
| `cudaMalloc` | 标准分配, 256B 对齐 | 通用 Device 内存 |
| `cudaMallocPitch` | 2D 行对齐分配 | 2D 矩阵, 保证行对齐 |
| `cudaMalloc3D` | 3D 分配 | 3D 数组 |
| `cudaMallocAsync` | 流级异步分配 (CUDA 11.2+) | 避免同步开销 |
| `cudaMemPool*` | 内存池 (CUDA 11.2+) | 减少分配/释放开销 |

### 7.3 内存池 (Memory Pool)

```cpp
// 创建内存池
cudaMemPool_t pool;
cudaMemPoolProps props = {};
props.allocType = cudaMemAllocationTypePinned;
props.handleTypes = cudaMemHandleTypeNone;
props.location.type = cudaMemLocationTypeDevice;
props.location.id = 0;
cudaMemPoolCreate(&pool, &props);

// 异步分配/释放 (无同步)
void* ptr;
cudaMallocFromPoolAsync(&ptr, size, pool, stream);
// ... 使用 ...
cudaFreeAsync(ptr, stream);

// 设置释放阈值 (避免归还 OS)
size_t threshold = 1ULL << 30;  // 1 GB
cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
```

> **LLM 推理场景:** 使用 `cudaMallocAsync` + Memory Pool 避免频繁 `cudaMalloc/cudaFree` 导致的 implicit synchronization。

### 7.4 合并访问规则

Global Memory 事务遵循 **Sector** 机制:

```
L1 路径 (默认):
  Warp 32 个线程的请求 → 合并为 128B Cache Line → 4 × 32B Sectors
  仅传输实际触及的 Sectors

L2 路径:
  以 32B Sector 为粒度

效率指标: sectors_per_request
  完美合并 (连续 float): 4 sectors / 1 request = 最优
  最差 (32 个 stride-128B): 32 sectors / 1 request
```

> **详细的合并访问分析见 [coalesced-memory-access.md](coalesced-memory-access.md)**

---

## 8. Pinned Memory (Page-Locked Host Memory)

### 8.1 Pageable vs Pinned Memory

```
Pageable Memory (默认 malloc):            Pinned Memory (cudaMallocHost):
┌────────────┐                           ┌────────────┐
│ Host DRAM  │                           │ Host DRAM  │
│ (可换页)   │                           │ (页锁定)   │
└─────┬──────┘                           └─────┬──────┘
      │ 1. CPU 拷贝到 pinned staging     │     │
      ▼                                       │
┌────────────┐                                │
│ Staging    │                                │
│ Buffer     │                                │
│ (driver    │                                │
│  内部)     │                                │
└─────┬──────┘                                │
      │ 2. DMA 传输                           │ 1. 直接 DMA 传输
      ▼                                       ▼
┌────────────┐                           ┌────────────┐
│ GPU HBM    │                           │ GPU HBM    │
└────────────┘                           └────────────┘
    2 步                                     1 步
    (额外拷贝)                              (直接 DMA)
```

### 8.2 分配 API

```cpp
// 方法 1: cudaMallocHost — 分配 pinned memory
float* h_pinned;
cudaMallocHost(&h_pinned, N * sizeof(float));
// ... 使用 ...
cudaFreeHost(h_pinned);

// 方法 2: cudaHostAlloc — 更多控制选项
float* h_mapped;
cudaHostAlloc(&h_mapped, N * sizeof(float),
    cudaHostAllocDefault      // 普通 pinned
    | cudaHostAllocMapped     // 同时映射到 GPU 地址空间 (Zero-Copy)
    | cudaHostAllocWriteCombined  // Write-Combined (GPU 读更快, CPU 读极慢)
    | cudaHostAllocPortable   // 所有 CUDA Context 可见
);

// 获取 GPU 端指针 (Mapped Memory)
float* d_mapped;
cudaHostGetDevicePointer(&d_mapped, h_mapped, 0);
// Kernel 可直接通过 d_mapped 读取 Host 内存 (通过 PCIe/NVLink)
```

### 8.3 Zero-Copy (Mapped Pinned Memory)

```
GPU Kernel
    │
    ▼ 读/写 d_mapped 指针
┌──────────────┐
│ PCIe / NVLink │  ← 每次访问都经过总线
└──────┬───────┘
       ▼
┌──────────────┐
│ Host DRAM    │  ← 数据物理在 Host
│ (pinned)     │
└──────────────┘

延迟: PCIe 4.0 ~1–5 μs 单向
      NVLink (Grace Hopper) ~0.5 μs
带宽: PCIe 4.0 x16 ~25 GB/s (vs HBM3 3.35 TB/s = 134× 差距)
```

**Zero-Copy 适用场景:**
- 数据仅访问一次 → 不值得拷贝到 HBM
- 数据量远超 GPU 显存 → 流式处理
- CPU-GPU 协作频繁交换少量数据

**Zero-Copy 不适用场景:**
- 数据多次复用 (应拷贝到 HBM)
- 高带宽需求 (PCIe << HBM)
- LLM 推理的权重/KV Cache (应在 HBM)

### 8.4 Pinned Memory 与传输性能

```
传输模式对比 (PCIe 4.0 x16):
┌─────────────────────────┬──────────┬──────────┐
│ 传输方式                 │ 带宽      │ 重叠能力 │
├─────────────────────────┼──────────┼──────────┤
│ Pageable → GPU          │ ~12 GB/s │ 无       │
│ Pinned → GPU            │ ~25 GB/s │ 可重叠   │
│ Pinned → GPU (Async)    │ ~25 GB/s │ 完全重叠 │
│ Write-Combined → GPU    │ ~26 GB/s │ 可重叠   │
└─────────────────────────┴──────────┴──────────┘
```

> **关键:** `cudaMemcpyAsync` 只有在源/目标是 Pinned Memory 时才真正异步。Pageable Memory 的 `cudaMemcpyAsync` 仍然同步 (driver 内部先拷贝到 staging buffer)。

### 8.5 注意事项

- **不要过度分配:** Pinned Memory 锁定物理页, 减少 OS 可用内存, 可能导致系统交换
- **分配开销大:** `cudaMallocHost` 比 `malloc` 慢 10–100×, 应在初始化时分配并复用
- **Write-Combined 陷阱:** `cudaHostAllocWriteCombined` 对 GPU 读最优, 但 CPU 读极慢 (无缓存, 每次读走总线)

---

## 9. Unified Memory (Managed Memory)

### 9.1 基本概念

Unified Memory 提供 CPU 和 GPU 共享的**单一地址空间**，运行时自动在 Host 和 Device 间迁移页面:

```
              Unified Virtual Address Space
              ┌──────────────────────────────────────┐
              │   ptr = cudaMallocManaged(...)        │
              │                                       │
              │   CPU 代码: *ptr = 42;               │ ← 自动在 Host 端
              │   GPU Kernel: val = *ptr;            │ ← 自动迁移到 Device
              │                                       │
              └──────────────────────────────────────┘

Page Migration Engine (Pascal+):
┌──────────┐    Page Fault    ┌──────────┐
│ Host     │  ◄────────────► │ Device   │
│ DRAM     │    (4 KB page   │ HBM      │
│          │     migration)   │          │
└──────────┘                  └──────────┘
```

### 9.2 使用方法

```cpp
// 分配
float* data;
cudaMallocManaged(&data, N * sizeof(float));

// CPU 使用 — 自动在 Host 端
for (int i = 0; i < N; i++) data[i] = i;

// GPU 使用 — 自动迁移到 Device (产生 Page Fault)
kernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();

// CPU 再次使用 — 自动迁移回 Host
printf("%f\n", data[0]);

cudaFree(data);  // 与 cudaMalloc 相同的释放函数
```

### 9.3 Page Fault 与性能代价

```
首次 GPU 访问 Managed Memory 流程:
1. GPU Thread 访问地址 → TLB Miss
2. GPU MMU 产生 Page Fault
3. Page Fault 中断发送到 CPU Driver
4. Driver 确定页面位置 (Host)
5. 启动 DMA: Host → Device (4 KB 页)
6. 更新 GPU Page Table
7. GPU Thread 恢复执行

每个 Page Fault 开销: ~20–50 μs
  vs cudaMemcpy 批量传输: 摊销后 ~ns/page
```

**性能对比 (典型场景):**

| 方法 | 1 GB 传输时间 | 相对性能 |
|------|:-----------:|:-------:|
| `cudaMemcpy` (Pinned → Device) | ~40 ms | 1× (基准) |
| `cudaMallocManaged` (首次 GPU 访问) | ~120–360 ms | 3–9× 慢 |
| `cudaMallocManaged` + `cudaMemPrefetchAsync` | ~45 ms | ~1.1× |

### 9.4 优化提示 (Memory Hints)

```cpp
// ========== 预取: 避免 Page Fault ==========
// 提前将数据迁移到 Device (类似 cudaMemcpy 但更灵活)
cudaMemPrefetchAsync(data, N * sizeof(float), deviceId, stream);

// 预取回 Host
cudaMemPrefetchAsync(data, N * sizeof(float), cudaCpuDeviceId, stream);

// ========== 访问提示: 优化放置策略 ==========

// ReadMostly: CPU 和 GPU 各持一份副本 (读不触发迁移)
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, deviceId);

// PreferredLocation: 建议数据优先放在指定位置
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, deviceId);

// AccessedBy: 建立直接映射, 避免迁移 (可能走 PCIe 远程访问)
cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, deviceId);
```

**各提示的行为对比:**

| Hint | GPU 读 | GPU 写 | CPU 读 | 迁移 | 适用场景 |
|------|--------|--------|--------|------|---------|
| 无 (默认) | fault→迁移 | fault→迁移 | fault→迁移 | 频繁 | — |
| ReadMostly | 本地副本 | 使其他副本失效 | 本地副本 | 仅写时 | CPU+GPU 都读, 很少写 |
| PreferredLocation(GPU) | 本地 | 本地 | fault→远程访问 | 仅初次 | GPU 密集使用 |
| AccessedBy(GPU) | 可远程/本地 | 可远程/本地 | 本地 | 无 | GPU 偶尔访问 |
| Prefetch + Preferred | 本地 | 本地 | fault→迁移 | 预取 | 最佳 GPU 性能 |

### 9.5 硬件一致性演进

```
Pascal/Volta/Ampere/Hopper (离散 GPU):
  ┌────────┐  PCIe/NVLink  ┌────────┐
  │  CPU   │ ◄───────────► │  GPU   │   软件一致性: Page Migration
  │ (Host) │               │(Device)│   Page Fault 驱动
  └────────┘               └────────┘

Grace Hopper (统一架构):
  ┌────────────────────────────────────┐
  │     Grace CPU ←── NVLink-C2C ──→ Hopper GPU  │
  │                900 GB/s           │
  │     硬件一致性 (ATS/HMM)          │   原生缓存一致性
  │     无 Page Fault 开销            │   GPU 可直接访问 CPU 缓存
  └────────────────────────────────────┘
```

- **HMM (Heterogeneous Memory Management):** Linux 6.1+ 内核支持, GPU 复用 CPU 页表, 减少 driver 介入
- **ATS (Address Translation Services):** Grace Hopper 硬件一致性, GPU 通过 NVLink-C2C 直接访问 CPU 缓存, 延迟 ~100 ns

### 9.6 Unified Memory 的局限性

| 局限 | 说明 |
|------|------|
| **Page Fault 开销** | 每次 4 KB 页迁移 ~20–50 μs, 远慢于 cudaMemcpy 批量传输 |
| **页面颠簸** | CPU/GPU 交替访问同一页面 → 反复迁移 → 性能灾难 |
| **不可预测延迟** | Page Fault 在 Kernel 执行中发生, 阻塞触发 Fault 的 Warp (其他 Warp 可继续执行) |
| **TLB 压力** | 大规模 Managed Memory 增加 GPU TLB Miss |
| **Multi-GPU** | 不自动处理 Multi-GPU 场景的最优放置 |

> **LLM 推理建议:** 生产环境 LLM 推理 Kernel 应使用 `cudaMalloc` + 显式 `cudaMemcpy`。Unified Memory 适合原型开发和 CPU-GPU 协作的分析工作负载。

---

## 10. Tensor Memory — TMEM (Blackwell)

### 10.1 TMEM 概述

Blackwell DC (SM 10.0) 引入 **Tensor Memory (TMEM)**, 一种新的 256 KB/SM 片上存储, 专为 Tensor Core (`tcgen05`) 指令设计:

```
Blackwell SM 10.0 存储层次:
┌──────────────────────────────────────────────┐
│  Register File (256 KB/SM)                    │  ← 通用计算
├──────────────────────────────────────────────┤
│  Tensor Memory — TMEM (256 KB/SM)            │  ← 新增! tcgen05 专用
│  ┌─────────────────────────────────────────┐ │
│  │  累加器/结果存储                        │ │
│  │  解放 RF 中的 MMA 累加器寄存器         │ │
│  │  仅 tcgen05 指令可访问                 │ │
│  └─────────────────────────────────────────┘ │
├──────────────────────────────────────────────┤
│  Shared Memory (最大 228 KB/SM)              │  ← 不变
├──────────────────────────────────────────────┤
│  L1 Data Cache (与 SMEM 共享 256 KB 池)      │  ← 不变
└──────────────────────────────────────────────┘
```

### 10.2 TMEM vs Registers

| 对比 | Registers (RF) | TMEM |
|------|:--------------:|:----:|
| 容量/SM | 256 KB | 256 KB |
| 延迟 | ~1 cycle | ~1 cycle |
| 访问方式 | 所有指令 | 仅 tcgen05 |
| 用途 | 通用计算 | MMA 累加器 + 操作数 |
| 分配 | 编译器静态 | 显式 tcgen05.alloc/dealloc |
| 可用架构 | 所有 | Blackwell DC (SM 10.0) 仅 |

### 10.3 TMEM 对 LLM Kernel 的意义

**Hopper GEMM 的寄存器困境:**
- WGMMA 累加器: 64–256 个寄存器 (取决于 tile 大小)
- 输入 Fragment: 额外寄存器
- 流控 + 地址计算: 更多寄存器
- → Occupancy 受限 (通常 25–50%)

**Blackwell TMEM 解决方案:**
- 累加器放 TMEM (256 KB) → 释放 RF 中数百个寄存器
- RF 专注于标量计算 + 地址 + 流控
- → Occupancy 大幅提升 or 更大 tile

```
Hopper:  RF = 累加器(60%) + 计算(20%) + 地址(20%) → RF 紧张
Blackwell: RF = 计算(50%) + 地址(50%)              → RF 宽裕
           TMEM = 累加器(100%)                      → 专用
```

### 10.4 Double-Buffered TMEM

```
TMEM Double Buffering:
┌───────────────┬───────────────┐
│  TMEM Buf 0   │  TMEM Buf 1   │
│  (tcgen05     │  (epilogue    │
│   写累加器)    │   读结果)     │
└───────────────┴───────────────┘
         ↕ 交替
Producer Warpgroup: tcgen05 → TMEM Buf 0
Consumer Warpgroup: 读 TMEM Buf 1 → RF → epilogue 计算
```

> **SM 12.0 (RTX 5090):** 不支持 TMEM/tcgen05, 使用传统 mma.sync + RF 累加器。

---

## 11. 异步数据传输与拷贝

### 11.1 Host-Device 传输

```
传输模式演进:
┌─────────────────────────────────────────────────────┐
│ 同步: cudaMemcpy → 阻塞 CPU, 无法重叠              │
│                                                      │
│ 异步: cudaMemcpyAsync(stream) → CPU 不阻塞         │
│       但需 Pinned Memory 才真正异步                  │
│                                                      │
│ 计算/传输重叠:                                       │
│  Stream 1: ├── Copy H→D ──┤                         │
│  Stream 2:          ├── Kernel ──┤                   │
│  Stream 3:                  ├── Copy D→H ──┤        │
└─────────────────────────────────────────────────────┘
```

```cpp
// 计算/传输重叠的三流模式
cudaStream_t s1, s2, s3;
cudaStreamCreate(&s1); cudaStreamCreate(&s2); cudaStreamCreate(&s3);

for (int i = 0; i < num_chunks; i++) {
    cudaMemcpyAsync(d_in + i*chunk, h_in + i*chunk,
                    chunk_bytes, cudaMemcpyHostToDevice, s1);
    kernel<<<grid, block, 0, s2>>>(d_in + i*chunk, d_out + i*chunk);
    cudaMemcpyAsync(h_out + i*chunk, d_out + i*chunk,
                    chunk_bytes, cudaMemcpyDeviceToHost, s3);
}
```

### 11.2 Global → Shared Memory 异步拷贝

| 阶段 | 机制 | 特点 |
|------|------|------|
| **Pre-Ampere** | `smem[i] = gmem[i]` | 经过寄存器, 增加 RF 压力 |
| **Ampere (cp.async)** | `cp.async.ca.shared.global` | 绕过寄存器, 减少 RF 压力, 多 stage pipeline |
| **Hopper (TMA)** | `cp.async.bulk.tensor` | 硬件计算地址, 批量传输, 硬件 Swizzle |

```
Pre-Ampere:            Ampere cp.async:          Hopper TMA:
Global                 Global                    Global
  │                      │                         │
  ▼                      │                         │
Registers              ╳ (bypass)                 ╳ (bypass)
  │                      │                         │
  ▼                      ▼                         ▼
Shared Mem             Shared Mem                 Shared Mem
                                                  (+ hardware swizzle)
```

### 11.3 cp.async 详解 (Ampere)

```cpp
// 单次 16 字节拷贝 (float4 等效)
asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
    :: "r"(smem_addr), "l"(gmem_ptr));

// cp.async.cg — 绕过 L1 缓存 (仅 L2)
asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
    :: "r"(smem_addr), "l"(gmem_ptr));

// 提交并等待
asm volatile("cp.async.commit_group;\n");
asm volatile("cp.async.wait_group 0;\n");  // 等待所有完成
// wait_group N: 允许 N 个 group 仍在 flight
```

**cp.async 支持的拷贝大小:** 4, 8, 16 字节

### 11.4 TMA 详解 (Hopper)

```cpp
// 1. Host 端创建 Tensor Map 描述符
CUtensorMap tensorMap;
cuTensorMapEncodeTiled(&tensorMap,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    2,                          // 2D tensor
    d_ptr,                      // global pointer
    globalDims, globalStrides,
    boxDims, elementStrides,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B, // 硬件 Swizzle
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

// 2. Kernel 内: 单条指令加载整个 tile
// 仅 1 个线程执行 (通常 threadIdx.x == 0)
if (threadIdx.x == 0) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_addr), "l"(&tensorMap),
           "r"(coord_x), "r"(coord_y), "r"(mbar_addr));
}
```

> **详细的 TMA 文档见 [tensor-memory-accelerator.md](tensor-memory-accelerator.md)**

---

## 12. LLM Kernel 存储优化实战

### 12.1 GEMM (Tensor Core)

```
存储使用策略:
┌──────────────────────────────────────────┐
│ Global (HBM): A[M×K], B[K×N] 矩阵       │
│       │                                   │
│       ▼ TMA / cp.async (多 stage)        │
│ Shared Memory: A_tile[BM×BK], B_tile[BK×BN] │
│       │ (Swizzle 消除 Bank Conflict)     │
│       ▼ ldmatrix / 寄存器加载            │
│ Registers: Fragment A, Fragment B         │
│       │                                   │
│       ▼ mma.sync / wgmma / tcgen05       │
│ Registers / TMEM: 累加器 C[BM×BN]        │
│       │                                   │
│       ▼ Epilogue (bias + activation)     │
│ Global (HBM): D[M×N] 输出               │
└──────────────────────────────────────────┘
```

**各架构存储分配:**

| 架构 | SMEM 用量 | 累加器 | Pipeline |
|------|:--------:|:------:|:--------:|
| Ampere | ~2 × (BM×BK + BK×BN) × 2B | RF (主要瓶颈) | cp.async 3-stage |
| Hopper | ~3 × (BM×BK + BK×BN) × 2B | RF (wgmma) | TMA 4-stage + WarpSpec |
| Blackwell | ~3 × (BM×BK + BK×BN) × 2B | TMEM | TMA + tcgen05 |

### 12.2 FlashAttention

```
存储关键挑战: O(N²) Attention → O(N) SMEM 分块

每个 Block 的 SMEM 使用:
┌──────────────────────────────────┐
│  Q_tile [Br × d]     (常驻)     │  ← 加载一次, 反复使用
│  K_tile [Bc × d]     (流式)     │  ← 每次迭代替换
│  V_tile [Bc × d]     (流式)     │  ← 每次迭代替换
│  S_tile [Br × Bc]    (临时)     │  ← QK^T 结果, 可复用空间
└──────────────────────────────────┘

寄存器使用:
  O_acc [Br × d]       ← 在线累加输出
  m_prev, l_prev       ← Online Softmax 状态 (max, sum)
  m_new, l_new         ← 当前块的 max, sum
```

**SMEM 预算估算 (FP16, Br=Bc=128, d=128):**
- Q_tile: 128 × 128 × 2B = 32 KB
- K_tile: 128 × 128 × 2B = 32 KB
- V_tile: 128 × 128 × 2B = 32 KB
- 总计: ~96 KB (单 stage), ~160 KB (双 stage)

### 12.3 RMSNorm / LayerNorm

```
存储策略: 向量化加载 + Block Reduction

Global → Registers (float4 向量化加载)
                │
                ▼
         Shared Memory (Block Reduction)
                │
                ▼ (计算 mean/var 或 rms)
         Registers (归一化 + scale)
                │
                ▼
         Global (写回)

SMEM 用量: ~blockDim.x × sizeof(float) (用于 reduction)
寄存器: ~30–50 regs (向量化临时 + 累加)
```

### 12.4 Embedding Lookup

```
存储选择:
  Embedding Table [V × D]:
    V = 32000–128000 (vocab), D = 4096–8192 (hidden dim)
    总大小: 32000 × 4096 × 2B ≈ 250 MB (FP16)
    → 必须在 Global Memory (HBM)

  访问模式:
    Token IDs → 随机索引 → 不同行
    同一 Warp 内线程可能访问完全不同的行 → Non-Coalesced

  优化:
    1. __ldg() 走 Read-Only Cache (只读)
    2. 向量化加载 (float4) 减少指令数
    3. Warp 内线程协作加载同一行的不同列 → Coalesced
```

### 12.5 KV Cache 管理

```
KV Cache 存储层次决策:

Prefill 阶段:
  K, V 新生成 → 写入 HBM KV Cache (连续内存)
  Attention 计算 → K, V 加载到 SMEM 进行 FlashAttention

Decode 阶段 (逐 token):
  新 K, V: 1 token → 追加到 HBM KV Cache
  历史 KV: 所有 past tokens → 从 HBM 加载

  存储优化:
  ┌─────────────────────────────────────────┐
  │ L2 Persistence: 热 KV 段驻留 L2         │
  │   cudaAccessPolicyWindow → hitRatio=1.0 │
  │   短序列 KV 可完全驻留 L2 (126 MB)      │
  ├─────────────────────────────────────────┤
  │ __ldg(): KV 在 Decode 中只读            │
  │   Read-Only Cache 不被 store 驱逐       │
  ├─────────────────────────────────────────┤
  │ PagedAttention: 非连续 KV 块管理        │
  │   每块 (block_size × num_heads × d)     │
  │   块表 (Block Table) 存 Constant/Global │
  └─────────────────────────────────────────┘
```

### 12.6 量化 Kernel

```
存储使用模式:

INT8/FP8 GEMM Dequant:
  权重 (INT8): Global → SMEM → Registers → Tensor Core (mma.sync)
  Scale/Zero:
    Per-Tensor: Constant Memory (广播, ~1 cycle)
    Per-Channel: Global + __ldg() (每线程不同)
    Per-Group: Global + SMEM 缓存 (group 内共享)

GPTQ/AWQ (INT4):
  打包权重: 8 个 INT4 → 1 个 int32 → 寄存器解包
  Scale+Zero: Per-Group → __ldg() 或 SMEM 缓存
  解包后: Registers → FP16 累加

  寄存器压力: 打包/解包 临时变量 + scale 运算 → 60–80 regs
```

---

## 13. 存储选型决策指南

### 13.1 决策流程图

```
数据特征分析
     │
     ├─ 数据大小?
     │   ├─ < 8 KB, 全 Warp 统一访问 → Constant Memory
     │   ├─ < 228 KB/Block, 多次复用 → Shared Memory
     │   ├─ < 126 MB, 多 Kernel 复用 → L2 Persistence
     │   └─ 其他 → Global Memory (HBM)
     │
     ├─ 读写模式?
     │   ├─ Device 只读 → __ldg() / const __restrict__ / Texture
     │   ├─ 读写 + Block 内共享 → Shared Memory
     │   ├─ 读写 + 全局共享 → Global Memory + Atomic
     │   └─ CPU+GPU 共享 → Pinned (显式) 或 Unified (自动)
     │
     ├─ 访问模式?
     │   ├─ 连续/合并 → Global Memory (直接)
     │   ├─ Stride/随机 → 先加载到 SMEM, 再随机访问
     │   └─ 2D 空间局部性 → Texture Cache
     │
     └─ 生命周期?
         ├─ Kernel 内临时 → Registers 或 SMEM
         ├─ 跨 Kernel → Global Memory
         └─ 跨 Host/Device → Pinned 或 Unified
```

### 13.2 存储类型选型速查表

| 数据类型 | 推荐存储 | 原因 | 避免 |
|---------|---------|------|------|
| GEMM 输入 tile | SMEM (多 stage) | 高复用, Bank Conflict 可控 | 直接从 HBM 多次读 |
| MMA 累加器 | RF (Hopper) / TMEM (Blackwell) | ~1 cycle | SMEM (太慢) |
| Softmax 临时 (max, sum) | RF | 线程私有, 频繁更新 | SMEM (无需共享) |
| Block Reduction | SMEM | 线程间通信必需 | Atomic (太慢) |
| 量化 per-tensor scale | Constant Memory | 全 Warp 统一, ~1 cycle | Global (__ldg) |
| 量化 per-channel scale | Global + __ldg() | 每线程不同地址 | Constant (串行化) |
| 权重矩阵 | HBM → SMEM tiling | 太大, 无法驻留 on-chip | Unified Memory |
| KV Cache (Decode) | HBM + L2 Persistence + __ldg() | 只读, 跨 Kernel | 普通 ld.global |
| Embedding 表 | HBM + __ldg() | 大且随机访问 | Constant (>64KB) |
| 超参数 (eps, lr) | Constant | 小, 全局统一 | Registers (浪费) |
| Host↔Device 传输 | Pinned Memory | 异步 DMA, 无额外拷贝 | Pageable malloc |
| 原型开发 | Unified Memory | 简化代码 | 生产推理 |

### 13.3 带宽与延迟速查表

```
存储层次带宽 (H100 SXM 参考):

层次              │ 延迟 (cycles) │ 带宽/SM         │ 聚合带宽
──────────────────┼──────────────┼────────────────┼───────────
Register File     │     ~1       │ ~1 cyc, 非带宽受限 │    —
Shared Memory     │   ~20–30     │  ~200 GB/s      │  ~27 TB/s
L1 Cache          │   ~28–35     │  ~200 GB/s      │  ~27 TB/s
L2 Cache          │  ~150–273    │     —           │  ~12 TB/s
HBM (Global)      │  ~400–800    │  ~25 GB/s       │  3.35 TB/s
PCIe 5.0 x16      │  ~10K–50K    │     —           │  ~63 GB/s (单向)
NVLink 4.0        │   ~1K–5K     │     —           │  900 GB/s (双向)
```

> **注：** L1/SMEM 带宽 = 128 Bytes/cycle × SM 时钟。per-SM DRAM 带宽 = 3.35 TB/s ÷ 132 SMs ≈ 25 GB/s。L2 ~12 TB/s 为 NVIDIA 标称值，实测可达带宽约 5–6 TB/s (取决于访问模式)。

---

## 14. 诊断与分析

### 14.1 编译时信息

```bash
# 查看寄存器和 Shared Memory 使用
nvcc --ptxas-options=-v kernel.cu
# 输出:
# ptxas info: Used 128 registers, 49152 bytes smem, 384 bytes cmem[0]
# ptxas info: 0 bytes spill stores, 0 bytes spill loads

# cmem[0] — Kernel 参数 (自动放 Constant Memory)
# cmem[2] — __constant__ 变量
```

### 14.2 Nsight Compute 关键 Metrics

| Metric | 含义 | 目标 |
|--------|------|------|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Occupancy | 根据 Kernel 类型判断 |
| `l1tex__t_bytes_pipe_lsu_mem_local_op_ld` | Local Memory 读 (spill) | 趋近 0 |
| `l1tex__t_bytes_pipe_lsu_mem_local_op_st` | Local Memory 写 (spill) | 趋近 0 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` | SMEM Bank Conflict | 趋近 0 |
| `lts__t_sectors_srcunit_tex_op_read_hit_rate.pct` | L2 命中率 | > 80% |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | Global Load Sectors | 越少越好 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | Global Store Sectors | 越少越好 |
| `dram__bytes_read.sum` | HBM 实际读字节 | 接近理论最小 |
| `smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active` | LSU 管线利用率 | 判断访存瓶颈 |

### 14.3 常见诊断场景

```
场景 1: Kernel 性能低, 怀疑 Memory-Bound
  → 检查 dram__bytes_read vs 理论最小 → 如果远大于理论 → 合并度差
  → 检查 l1tex__t_sectors_pipe_lsu_mem_global_op_ld_sectors_per_request → 理想 4.0

场景 2: 高寄存器使用, 低 Occupancy
  → nvcc -v 查看 regs/thread
  → 检查 spill loads/stores 是否非零
  → 考虑 __launch_bounds__ 或代码重构

场景 3: SMEM Bank Conflict
  → ncu 检查 l1tex__data_bank_conflicts_pipe_lsu_mem_shared
  → 分析 stride pattern → Padding 或 Swizzle

场景 4: Unified Memory 性能差
  → 检查 GPU Page Fault 次数
  → 添加 cudaMemPrefetchAsync
  → 考虑切换到显式 cudaMemcpy
```

---

## 15. 存储优化检查清单

### 15.1 Registers

- [ ] `nvcc -v` 查看寄存器用量, 确认无意外 spilling
- [ ] 使用 `__launch_bounds__` 指导编译器寄存器分配
- [ ] 避免 `int64` 循环变量/索引 (隐式双倍寄存器)
- [ ] 大数组用 Shared Memory 替代寄存器数组
- [ ] 考虑重计算替代临时变量存储

### 15.2 Shared Memory

- [ ] 使用动态分配 (`extern __shared__`) 超过 48 KB
- [ ] 调用 `cudaFuncSetAttribute` 设置最大 SMEM 大小
- [ ] 检查 Bank Conflict → 使用 Padding 或 Swizzle
- [ ] 实现 Double Buffering 或 Multistage Pipeline
- [ ] Ampere 使用 `cp.async`, Hopper 使用 TMA
- [ ] SMEM vs L1 Carveout 根据 Kernel 类型选择

### 15.3 Constant Memory

- [ ] 全 Warp 统一读取的小数据 (< 64 KB) 使用 `__constant__`
- [ ] 非统一读取使用 `__ldg()` 替代 (避免串行化)
- [ ] Kernel 参数自动走 Constant Memory (无需显式)

### 15.4 Global Memory

- [ ] 确保 Coalesced 访问: 连续线程访问连续地址
- [ ] 使用向量化加载 (`float4`, `uint4`) 减少指令数
- [ ] 只读数据标注 `const __restrict__` 或显式 `__ldg()`
- [ ] 考虑 `cudaMallocAsync` + Memory Pool 避免同步
- [ ] 大矩阵使用 SMEM Tiling 提高复用

### 15.5 Host-Device 传输

- [ ] 使用 Pinned Memory (`cudaMallocHost`) 而非 `malloc`
- [ ] `cudaMemcpyAsync` + 多 Stream 实现计算/传输重叠
- [ ] 大传输分块 (chunked) 以实现 pipeline
- [ ] Write-Combined Pinned 用于 GPU 只读数据
- [ ] 生产 LLM 推理避免 Unified Memory

### 15.6 Cache 优化

- [ ] L2 Persistence 用于热数据 (KV Cache, 频繁 Kernel 间共享)
- [ ] Prefetch 提示 (`prefetch.global.L2`) 减少 miss
- [ ] Threadblock Swizzle 优化 L2 空间局部性
- [ ] 考虑 Kernel Fusion 减少中间结果的 HBM 往返

### 15.7 架构特定

- [ ] **Ampere:** `cp.async` 异步 Global→SMEM, 3+ stage pipeline
- [ ] **Hopper:** TMA 批量传输, DSMEM (Cluster), Warp Specialization
- [ ] **Blackwell DC:** TMEM 累加器, tcgen05, TMA 增强
- [ ] **Blackwell Consumer (SM 12.0):** mma.sync 路径, 无 TMEM/tcgen05