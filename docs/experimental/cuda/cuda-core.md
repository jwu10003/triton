# CUDA Core 硬件架构与编程模型

> 面向 LLM 高性能 Kernel 开发的 CUDA Core 深度解析
> 覆盖 SM 内部组织、SIMT 执行模型、Warp 调度、指令吞吐、架构演进

---

## 目录

1. [CUDA Core 概述](#1-cuda-core-概述)
2. [SM 内部组织](#2-sm-内部组织)
3. [SIMT 执行模型](#3-simt-执行模型)
4. [Warp 调度与延迟隐藏](#4-warp-调度与延迟隐藏)
5. [FP32 / INT32 数据通路演进](#5-fp32--int32-数据通路演进)
6. [指令吞吐量](#6-指令吞吐量)
7. [内存层级与 CUDA Core 交互](#7-内存层级与-cuda-core-交互)
8. [架构演进总览](#8-架构演进总览)
9. [CUDA Core vs Tensor Core](#9-cuda-core-vs-tensor-core)
10. [LLM Kernel 中的 CUDA Core 使用场景](#10-llm-kernel-中的-cuda-core-使用场景)

---

## 1. CUDA Core 概述

### 1.1 什么是 CUDA Core

CUDA Core 是 NVIDIA GPU 中的**标量处理单元**，每个 CUDA Core 每时钟周期执行一次浮点 (FP32) 或整数 (INT32) 运算。它是 GPU 通用计算的基本执行单元。

```
CPU Core:    复杂乱序流水线, 深度分支预测, 1-2 个线程
CUDA Core:   简单顺序流水线, 无分支预测, 靠海量线程切换隐藏延迟
```

### 1.2 历史渊源

2006 年 GeForce 8800 统一了顶点着色器和像素着色器为**统一着色器 (Unified Shader)**。CUDA Core 就是这些统一着色器——与 AMD "Stream Processor" 本质相同。NVIDIA 在 CUDA (Compute Unified Device Architecture) 平台中将它们暴露给通用计算。

### 1.3 NVIDIA 的 "CUDA Core" 计数方式

NVIDIA 官方以 **FP32 执行单元数量** 统计 CUDA Core 数：

| SM 类型 | FP32 单元 | INT32 单元 | NVIDIA 标称 CUDA Cores |
|---------|----------|----------|----------------------|
| Volta SM | 64 FP32 | 64 INT32 | **64** |
| Hopper SM | 128 FP32 | 64 INT32 | **128** |
| Ampere GA10x SM | 64 FP32 (专用) + 64 FP32/INT32 (双用) | — | **128** |

> INT32、FP64、SFU 等执行单元**不计入** CUDA Core 数量。

### 1.4 在 GPU 中的层级位置

```
GPU (整个芯片)
├── GPC (Graphics Processing Cluster) × N
│   ├── TPC (Texture Processing Cluster) × M
│   │   └── SM (Streaming Multiprocessor) × 1-2
│   │       ├── Sub-partition × 4
│   │       │   ├── CUDA Cores (FP32)     ← 本文主题
│   │       │   ├── INT32 单元
│   │       │   ├── FP64 单元 (数据中心 GPU)
│   │       │   ├── SFU (Special Function Units)
│   │       │   ├── Load/Store Units
│   │       │   ├── Tensor Core
│   │       │   ├── Warp Scheduler
│   │       │   └── Register File (64 KB)
│   │       ├── Shared Memory / L1 Cache
│   │       └── L0 Instruction Cache
│   └── Raster Engine (图形管线)
├── L2 Cache
└── Memory Controllers → HBM / GDDR
```

---

## 2. SM 内部组织

### 2.1 Sub-partition (处理块)

从 Maxwell 起，每个 SM 被划分为 **4 个 Sub-partition** (也称 Processing Block / Quadrant / SMSP)。每个 Sub-partition 是半独立的执行引擎：

#### Volta (CC 7.0) Sub-partition 结构

```
Sub-partition (1/4 SM)
├── Warp Scheduler × 1       ─ 每周期选择 1 个 eligible warp
├── Dispatch Unit × 1        ─ 每周期发射 1 条指令
├── Register File: 16,384 × 32-bit (64 KB)
├── FP32 Units × 16          ─ 16 个 CUDA Core
├── INT32 Units × 16          ─ 16 个独立整数单元
├── FP64 Units × 8            ─ 8 个双精度单元
├── SFU × 4                   ─ sin/cos/rsqrt/rcp
├── Load/Store Units × 8      ─ 地址生成 + 内存请求
├── Tensor Core × 2           ─ MMA 加速器
└── L0 Instruction Cache      ─ 指令缓存
```

#### 各架构 Sub-partition 对比

| 组件 | Volta 7.0 | Turing 7.5 | Ampere 8.0 | Ampere 8.6 | Ada 8.9 | Hopper 9.0 |
|------|----------|----------|----------|----------|---------|----------|
| FP32 单元 | 16 | 16 | 16 | 32* | 32* | 32 |
| INT32 单元 | 16 | 16 | 16 | 16* | 16* | 16 |
| FP64 单元 | 8 | ~0 | 8 | ~0 | ~0 | 16 |
| SFU | 4 | 4 | 4 | 4 | 4 | 4 |
| LD/ST 单元 | 8 | 4 | 8 | 8 | 8 | 8 |
| Tensor Core | 2 | 2 | 1 | 1 | 1 | 1 |
| Warp Scheduler | 1 | 1 | 1 | 1 | 1 | 1 |
| Register File | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB |

> *CC 8.6/8.9: 第二通路中 16 个 FP32/INT32 双用单元，可执行 FP32 或 INT32 但不能同时执行。

### 2.2 Register File (寄存器堆)

寄存器堆是 SM 中**最大的片上存储**，也是 CUDA Core 的直接操作数来源：

| 属性 | 值 |
|------|-----|
| 每 SM 总寄存器数 | 65,536 × 32-bit = **256 KB** |
| 每 Sub-partition | 16,384 × 32-bit = 64 KB |
| 每线程最大寄存器数 | **255** |
| 访问延迟 | **1 个时钟周期** |
| 分配方式 | Kernel 启动时静态分配，Warp 切换**零开销** |

```
寄存器使用 vs Occupancy 权衡:

  32 regs/thread → 2048 threads (64 warps) → 100% occupancy
  64 regs/thread → 1024 threads (32 warps) → 50% occupancy
 128 regs/thread → 512 threads  (16 warps) → 25% occupancy
 255 regs/thread → 256 threads  (8 warps)  → 12.5% occupancy
```

> 寄存器是每线程**私有**的——无需 save/restore，因此 Warp 切换完全无开销。

### 2.3 Shared Memory / L1 Cache

每 SM 有统一的 L1 + Shared Memory 缓存池，可配置分配比例：

| 架构 | L1 + Shared 总量 | 最大 Shared Memory |
|------|---------------|-----------------|
| Volta (CC 7.0) | 128 KB | 96 KB |
| Turing (CC 7.5) | 96 KB | 64 KB |
| Ampere (CC 8.0) | 192 KB | 164 KB |
| Ampere (CC 8.6) | 128 KB | 100 KB |
| Ada (CC 8.9) | 128 KB | 100 KB |
| Hopper (CC 9.0) | 256 KB | 228 KB |
| Blackwell (CC 10.0) | 256 KB | 228 KB |
| Blackwell (CC 12.0) | 128 KB | 128 KB |

---

## 3. SIMT 执行模型

### 3.1 Warp：32 线程为一组

GPU 将线程组织为 **Warp**，每个 Warp 包含 **32 个线程**。Warp 是 GPU 调度和执行的基本单位。

```
Thread Block (CTA)
├── Warp 0:  threads [0,  31]
├── Warp 1:  threads [32, 63]
├── Warp 2:  threads [64, 95]
└── ...

若 Block 有 256 个线程 → 8 个 Warp
若 Block 有 100 个线程 → 4 个 Warp (最后 Warp 仅 4 线程活跃, 28 线程浪费)
```

### 3.2 SIMT vs SIMD

| 特性 | SIMD (CPU AVX-512) | SIMT (CUDA Warp) |
|------|-------------------|------------------|
| 宽度 | 编译时固定 (128/256/512-bit) | 固定 32 线程 |
| 线程身份 | 无独立线程概念 | 每线程有 `threadIdx`、独立寄存器 |
| 分支 | 标量分支 + 向量 mask | 硬件自动 mask + 重汇合 |
| 内存访问 | 连续/gather/scatter | 每线程独立地址 (coalescing 由 LD/ST 硬件完成) |
| 程序计数器 | 共享 | Volta+ 每线程独立 PC |

### 3.3 Warp Divergence (分支分歧)

当 Warp 内线程走不同分支时：

```cpp
if (threadIdx.x < 16) {
    // Path A: 前 16 个线程执行, 后 16 个线程被 mask 掉
    a = compute_a();
} else {
    // Path B: 后 16 个线程执行, 前 16 个线程被 mask 掉
    a = compute_b();
}
// 重汇合: 所有 32 线程继续
```

**性能影响：**
- 两个分支被**串行化**执行：Warp 先执行 Path A (mask 后半)，再执行 Path B (mask 前半)
- 简单 if/else → 2× 时间
- Divergence 仅影响 **Warp 内** 线程；不同 Warp 间互不影响

**优化策略：**
- 按 Warp 对齐分支条件 (使同一 Warp 内所有线程走同一路径)
- 编译器对短分支使用 **predicated execution** (谓词执行)，避免实际分支

### 3.4 Independent Thread Scheduling (Volta+)

Volta 前：整个 Warp 共享一个程序计数器 (PC)。
Volta 起：每个线程拥有独立 PC 和调用栈。

```
Pre-Volta:   Warp PC ─────────────────────→ 所有线程必须重汇合
Volta+:      T0 PC ──→ ╲
             T1 PC ──→  ╲
             T2 PC ──→   ├─ 硬件调度优化器动态分组活跃线程
             ...         │
             T31 PC ──→ ╱
```

**影响：**
- 支持 Warp 内线程间细粒度同步 (`__syncwarp()`)
- 允许 Sub-warp 级别的分歧和重汇合
- 实现无饥饿 (starvation-free) 算法

---

## 4. Warp 调度与延迟隐藏

### 4.1 Warp Scheduler 工作流程

每个 Sub-partition 有 1 个 Warp Scheduler，每时钟周期执行：

```
1. 检查所有驻留 Warp 的状态
   ┌─────────────────────────────────────────┐
   │ Warp 状态:                              │
   │  • Stalled:  等待内存/寄存器依赖/同步屏障 │
   │  • Eligible: 指令已取回, 操作数就绪,      │
   │              功能单元可用                  │
   │  • Selected: 被选中执行                   │
   └─────────────────────────────────────────┘
2. 从 Eligible Warps 中选择一个 (最老优先 / 轮询策略)
3. 发射该 Warp 的下一条指令到功能单元
4. 该指令在 32 个活跃 Lane 上并行执行
```

### 4.2 零开销 Warp 切换

| 上下文切换 | 开销 |
|-----------|------|
| CPU 线程切换 | 数百~数千时钟周期 (save/restore 寄存器、TLB flush) |
| GPU Warp 切换 | **0 时钟周期** (寄存器常驻、无需 save/restore) |

**原因：** 每个线程的寄存器在 Kernel 启动时就被永久分配在 Register File 中，Warp Scheduler 只是切换了 "当前执行" 的指针。

### 4.3 延迟隐藏 (Latency Hiding)

这是 GPU 性能的**核心机制**——靠大量并发 Warp 覆盖内存访问延迟：

```
时间线 (每行代表一个时钟周期):

Cycle 0:   Warp A 执行 FP32 指令
Cycle 1:   Warp A 发起 Global Memory Load (延迟 ~400 cycles)
Cycle 2:   Warp A stalled → 切换到 Warp B (零开销)
Cycle 3:   Warp B 执行 FP32 指令
Cycle 4:   Warp B 执行 FP32 指令
...
Cycle 401: Warp A 的数据到达 → Warp A 变为 Eligible
Cycle 402: Warp A 继续执行
```

**所需 Warp 数量估算：**

```
需要的活跃 Warp 数 ≈ 延迟 (cycles) / 每 Warp 指令间隔 (cycles)

例: Global Memory 延迟 400 cycles, FP32 指令延迟 4 cycles
    → 需 ~100 Warp 完美隐藏延迟
    → 实际因 ILP 存在, 通常 32-64 Warp 已足够
```

### 4.4 Occupancy (占用率)

```
Occupancy = 活跃 Warp 数 / SM 支持的最大 Warp 数
```

| 架构 | 每 SM 最大 Warp 数 | 最大线程数 |
|------|------------------|---------|
| Volta ~ Hopper (CC 7.0–9.0) | **64** | 2,048 |
| Blackwell (CC 10.0) | **64** | 2,048 |
| Blackwell (CC 12.0) | **48** | 1,536 |

**限制 Occupancy 的三个资源（取最小值）：**

1. **寄存器数/线程：** 65,536 regs ÷ (threads/SM) → 寄存器多的 Kernel 减少并发线程
2. **Shared Memory/Block：** SM 的 Shared Memory 有限，大 Block 可能只能放少量 Block
3. **Block 数/SM：** 最大 32 个 Block/SM (Hopper+)，小 Block 可能因此受限

> 高 Occupancy ≠ 一定高性能。有时降低 Occupancy 以使用更多寄存器 (减少 spill) 反而更快。

---

## 5. FP32 / INT32 数据通路演进

### 5.1 Pre-Volta (Pascal 及更早)：共享通路

```
Pascal SM 每个 CUDA Core:
┌─────────────────┐
│  统一 ALU       │
│  FP32 或 INT32  │ ← 每周期只能执行一种
│  (共享流水线)    │
└─────────────────┘

影响: 循环中常见的 FP32 计算 + INT32 地址运算竞争同一单元
     → INT32 指令"偷走" FP32 执行槽
```

### 5.2 Volta / Turing / Ampere A100：独立分离通路

```
Volta SM (每 Sub-partition):
┌──────────┐  ┌──────────┐
│ FP32 ×16 │  │ INT32 ×16│ ← 物理分离, 每周期可同时执行
│ (专用)   │  │ (专用)   │
└──────────┘  └──────────┘
     ↓              ↓
每 SM 每周期: 64 FP32 + 64 INT32 并发
```

**动机：** 典型着色器/计算 Kernel 中，每 100 条 FP32 指令约伴随 36 条 INT32 指令 (地址计算、循环计数等)。分离后 INT32 执行几乎 "免费"。

### 5.3 Ampere GA10x / Ada：双用通路 (FP32 翻倍)

```
Ampere GA10x SM (每 Sub-partition):
┌──────────┐  ┌─────────────┐
│ FP32 ×16 │  │ FP32/INT32  │ ← 可执行 FP32 或 INT32, 不能同时
│ (专用)   │  │ ×16 (双用)  │
└──────────┘  └─────────────┘

模式选择:
  纯 FP32 模式:  32 FP32/sub-partition = 128 FP32/SM (吞吐翻倍)
  混合模式:      16 FP32 + 16 INT32 /sub-partition = 64 FP32 + 64 INT32 /SM
  不能:          128 FP32 + 64 INT32 同时
```

NVIDIA 因此将 GA10x 的 "CUDA Core" 数标为 128/SM (两组 FP32 能力之和)。

### 5.4 Hopper：回归专用分离 + 规模翻倍

```
Hopper SM (每 Sub-partition):
┌──────────┐  ┌──────────┐  ┌──────────┐
│ FP32 ×32 │  │ INT32 ×16│  │ FP64 ×16 │ ← 三组全部独立, 可同时执行
│ (专用)   │  │ (专用)   │  │ (专用)   │
└──────────┘  └──────────┘  └──────────┘

每 SM 每周期: 128 FP32 + 64 INT32 + 64 FP64 并发
```

### 5.5 各架构 FP32/INT32 并发能力总结

| 架构 | CC | FP32/SM | INT32/SM | 并发 | 备注 |
|------|-----|---------|---------|------|------|
| Pascal | 6.0/6.1 | 64/128 | — | FP32 或 INT32 | 共享 ALU |
| Volta | 7.0 | 64 | 64 | FP32 + INT32 | 物理分离 |
| Turing | 7.5 | 64 | 64 | FP32 + INT32 | 同 Volta |
| Ampere A100 | 8.0 | 64 | 64 | FP32 + INT32 | 同 Volta |
| Ampere GA10x | 8.6 | 128 | 64 | 128 FP32 或 64 FP32+64 INT32 | 双用通路 |
| Ada | 8.9 | 128 | 64 | 同 CC 8.6 | 双用通路 |
| Hopper | 9.0 | 128 | 64 | FP32 + INT32 + FP64 全并发 | 三路专用 |

---

## 6. 指令吞吐量

### 6.1 每 SM 每时钟周期吞吐 (Results/Clock/SM)

| 指令类型 | CC 7.0 | CC 7.5 | CC 8.0 | CC 8.6/8.9 | CC 9.0 |
|----------|--------|--------|--------|-----------|--------|
| **FP32 add/mul/fma** | 64 | 64 | 64 | 128 | 128 |
| **FP64 add/mul/fma** | 32 | 2 | 32 | 2 | 64 |
| **INT32 add** | 64 | 64 | 64 | 64 | 64 |
| **FP16×2 (half2)** | 128 | 128 | 128 | 256 | 256 |
| **BF16×2** | — | — | 128 | 256 | 256 |
| **SFU** (sin/cos/rsqrt) | 16 | 16 | 16 | 16 | 32 |
| **类型转换** | 16 | 16 | 32 | 32 | 64 |
| **LD/ST 单元** | 32 | 16 | 32 | 32 | 32 |

### 6.2 关键指令延迟 (Latency in Clock Cycles)

| 指令类型 | CC 7.0+ | CC 6.x (Pascal) |
|----------|---------|----------------|
| FP32 算术 (add/mul/fma) | **4** cycles | 6 cycles |
| FP64 算术 | **4** cycles | — |
| INT32 算术 | **4** cycles | 6 cycles |
| SFU (sin/cos 等) | ~8 cycles | — |
| Shared Memory 读取 | ~23 cycles | — |
| L1 Cache 命中 | ~28-33 cycles | — |
| L2 Cache 命中 | ~200 cycles | — |
| Global Memory (HBM) | ~400-800 cycles | — |

### 6.3 FP64 吞吐的数据中心 vs 消费级差异

| GPU 类别 | FP64 : FP32 吞吐比 | 原因 |
|---------|-------------------|------|
| V100 / A100 / H100 | **1:2** | 数据中心 GPU 配备完整 FP64 单元 |
| RTX 3090 / RTX 4090 / RTX 5090 | **1:64** | 消费级仅保留极少 FP64 单元 (兼容性) |

### 6.4 FP16 / BF16 的 Packed 执行

CUDA Core 可使用 `half2` / `__nv_bfloat162` 类型将两个 16-bit 值打包到一个 32-bit 寄存器中，一条指令处理两个元素：

```cpp
// 标量 FP16 — 每条指令 1 次运算 (浪费 FP32 单元一半带宽)
half a, b, c;
c = __hadd(a, b);

// Packed FP16 — 每条指令 2 次运算 (吞吐翻倍)
half2 a2, b2, c2;
c2 = __hadd2(a2, b2);  // 同时计算 low 和 high 两个 FP16
```

> LLM Kernel 中非矩阵乘操作 (如 element-wise、reduction) 应优先使用 `half2` / `__nv_bfloat162` 向量化。

---

## 7. 内存层级与 CUDA Core 交互

### 7.1 内存层级一览

```
                      延迟        带宽          容量
                     ┌────────────────────────────────────┐
Register File:       │ 1 cycle    ~25 TB/s (估算)   256 KB/SM │  ← CUDA Core 直接操作数
                     ├────────────────────────────────────┤
Shared Memory:       │ ~23 cycles  ~128 B/clk/SM    ≤228 KB/SM│  ← 线程块内协作
                     ├────────────────────────────────────┤
L1 Cache:            │ ~28-33 cyc  ~128 B/clk/SM    L1+SMEM 共│  ← 自动缓存
                     ├────────────────────────────────────┤
L2 Cache:            │ ~200 cycles ~6-12 TB/s       6-192 MB  │  ← 全芯片共享
                     ├────────────────────────────────────┤
Global Memory (HBM): │ 400-800 cyc 0.9-8 TB/s      32-192 GB │  ← 主存
                     └────────────────────────────────────┘
```

### 7.2 CUDA Core 访存流程

```
1. CUDA Core 执行 Load 指令
2. LD/ST 单元生成地址, 请求送往 L1 Cache
3. L1 Miss → 请求传递到 L2 Cache
4. L2 Miss → 请求传递到 HBM
5. 数据到达前, 该 Warp 标记为 Stalled
6. Warp Scheduler 立即切换到其他 Eligible Warp (零开销)
7. 数据到达 → 写入寄存器, 清除 Scoreboard 位
8. 该 Warp 变为 Eligible, 等待被再次调度
```

> NVIDIA 使用 **Stall-on-use** 策略：Load 指令本身不阻塞 Warp，仅当后续指令读取该寄存器时才检查数据是否就绪。

### 7.3 各 GPU L2 Cache 容量

| GPU | L2 Cache |
|-----|----------|
| V100 | 6 MB |
| A100 | 40 MB |
| H100 | 50 MB |
| B200 | 192 MB |
| RTX 3090 | 6 MB |
| RTX 4090 | 72 MB |

### 7.4 Memory Coalescing (合并访存)

Warp 内 32 个线程的 Global Memory 访问由 LD/ST 单元合并为最少的内存事务：

```
理想合并 (连续访问):
  T0→addr+0, T1→addr+4, T2→addr+8, ... T31→addr+124
  → 1 个 128B 事务 (完美合并)

非合并 (跳步访问):
  T0→addr+0, T1→addr+128, T2→addr+256, ...
  → 32 个独立事务 → 性能下降 32×
```

---

## 8. 架构演进总览

### 8.1 CUDA Core 架构代际对比

| 架构 | 年份 | CC | FP32/SM | INT32/SM | FP64/SM | TC/SM | 关键创新 |
|------|------|-----|---------|---------|---------|-------|---------|
| Tesla | 2008 | 1.x | 8 | — | 0 | — | 首代 CUDA |
| Fermi | 2010 | 2.x | 32 | — | 16 | — | 首次 FP64, 2 Warp Scheduler |
| Kepler | 2012 | 3.x | 192 | — | 64 | — | 4 Warp Scheduler × 2 Dispatch |
| Maxwell | 2014 | 5.x | 128 | — | 0 | — | 4 Sub-partition 设计, 2× 能效 |
| Pascal | 2016 | 6.x | 64/128 | — | 32/4 | — | NVLink, HBM2 |
| **Volta** | 2017 | 7.0 | 64 | 64 | 32 | 8 | **FP32/INT32 分离**, 独立线程调度, 首代 Tensor Core |
| Turing | 2018 | 7.5 | 64 | 64 | 2 | 8 | RT Core, INT8/INT4 Tensor Core |
| **Ampere** | 2020 | 8.0 | 64 | 64 | 32 | 4 | cp.async, 异步内存拷贝 |
| Ampere | 2020 | 8.6 | 128 | 64* | 2 | 4 | **FP32 吞吐翻倍** (双用通路) |
| Ada | 2022 | 8.9 | 128 | 64* | 2 | 4 | 3rd gen RT Core |
| **Hopper** | 2022 | 9.0 | 128 | 64 | 64 | 4 | TMA, Thread Block Cluster, WGMMA |
| Blackwell | 2024 | 10.0 | 128 | 128 | ~64 | 4 | 双芯封装, TMEM, tcgen05 |
| Blackwell | 2025 | 12.0 | 128 | 128 | — | 4 | 消费级, mma.sync 扩展 |

### 8.2 数据中心 GPU 规格

| 规格 | V100 SXM2 | A100 SXM | H100 SXM5 | B200 |
|------|-----------|----------|-----------|------|
| SM 数 | 80 | 108 | 132 | 148 |
| CUDA Core/SM | 64 | 64 | 128 | 128 |
| **总 CUDA Cores** | **5,120** | **6,912** | **16,896** | **18,944** |
| FP64 Cores | 2,560 | 3,456 | 8,448 | ~9,472 |
| Register File/SM | 256 KB | 256 KB | 256 KB | 256 KB |
| L2 Cache | 6 MB | 40 MB | 50 MB | 192 MB |
| HBM 容量 | 32 GB | 80 GB | 80 GB | 192 GB |
| HBM 带宽 | 900 GB/s | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s |
| FP32 峰值 TFLOPS | ~15.7 | ~19.5 | ~66.9 | ~90 |
| FP64 峰值 TFLOPS | ~7.8 | ~9.7 | ~33.5 | ~45 |
| TDP | 300 W | 400 W | 700 W | 1,000 W |

### 8.3 消费级 GPU 规格

| 规格 | RTX 3090 | RTX 4090 | RTX 5090 |
|------|----------|----------|----------|
| 架构 | Ampere GA102 | Ada AD102 | Blackwell GB202 |
| CC | 8.6 | 8.9 | 12.0 |
| SM 数 | 82 | 128 | 170 |
| CUDA Core/SM | 128 | 128 | 128 |
| **总 CUDA Cores** | **10,496** | **16,384** | **21,760** |
| FP32 峰值 TFLOPS | ~35.6 | ~82.6 | ~104.8 |
| 显存 | 24 GB GDDR6X | 24 GB GDDR6X | 32 GB GDDR7 |
| 显存带宽 | 936 GB/s | 1,008 GB/s | 1,792 GB/s |
| TDP | 350 W | 450 W | 575 W |

---

## 9. CUDA Core vs Tensor Core

### 9.1 本质差异

| 维度 | CUDA Core | Tensor Core |
|------|-----------|-------------|
| 类型 | 通用标量处理器 | 固定功能矩阵加速器 |
| 每 Core 操作 | 1 次 FP32/INT32 | 4×4×4 FMA (64 次乘加) |
| 参与线程 | 单线程执行, Warp 级调度 | Warp 级 (32 线程) 或 Warpgroup 级 (128 线程) 协作 |
| 支持精度 | FP32, FP64, INT32 | FP16, BF16, FP8, FP4, INT8 等 |
| 灵活性 | 任意标量运算、分支、控制流 | 仅矩阵乘累加 (D = A×B + C) |
| 编程方式 | 标准 CUDA C++ (`a + b * c`) | WMMA / mma.sync / wgmma PTX |

### 9.2 吞吐对比 (以 H100 为例)

```
CUDA Core FP32:    128 ops/cycle/SM × 132 SM × 1.83 GHz ≈  31 TFLOPS (FMA: 62 TFLOPS)
Tensor Core FP16:  ~990 TFLOPS
Tensor Core FP8:   ~1,979 TFLOPS

→ Tensor Core FP16 吞吐是 CUDA Core FP32 的 ~16×
→ Tensor Core FP8  吞吐是 CUDA Core FP32 的 ~32×
```

### 9.3 互补关系

```
LLM 推理 / 训练中的典型 Kernel:

  ┌─────────────────────────────────────────────────┐
  │ Linear Layer (GEMM)        → Tensor Core  (>90%) │
  │ Attention QK^T, PV         → Tensor Core         │
  ├─────────────────────────────────────────────────┤
  │ RMSNorm / LayerNorm        → CUDA Core           │
  │ SiLU / GeLU 激活函数       → CUDA Core (SFU)     │
  │ Softmax                    → CUDA Core           │
  │ Residual Add               → CUDA Core           │
  │ Embedding Lookup           → CUDA Core + LD/ST   │
  │ Top-k / Sampling           → CUDA Core           │
  │ RoPE (旋转位置编码)        → CUDA Core           │
  │ Quantize / Dequantize      → CUDA Core           │
  └─────────────────────────────────────────────────┘
```

虽然 GEMM 占 >90% FLOPs，但非 GEMM 操作常成为**内存带宽瓶颈**，优化 CUDA Core Kernel (向量化、合并访存、减少分支) 对端到端性能至关重要。

---

## 10. LLM Kernel 中的 CUDA Core 使用场景

### 10.1 Element-wise 操作

激活函数、残差加法、缩放等逐元素操作完全由 CUDA Core 执行：

```cpp
// 典型 fused kernel: residual add + RMSNorm + SiLU
__global__ void fused_residual_rmsnorm_silu(
    half2* output, const half2* input, const half2* residual,
    const half2* weight, float eps, int hidden_dim)
{
    // 1. Residual Add (CUDA Core: FP16 add)
    half2 x = __hadd2(input[idx], residual[idx]);

    // 2. RMSNorm - reduction (CUDA Core: FP32 fma + warp shuffle)
    float sum_sq = /* warp-level reduction of x^2 */;
    float rms = rsqrtf(sum_sq / hidden_dim + eps);  // SFU: rsqrt

    // 3. Scale + SiLU (CUDA Core: FP16 mul + SFU for sigmoid)
    half2 normed = __hmul2(x, __float2half2_rn(rms));
    half2 gated = __hmul2(normed, /* silu(gate) */);

    output[idx] = gated;
}
```

**优化要点：**
- 使用 `half2` 向量化，吞吐翻倍
- 合并多个 element-wise 操作为一个 fused kernel，减少 Global Memory 读写
- 利用 Warp Shuffle (`__shfl_xor_sync`) 进行 reduction，避免 Shared Memory

### 10.2 Reduction (归约)

Softmax、LayerNorm 等需要对整行/整块求和：

```
Warp 级 Reduction (32 线程):
  T0  T1  T2  ... T31
   \  /    \  /
   add     add        ← __shfl_xor_sync, offset=1
    \      /
     add              ← __shfl_xor_sync, offset=2
      \  /
      add              ← __shfl_xor_sync, offset=4
       ...
      result           ← 5 步完成 32 元素归约 (log2(32)=5)
```

### 10.3 Epilogue 操作

Tensor Core 完成 GEMM 后的后处理（偏置加法、激活函数、量化）由 CUDA Core 在同一 Kernel 内执行：

```
GEMM Kernel 流程:
  1. 从 Global Memory 加载 A, B tile   → LD/ST + CUDA Core (地址计算)
  2. 矩阵乘累加                        → Tensor Core (mma.sync / wgmma)
  3. Epilogue:
     a. 偏置加法                       → CUDA Core (FP32 add)
     b. 激活函数 (GeLU/SiLU)           → CUDA Core (SFU + FP32 mul)
     c. 量化到 FP8                     → CUDA Core (类型转换)
     d. 写回 Global Memory             → LD/ST
```

### 10.4 性能优化检查清单

- [ ] **向量化**：使用 `half2` / `float4` 提高每条指令处理的数据量
- [ ] **合并访存**：确保 Warp 内线程访问连续地址 (128B 对齐)
- [ ] **减少 Warp Divergence**：分支条件按 Warp 边界对齐
- [ ] **Kernel 融合**：多个 element-wise 操作合并为一个 Kernel
- [ ] **Occupancy 平衡**：寄存器使用和并发 Warp 数之间权衡
- [ ] **Warp Shuffle 替代 Shared Memory**：小规模 reduction 用 Shuffle
- [ ] **利用 INT32 并发**：Volta+ 的整数运算与浮点运算可重叠

---

## 参考资源

- [Modal GPU Glossary: CUDA Core](https://modal.com/gpu-glossary/device-hardware/cuda-core)
- [Modal GPU Glossary: Streaming Multiprocessor](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor)
- [Modal GPU Glossary: Warp Scheduler](https://modal.com/gpu-glossary/device-hardware/warp-scheduler)
- [A Comprehensive Guide to CUDA Cores in NVIDIA GPU (Medium)](https://naddod.medium.com/a-comprehensive-guide-to-cuda-cores-in-nvidia-gpu-a17734e70979)
- [What is a CUDA Core and How Do They Work (Corsair)](https://www.corsair.com/us/en/explorer/gamer/gaming-pcs/what-is-a-cuda-core-and-how-do-they-work/)
- [NVIDIA Volta Architecture Whitepaper (PDF)](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
- [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
- [CUDA Programming Guide — Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [A History of NVIDIA Stream Multiprocessor (Fabien Sanglard)](https://fabiensanglard.net/cuda/)
- [SIMT and Warps — Cornell Virtual Workshop](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/simt_warp)

---

*本文档作为 LLM Kernel Agent 的 CUDA Core 硬件架构参考。配合 `tensor-core.md`（Tensor Core 硬件）和 `mma-wmma.md`（Tensor Core 编程接口）共同使用。*
