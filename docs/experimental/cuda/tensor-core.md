# Tensor Core 硬件架构与演进

> 面向 LLM 高性能 Kernel 开发的 Tensor Core 硬件深度解析
> 覆盖 Volta (1st Gen) → Turing (2nd) → Ampere (3rd) → Hopper (4th) → Blackwell (5th)

---

## 目录

1. [Tensor Core 概述](#1-tensor-core-概述)
2. [基本工作原理](#2-基本工作原理)
3. [第一代：Volta (SM 7.0)](#3-第一代volta-sm-70)
4. [第二代：Turing (SM 7.5)](#4-第二代turing-sm-75)
5. [第三代：Ampere (SM 8.0/8.6)](#5-第三代ampere-sm-8086)
6. [第四代：Hopper (SM 9.0)](#6-第四代hopper-sm-90)
7. [第五代：Blackwell (SM 10.0)](#7-第五代blackwell-sm-100)
8. [数据类型全景](#8-数据类型全景)
9. [Transformer Engine](#9-transformer-engine)
10. [性能规格对比](#10-性能规格对比)
11. [LLM Kernel 开发中的架构选型](#11-llm-kernel-开发中的架构选型)

---

## 1. Tensor Core 概述

### 1.1 什么是 Tensor Core

Tensor Core 是 NVIDIA GPU 中专为**矩阵乘累加 (Matrix Multiply-Accumulate, MMA)** 设计的硬件计算单元。与传统 CUDA Core 逐标量/逐向量计算不同，Tensor Core 在单条指令中完成小矩阵的乘法和累加：

```
D = A × B + C
```

其中 A、B、C、D 为小矩阵 (如 4×4、8×8、16×8 等)，操作数可以是不同精度。

### 1.2 为何 LLM 需要 Tensor Core

| 操作 | 计算特征 | Tensor Core 优势 |
|------|---------|----------------|
| Linear / GEMM | 密集矩阵乘 | 直接加速，10–20× vs CUDA Core |
| Attention (QK^T, PV) | 批量矩阵乘 | WGMMA/TMA 异步流水线 |
| 量化推理 (INT8/FP8/FP4) | 低精度 GEMM | 低精度吞吐翻倍 |
| MoE Expert 计算 | 小批量 GEMM | 多 Tensor Core 并行 |

LLM 中 >90% 的计算量集中在矩阵乘法，Tensor Core 是绝对性能瓶颈。

### 1.3 在 SM 中的位置

```
SM (Streaming Multiprocessor)
├── Warp Scheduler × 4
├── Register File (256 KB)
├── CUDA Cores (FP32/INT32)
├── Tensor Cores × 4          ← 每个 SM 4 个 (Ampere+)
├── SFU (Special Function Units)
├── Load/Store Units
├── Shared Memory / L1 Cache
└── [Tensor Memory — Blackwell only]
```

> **注意：** 程序员不需要关心"哪个 Tensor Core 执行哪部分计算"——硬件自动将 MMA 指令分配到可用的 Tensor Core。

---

## 2. 基本工作原理

### 2.1 最小计算单元：4×4×4 MMA

每个 Tensor Core 的物理计算阵列执行 4×4×4 矩阵乘累加：

```
A [4×4] × B [4×4] + C [4×4] = D [4×4]

= 64 次 FMA (Fused Multiply-Add) / cycle
= 128 FLOPS / cycle (每次 FMA 算 2 FLOPS)
```

### 2.2 从 4×4 到 Warp 级操作

硬件将多个 4×4 操作组合，暴露给软件的是**更大的 Warp 级矩阵操作**：

```
Volta/Turing:  8 线程协作 → m8n8k4 (基础)
Ampere:       32 线程 (1 Warp) → m16n8k16 (主力)
Hopper:      128 线程 (4 Warps = Warpgroup) → m64n256k16 (最大)
Blackwell:   单线程发射 → 硬件跨 2 SM 执行 → m256n256k16
```

### 2.3 操作数来源演进

| 代次 | A 操作数 | B 操作数 | C/D 累加器 | 备注 |
|------|---------|---------|-----------|------|
| Volta–Ampere | 寄存器 | 寄存器 | 寄存器 | 全部经由寄存器 |
| Hopper (WGMMA) | Shared Memory 或 寄存器 | Shared Memory | 寄存器 | A 可不过寄存器 |
| Blackwell (tcgen05) | Shared Memory | Shared Memory | Tensor Memory (TMEM) | 全部脱离通用寄存器 |

**趋势：** Tensor Core 吞吐每代翻倍，但全局内存延迟没有降低。为了喂饱 Tensor Core，操作数来源从寄存器逐步迁移到 Shared Memory / Tensor Memory，减少寄存器带宽瓶颈。

---

## 3. 第一代：Volta (SM 7.0)

### 3.1 代表硬件

| GPU | SM 数 | Tensor Core/SM | 总 Tensor Core | FP16 TFLOPS |
|-----|-------|---------------|---------------|-------------|
| Tesla V100 SXM2 | 80 | 8 | 640 | 125 |
| Titan V | 80 | 8 | 640 | 110 |

### 3.2 技术特征

- **首次引入 Tensor Core** (2017)
- 支持数据类型：仅 **FP16** 输入，FP16/FP32 累加
- 每 SM 有 8 个 Tensor Core，分为 4 组（每组 2 个，与 sub-partition 对应）
- 暴露的最小 MMA 单元：**m8n8k4** (8 线程协作)
- 每个 Tensor Core 每周期 64 FMA → 每 SM 每周期 1024 FMA (FP16)

### 3.3 编程接口

- **WMMA API** (CUDA 9.0)：首次提供 C++ 级 Tensor Core 编程接口
- **mma.sync PTX** (CUDA 10.1)：PTX 级直接控制，更高性能
- 主力形状：`m16n16k16` (WMMA), `m8n8k4` (PTX mma.sync)

### 3.4 限制

- 仅 FP16 → 训练精度受限，需要混合精度 (FP16 计算 + FP32 主权重)
- WMMA API 的 fragment 布局不透明，性能调优困难
- 8 个 Tensor Core/SM 但后续架构精简为 4 个/SM（单个 TC 吞吐翻倍）

---

## 4. 第二代：Turing (SM 7.5)

### 4.1 代表硬件

| GPU | SM 数 | Tensor Core/SM | 总 Tensor Core | FP16 TFLOPS |
|-----|-------|---------------|---------------|-------------|
| RTX 2080 Ti | 68 | 8 | 544 | 107 |
| Tesla T4 | 40 | 8 | 320 | 65 |

### 4.2 技术特征

- 新增 **INT8、INT4、Binary (1-bit)** 数据类型
- 首次将 Tensor Core 引入消费级 GPU (RTX 20 系列)
- INT8 支持使推理量化成为可能

### 4.3 新增 MMA 形状

| 数据类型 | 新增形状 | 用途 |
|----------|---------|------|
| FP16 | `m16n8k8` | 更高效的 Warp 级 MMA |
| INT8 | `m8n8k16`, `m16n8k16` | INT8 量化推理 |
| INT4 | `m8n8k32`, `m16n8k32` | 极低精度推理 |
| Binary | `m8n8k128` | 二值网络 |

---

## 5. 第三代：Ampere (SM 8.0/8.6)

### 5.1 代表硬件

| GPU | SM 数 | Tensor Core/SM | 总 Tensor Core | 算力版本 |
|-----|-------|---------------|---------------|---------|
| A100 SXM | 108 | 4 | 432 | SM 8.0 |
| RTX 3090 | 82 | 4 | 328 | SM 8.6 |
| A30 | 56 | 4 | 224 | SM 8.0 |

### 5.2 关键创新

#### (1) 新数据类型

| 类型 | 格式 | 用途 |
|------|------|------|
| **TF32** (TensorFloat-32) | 1 sign + 8 exp + 10 mantissa = 19 bits | FP32 训练的透明加速 (自动截断 FP32 低位) |
| **BF16** (BFloat16) | 1 sign + 8 exp + 7 mantissa = 16 bits | 训练 (比 FP16 更大动态范围) |

#### (2) 每 SM Tensor Core 数量从 8 → 4，单个吞吐翻倍

```
Volta:  8 TC/SM × 64 FMA/TC/cycle = 512 FMA/SM/cycle (FP16)
Ampere: 4 TC/SM × 256 FMA/TC/cycle = 1024 FMA/SM/cycle (FP16)
```

每代总吞吐翻倍，但用更少的 TC 实现 (每个 TC 更宽)。

#### (3) 完整 Warp 级 MMA (Full-Warp MMA)

Ampere 的 `mma.sync` 指令由完整的 32 线程 Warp 协作执行，主力形状为 **m16n8k16**，取代 Volta 的 8 线程 m8n8k4。

#### (4) 2:4 结构化稀疏 (Fine-Grained Structured Sparsity)

A100 引入硬件加速的 2:4 稀疏模式——每 4 个连续元素中恰好 2 个为零。Sparse Tensor Core 仅对非零值计算，理论吞吐翻倍。

### 5.3 A100 Tensor Core 性能

| 精度 | Dense TFLOPS | Sparse TFLOPS |
|------|-------------|--------------|
| FP64 | 19.5 | — |
| TF32 | 156 | 312 |
| BF16 / FP16 | 312 | 624 |
| INT8 | 624 TOPS | 1248 TOPS |
| INT4 | 1248 TOPS | 2496 TOPS |

---

## 6. 第四代：Hopper (SM 9.0)

### 6.1 代表硬件

| GPU | SM 数 | Tensor Core/SM | 总 Tensor Core | 算力版本 |
|-----|-------|---------------|---------------|---------|
| H100 SXM5 | 132 | 4 | 528 | SM 9.0 |
| H200 SXM | 132 | 4 | 528 | SM 9.0 |

### 6.2 关键创新

#### (1) FP8 数据类型

| 格式 | 结构 | 用途 |
|------|------|------|
| **E4M3** | 1+4+3 = 8 bit | 推理 (精度优先，范围 ±448) |
| **E5M2** | 1+5+2 = 8 bit | 训练 (范围优先，范围 ±57344) |

FP8 使 Tensor Core 吞吐相比 FP16 翻倍 (同硅面积处理 2× 数据)。

#### (2) Warpgroup MMA (WGMMA)

- **128 线程 (4 Warps)** 协作执行单个 MMA
- **异步执行**：`wgmma.mma_async` 不阻塞发射线程，可与数据加载重叠
- **操作数直接从 Shared Memory 读取**，无需先加载到寄存器
- 最大形状：**m64n256k16**

```
Ampere mma.sync:  32 线程, 同步, A/B 从寄存器
Hopper wgmma:    128 线程, 异步, A/B 可从 Shared Memory
```

#### (3) Transformer Engine

专用硬件单元，在 Tensor Core 输入/输出路径上实现动态精度缩放：

```
FP32/BF16 权重 ──→ [Transformer Engine: 动态量化] ──→ FP8 输入
                                                       ↓
                                                  Tensor Core MMA (FP8)
                                                       ↓
FP32/BF16 输出 ←── [Transformer Engine: 反量化] ←── FP8/FP32 累加
```

- 自动计算每张量的缩放因子 (scaling factor)
- 在 FP8 精度下保持接近 FP16 的训练精度
- cuBLAS / cuDNN 自动利用

#### (4) TMA (Tensor Memory Accelerator)

虽然 TMA 不属于 Tensor Core 本身，但它是 Hopper 架构中**喂饱 Tensor Core** 的关键组件：

- 硬件多维张量拷贝引擎
- Global → Shared Memory 绕过寄存器
- 自动边界处理、Swizzle
- 仅需 1 个线程发起 (其余线程做计算)

### 6.3 H100 Tensor Core 性能

| 精度 | Dense TFLOPS | Sparse TFLOPS | vs A100 Dense |
|------|-------------|--------------|--------------|
| FP64 | 67 | — | 3.4× |
| TF32 | ~495 | ~990 | 3.2× |
| BF16 / FP16 | ~990 | ~1979 | 3.2× |
| FP8 | ~1979 | ~3958 | ∞ (A100 无 FP8) |
| INT8 | ~1979 TOPS | ~3958 TOPS | 3.2× |

---

## 7. 第五代：Blackwell (SM 10.0)

### 7.1 代表硬件

| GPU | SM 数 | Tensor Core/SM | 总 Tensor Core | 算力版本 |
|-----|-------|---------------|---------------|---------|
| B200 | 148 (2×74) | 4 | ~592 | SM 10.0 |
| B100 | 140 | 4 | 560 | SM 10.0 |
| GB200 (双 GPU) | 2×148 | 4 | ~1184 | SM 10.0 |
| **RTX 5090** | 170 | 4 | 680 | **SM 12.0** |

> **重要：SM 10.0 vs SM 12.0 是完全不同的 Tensor Core 编程模型。** RTX 5090 (消费级 Blackwell, SM 12.0) 的 Tensor Core 使用 **`mma.sync` 扩展版** (与 Ampere 同一编程范式，但增加了 FP8/FP4/FP6 新数据类型支持)。SM 12.0 芯片 (GB202) 上**不存在 TMEM 硬件**，这是硅片级的架构差异，不是软件限制。
>
> | 特性 | SM 10.0 (B200/B100) | SM 12.0 (RTX 5090) |
> |------|:-------------------:|:-------------------:|
> | tcgen05.mma | ✅ | ❌ |
> | TMEM (256 KB/SM) | ✅ | ❌ (硬件不存在) |
> | CTA Pair (2-SM 协作) | ✅ | ❌ |
> | WGMMA | ✅ | ❌ |
> | TMA | ✅ | ✅ |
> | **mma.sync (扩展版)** | ❌ | ✅ |
> | FP4 / FP6 数据类型 | ✅ | ✅ |
> | FP8 数据类型 | ✅ | ✅ |
> | Tensor Core 编程范式 | tcgen05 (单线程发射) | mma.sync (Warp 级同步) |
> | SASS 指令 | UTC*/UTMA* | HMMA.16816 |
>
> **对 Kernel 开发者的影响：** 需要维护三套独立代码路径 — Hopper (wgmma)、数据中心 Blackwell (tcgen05)、消费级 Blackwell (mma.sync 扩展)。SM 10.0 上一条 tcgen05 处理的 64×64 tile，在 SM 12.0 上需要拆成多个 m16n8k16 的 mma.sync 调用。

### 7.2 关键创新

#### (1) FP4 / FP6 数据类型

| 格式 | 结构 | 精度 | 用途 |
|------|------|------|------|
| **NVFP4** (E2M1) | 1+2+1 = 4 bit | 低 | 推理 (2 级缩放补偿精度) |
| **FP6** (E3M2) | 1+3+2 = 6 bit | 中低 | 推理 (精度/吞吐平衡) |

NVFP4 采用**两级缩放**：FP8 (E4M3) 微块缩放 (16 值一组) + FP32 张量级缩放。

#### (2) tcgen05 指令 (第五代 MMA)

**彻底脱离 Warp 同步范式：**

```
Volta–Ampere:  mma.sync    → 32 线程同步执行
Hopper:        wgmma       → 128 线程 (Warpgroup) 异步执行
Blackwell:     tcgen05.mma → 单线程发射, 硬件调度
```

关键变化：
- **操作数全部脱离通用寄存器**：A, B 来自 Shared Memory，C/D 在 Tensor Memory (TMEM)
- **单线程发射**：`tcgen05.mma` 由单个线程发起，消除 Warp 同步开销
- **跨 2 SM 执行**：CTA Pair 模式下两个 SM 的 Tensor Core 协作

#### (3) Tensor Memory (TMEM)

```
每 SM 集成 256 KB 专用 TMEM
├── 512 列 × 128 行 × 32 bit
├── 通过 tcgen05.alloc / tcgen05.dealloc 显式管理
├── 累加器 D 始终驻留在 TMEM (不占用通用寄存器)
├── tcgen05.ld: TMEM → 寄存器 (后处理)
├── tcgen05.st: 寄存器 → TMEM
└── tcgen05.cp: Shared Memory → TMEM
```

TMEM 解决了累加器占用大量寄存器的问题（如 GEMM 中 m64n128 累加器需 64×128/128 = 64 个 FP32 寄存器/线程）。

#### (4) CTA Pair 执行

两个相邻 CTA (在同一 Cluster 中) 共享操作数，跨 2 个 SM 的 Tensor Core 协作执行单个 MMA：

```
SM 0 (CTA 0) ──→ 提供 A 的上半部分 + TMEM 0
                      ↘
                       Tensor Core 协作 MMA: m256×n256×k16
                      ↗
SM 1 (CTA 1) ──→ 提供 A 的下半部分 + TMEM 1
                  B 从 Shared Memory 共享
```

**效果：** 等效单 SM 的 Shared Memory 容量翻倍（每个 SM 只需加载一半操作数）。

#### (5) 4:8 结构化稀疏 (NVFP4)

Blackwell 为 NVFP4 引入 pairwise 4:8 稀疏：每 8 个元素分为 4 对，其中 2 对非零、2 对被剪枝。

### 7.3 B200 Tensor Core 性能

| 精度 | Dense TFLOPS | Sparse TFLOPS | vs H100 Dense |
|------|-------------|--------------|--------------|
| TF32 | ~1,125 | ~2,250 | ~2.3× |
| BF16 / FP16 | ~2,250 | ~4,500 | ~2.3× |
| FP8 | ~4,500 | ~9,000 | ~2.3× |
| FP6 | ~5,135 | — | ∞ |
| FP4 | ~9,000 | ~18,000 | ∞ |
| INT8 | ~4,500 TOPS | ~9,000 TOPS | ~2.3× |

---

## 8. 数据类型全景

### 8.1 Tensor Core 支持的数据类型总表

| 数据类型 | 位宽 | 格式 (S+E+M) | 引入代次 | 主要用途 |
|----------|------|-------------|---------|---------|
| FP64 | 64 | 1+11+52 | Ampere | 科学计算 (非 LLM 场景) |
| FP32 | 32 | 1+8+23 | — | 累加器/主权重 (不直接输入 TC) |
| TF32 | 19* | 1+8+10 | Ampere | FP32 训练的透明加速 |
| BF16 | 16 | 1+8+7 | Ampere | 训练 (大动态范围) |
| FP16 | 16 | 1+5+10 | Volta | 训练/推理 (高精度) |
| FP8 E4M3 | 8 | 1+4+3 | Hopper | 推理/前向 |
| FP8 E5M2 | 8 | 1+5+2 | Hopper | 训练/反向梯度 |
| FP6 E3M2 | 6 | 1+3+2 | Blackwell | 推理 (MX 格式) |
| FP4 E2M1 | 4 | 1+2+1 | Blackwell | 推理 (NVFP4) |
| INT8 | 8 | 有/无符号 | Turing | INT8 量化推理 |
| INT4 | 4 | 有/无符号 | Turing | (Hopper 起废弃) |
| Binary | 1 | 0/1 | Turing | 二值网络 (极少使用) |

*TF32 实际占 32 位存储，Tensor Core 仅使用高 19 位计算。

### 8.2 各架构支持矩阵

| 数据类型 | Volta 7.0 | Turing 7.5 | Ampere 8.0 | Hopper 9.0 | Blackwell 10.0 |
|----------|:---------:|:----------:|:----------:|:----------:|:--------------:|
| FP16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| BF16 | ❌ | ❌ | ✅ | ✅ | ✅ |
| TF32 | ❌ | ❌ | ✅ | ✅ | ✅ |
| FP64 | ❌ | ❌ | ✅ | ✅ | ✅* |
| FP8 | ❌ | ❌ | ❌ | ✅ | ✅ |
| FP6 | ❌ | ❌ | ❌ | ❌ | ✅ |
| FP4 | ❌ | ❌ | ❌ | ❌ | ✅ |
| INT8 | ❌ | ✅ | ✅ | ✅ | ✅ |
| INT4 | ❌ | ✅ | ✅ | ❌ | ❌ |
| Binary | ❌ | ✅ | ✅ | ❌ | ❌ |

*Blackwell 的 FP64 Tensor Core 吞吐较低，非主要使用场景。

### 8.3 LLM 常用精度策略

| 场景 | 推荐精度 | A/B 输入 | 累加器 | 备注 |
|------|---------|---------|--------|------|
| 训练 (Ampere) | 混合精度 | BF16 | FP32 | 主权重 FP32，TF32 自动生效 |
| 训练 (Hopper) | FP8 混合 | FP8 E4M3/E5M2 | FP32 | Transformer Engine 动态缩放 |
| 推理 (Ampere) | INT8 / FP16 | INT8 或 FP16 | INT32 / FP32 | W8A8 / W4A16 |
| 推理 (Hopper) | FP8 | E4M3 | FP32 | 吞吐最高 |
| 推理 (Blackwell) | FP4 | NVFP4 | FP32 | 两级缩放 |

---

## 9. Transformer Engine

### 9.1 概述

Transformer Engine 是 Hopper 引入的专用硬件单元，位于 Tensor Core 的数据路径上，实现 FP8 的动态量化/反量化：

### 9.2 工作原理

```
每层的输入/输出 → 统计最大绝对值 (amax)
                    ↓
              计算缩放因子: scale = max_representable / amax
                    ↓
              量化: FP8_val = round(FP32_val × scale)
                    ↓
              Tensor Core MMA (FP8)
                    ↓
              反量化: FP32_result = FP8_result / scale
```

### 9.3 Delayed Scaling vs Per-Tensor Scaling

| 策略 | 延迟 | 精度 | 实现 |
|------|------|------|------|
| Delayed Scaling | 使用上一次迭代的 amax | 略低 | 默认，无额外开销 |
| Per-Tensor Scaling | 当前迭代实时计算 | 高 | 需要额外 reduction |
| Block Scaling (Blackwell) | 每 16 元素一个 FP8 scale | 最高 | NVFP4 硬件支持 |

### 9.4 软件接口

```python
# PyTorch + Transformer Engine
import transformer_engine.pytorch as te

model = te.Linear(4096, 4096, bias=True)
# 自动使用 FP8 + Transformer Engine
with te.fp8_autocast():
    output = model(input)
```

---

## 10. 性能规格对比

### 10.1 数据中心 GPU 全面对比

| 规格 | V100 SXM2 | A100 SXM | H100 SXM5 | B200 |
|------|-----------|----------|-----------|------|
| **架构** | Volta | Ampere | Hopper | Blackwell |
| **工艺** | 12nm | 7nm | 4nm | 4nm 双芯 |
| **SM 数** | 80 | 108 | 132 | 148 |
| **TC/SM** | 8 | 4 | 4 | 4 |
| **TC 代次** | 1st | 3rd | 4th | 5th |
| **FP16 TC TFLOPS** | 125 | 312 | ~990 | ~2,250 |
| **TF32 TC TFLOPS** | — | 156 | ~495 | ~1,125 |
| **FP8 TC TFLOPS** | — | — | ~1,979 | ~4,500 |
| **FP4 TC TFLOPS** | — | — | — | ~9,000 |
| **INT8 TC TOPS** | — | 624 | ~1,979 | ~4,500 |
| **HBM 容量** | 32 GB | 80 GB | 80 GB | 192 GB |
| **HBM 带宽** | 900 GB/s | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s |
| **L2 Cache** | 6 MB | 40 MB | 50 MB | 192 MB |
| **Shared Mem/SM** | 96 KB | 164 KB | 228 KB | 228 KB |
| **TDP** | 300W | 400W | 700W | 1000W |

### 10.2 每瓦性能趋势

```
FP16 TFLOPS/Watt:
V100:  125/300  = 0.42
A100:  312/400  = 0.78  (1.9×)
H100:  990/700  = 1.41  (3.4×)
B200:  2250/1000 = 2.25  (5.4×)
```

能效随代次显著提升，主要受益于低精度计算和操作数来源优化。

---

## 11. LLM Kernel 开发中的架构选型

### 11.1 按目标架构选择编程接口

| 目标架构 | 推荐 MMA 接口 | 推荐数据加载 | 备注 |
|---------|-------------|------------|------|
| SM 8.0 (A100) | `mma.sync.m16n8k16` | `cp.async` | CUTLASS 3.x / 手写 PTX |
| SM 8.6 (RTX 3090) | `mma.sync.m16n8k16` | `cp.async` | 同 A100，注意 SM 数差异 |
| SM 8.9 (RTX 4090) | `mma.sync.m16n8k16` (含 FP8) | `cp.async` | Ada 支持 FP8 MMA |
| SM 9.0 (H100) | `wgmma.mma_async` | TMA | 必须用 Warpgroup + TMA |
| SM 10.0 (B200) | `tcgen05.mma` | TMA | TMEM + CTA Pair (数据中心独有) |
| SM 12.0 (RTX 5090) | `mma.sync` (扩展版) | TMA / `cp.async` | Ampere 编程模型 + 新数据类型 (FP8/FP4/FP6) |

### 11.2 关键架构差异对 Kernel 设计的影响

| 设计维度 | Ampere (SM 8.x) | Hopper (SM 9.0) | Blackwell (SM 10.0) |
|---------|----------------|----------------|-------------------|
| Thread Block 大小 | 128–256 | 128 (Warpgroup) | 128–256 |
| 数据搬运 | `cp.async` + 双缓冲 | TMA + `mbarrier` | TMA + `mbarrier` |
| MMA 执行 | 同步 (mma.sync) | 异步 (wgmma) | 异步 (tcgen05) |
| 累加器存储 | 寄存器 | 寄存器 | TMEM |
| Shared Mem 布局 | Swizzle / Padding | TMA Swizzle (128B) | TMA Swizzle |
| 最大 MMA Tile | m16n8k16 | m64n256k16 | m256n256k16 (2SM) |
| Warp 专化 | 可选 | 推荐 (Producer/Consumer) | 推荐 |

### 11.3 Tensor Core 饱和度检查

```
Tensor Core 利用率 = 实际 TFLOPS / 峰值 TFLOPS

目标: > 70% (高性能 GEMM)

影响因素:
1. 数据搬运是否跟得上 TC 消费速度 (TMA/cp.async 流水线深度)
2. MMA tile 是否足够大 (增加算术强度)
3. Epilogue (存储 + 后处理) 是否与下一轮 MMA 重叠
4. 寄存器/Shared Memory 是否允许足够的 occupancy
```

---

## 参考资源

- [NVIDIA Tensor Core Evolution: From Volta To Blackwell (SemiAnalysis)](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)
- [Modal GPU Glossary: Tensor Core](https://modal.com/gpu-glossary/device-hardware/tensor-core)
- [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [NVIDIA RTX Blackwell GPU Architecture Whitepaper](https://images.nvidia.cn/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- [Microbenchmarking NVIDIA's Blackwell Architecture (arXiv)](https://arxiv.org/html/2512.02189v1)
- [Programming Tensor Cores in CUDA 9 (NVIDIA Blog)](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [Colfax Research: CUTLASS Tutorial — WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
- [Colfax Research: CUTLASS Tutorial — Tensor Memory on Blackwell](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
- [Accelerating Sparsity in the NVIDIA Ampere Architecture (NVIDIA Blog)](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)

---

*本文档作为 LLM Kernel Agent 的 Tensor Core 硬件架构参考。配合 `mma-wmma.md`（编程接口）和 `official-std/` 目录下的文档共同使用。*
