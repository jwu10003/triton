# 计算与访存效率平衡 (Compute-Memory Efficiency) 深度指南

> 面向 LLM 高性能 Kernel 开发的性能瓶颈分析、Roofline 建模与优化策略全面解析
> 覆盖 Roofline Model、Arithmetic Intensity 计算、三类瓶颈诊断、Memory-Bound/Compute-Bound 优化、Latency Hiding、Kernel Fusion、LLM 推理实战

---

## 目录

1. [性能瓶颈概述](#1-性能瓶颈概述)
2. [Roofline Model 原理](#2-roofline-model-原理)
3. [Arithmetic Intensity 计算方法](#3-arithmetic-intensity-计算方法)
4. [GPU 硬件参数与 Ridge Point](#4-gpu-硬件参数与-ridge-point)
5. [三类性能瓶颈诊断](#5-三类性能瓶颈诊断)
6. [Memory-Bound Kernel 优化策略](#6-memory-bound-kernel-优化策略)
7. [Compute-Bound Kernel 优化策略](#7-compute-bound-kernel-优化策略)
8. [Latency-Bound 与延迟隐藏](#8-latency-bound-与延迟隐藏)
9. [双 Roofline：CUDA Core vs Tensor Core](#9-双-rooflinecuda-core-vs-tensor-core)
10. [Hierarchical Roofline：多级缓存 Roofline](#10-hierarchical-roofline多级缓存-roofline)
11. [Kernel Fusion 提升 Arithmetic Intensity](#11-kernel-fusion-提升-arithmetic-intensity)
12. [LLM Kernel 实战：瓶颈分析与优化](#12-llm-kernel-实战瓶颈分析与优化)
13. [Nsight Compute Roofline 诊断](#13-nsight-compute-roofline-诊断)
14. [优化决策流程图与检查清单](#14-优化决策流程图与检查清单)

---

## 1. 性能瓶颈概述

### 1.1 GPU Kernel 的三类瓶颈

GPU Kernel 的性能受三个因素制约——**计算吞吐、访存带宽、指令开销**。理解当前瓶颈是优化的第一步：

```
GPU Kernel 性能瓶颈
├── Compute-Bound (计算受限)
│     Kernel 执行时间由 ALU / Tensor Core 计算能力决定
│     → 算术强度 (AI) > Ridge Point
│     → 优化方向: 减少指令数、提高 ILP、使用 Tensor Core
│
├── Memory-Bound (访存受限)
│     Kernel 执行时间由内存带宽决定
│     → 算术强度 (AI) < Ridge Point
│     → 优化方向: 减少访存量、提高 Cache 命中、Kernel Fusion
│
└── Latency-Bound (延迟受限 / Overhead-Bound)
      Kernel 执行时间由指令延迟 + 同步开销主导
      → 既未达到计算峰值, 也未饱和带宽
      → 优化方向: 提高 Occupancy、增加 ILP、减少同步
```

### 1.2 LLM 推理中各 Kernel 的典型瓶颈

| Kernel 类型 | Prefill 阶段 | Decode 阶段 | 说明 |
|------------|:----------:|:----------:|------|
| **GEMM (QKV/FFN 投影)** | Compute-Bound | Memory-Bound | Decode 退化为 GEMV (batch=1 时 AI < 1) |
| **Attention (FlashAttention)** | Compute-Bound | Memory-Bound | Decode 时 KV-Cache 扫描破坏 L2 局部性 |
| **FFN (SwiGLU/GELU)** | Compute-Bound (~95 FLOP/B) | Memory-Bound (~8 FLOP/B) | AI 从 Prefill 到 Decode 暴跌 12× |
| **RMSNorm / LayerNorm** | Memory-Bound | Memory-Bound | 始终 memory-bound (AI < 1) |
| **Softmax** | Memory-Bound | Memory-Bound | 纯 elementwise + reduction |
| **RoPE** | Memory-Bound | Memory-Bound | 三角函数计算量小, 受读写带宽限制 |
| **Embedding / Sampling** | Memory-Bound | Memory-Bound | 查表 + scatter/gather |

> **关键洞察：** Prefill 阶段的大矩阵乘法通常是 compute-bound，而 Decode 阶段由于 batch 维度退化（单 token 生成），几乎所有 Kernel 都变为 memory-bound。这就是为什么 LLM 推理优化的核心挑战在于 **最大化 Decode 阶段的访存效率**。

---

## 2. Roofline Model 原理

### 2.1 Roofline 模型定义

Roofline Model 是一个可视化性能分析框架，将 Kernel 的性能上限建模为**计算峰值**和**带宽峰值**的函数：

```
                          性能 (FLOP/s)
                              │
                              │
Peak Compute ─────────────────┼──────────────────── (水平屋顶)
                             /│
                            / │
                           /  │        Compute-Bound 区域
                          /   │
                         /    │
                        /     │
     Memory-Bound 区域 /      │
                      /       │
                     / ←Ridge Point
                    /         │
                   /          │
                  /           │
                 /            │
                /             │
──────────────/───────────────┼──────────────────── Arithmetic Intensity
              0            Ridge              (FLOP/Byte)
                           Point
```

**核心公式：**

```
Attainable Performance = min(Peak_Compute, AI × Peak_Bandwidth)
```

其中：
- **Peak_Compute** = 硬件计算峰值 (FLOP/s)
- **Peak_Bandwidth** = 硬件内存带宽峰值 (Byte/s)
- **AI (Arithmetic Intensity)** = Kernel 的算术强度 (FLOP/Byte)

### 2.2 Roofline 的关键概念

| 概念 | 定义 | 公式 |
|------|------|------|
| **Arithmetic Intensity (AI)** | 每字节访存执行的浮点运算数 | `FLOP / Byte_transferred` |
| **Ridge Point** | 斜坡与水平屋顶的交点 | `Peak_Compute / Peak_Bandwidth` |
| **Machine Balance** | 硬件的计算带宽比 = Ridge Point | (FLOP/Byte) |
| **Achieved Performance** | Kernel 实际达到的 FLOP/s | 由 Profiler 测量 |
| **Performance Gap** | Achieved 到 Roofline 的距离 | 优化空间的度量 |

### 2.3 Roofline 上的性能解读

```
                     FLOP/s (log)
                         │
Peak ════════════════════╪════════════════════ Roofline
                        ╱│
                       ╱ │
     ★ C             ╱  │       ★ Kernel A: 离 memory roof 很近
       ╱            ╱   │         → 已接近带宽极限, 需提高 AI
      ╱            ╱    │       ★ Kernel B: 离 memory roof 很远
     ╱   ★ A      ╱     │         → 带宽未饱和, 有大量优化空间
    ╱            ╱      │       ★ Kernel C: 离 compute roof 很远
   ╱  ★ B      ╱       │         → 可增加 ILP / 用 Tensor Core
  ╱           ╱        │
 ╱           ╱         │
╱           ╱          │
────────────────────────┼───── AI (FLOP/Byte, log)
```

**关键推论：**
- AI < Ridge Point → **Memory-Bound**: 提高性能只能靠 (1) 减少访存量 (2) 提高带宽利用率
- AI > Ridge Point → **Compute-Bound**: 提高性能只能靠 (1) 减少指令数 (2) 用更快的计算单元 (Tensor Core)
- AI ≈ Ridge Point 且靠近 Roofline → **最理想状态**: 计算与访存同时饱和

---

## 3. Arithmetic Intensity 计算方法

### 3.1 通用公式

```
AI = Total_FLOPs / Total_Bytes_Transferred
```

**注意事项：**
- FLOPs 包含所有浮点运算（加减乘除、FMA 算 2 个 FLOPs）
- Bytes 指 **实际** 与 Global Memory / HBM 交换的字节数（不是理论最小值）
- Nsight Compute 测量的是实际 DRAM 事务，包含 Cache Miss 导致的冗余传输

### 3.2 GEMM 的 Arithmetic Intensity

GEMM: C(M×N) = A(M×K) × B(K×N)

```
Total FLOPs  = 2 × M × N × K          (每个输出元素: K次乘 + K次加 = 2K FLOPs)
Total Bytes  = P × (M×K + K×N + M×N)   (读 A + 读 B + 写 C; 典型 β=0 不读 C)
其中 P = 每个元素的字节数 (FP16: 2, FP32: 4, BF16: 2)

AI_GEMM = 2×M×N×K / [P × (M×K + K×N + M×N)]
```

> **FLOPs 约定：** FMA (Fused Multiply-Add) 计为 2 FLOPs，与 NVIDIA 公布的峰值 TFLOPS 一致。GEMM 每输出元素含 K 次 FMA = 2K FLOPs。

**简化形式 (FP16, P=2, M=N=K=n):**

```
AI = 2n³ / [2 × 3n²] = n / 3     (大方阵 GEMM)
```

**不同形状 GEMM 的 AI 示例 (FP16):**

| M | N | K | AI (FLOP/Byte) | H100 瓶颈 (Ridge ≈ 296) |
|:---:|:---:|:---:|:--------------:|:----------------------:|
| 1 | 4096 | 4096 | 1.0 | **Memory-Bound** (GEMV) |
| 8 | 4096 | 4096 | 8.0 | **Memory-Bound** |
| 64 | 4096 | 4096 | 62 | **Memory-Bound** |
| 512 | 4096 | 4096 | 410 | Compute-Bound |
| 2048 | 4096 | 4096 | 1024 | Compute-Bound |
| 4096 | 4096 | 4096 | 1365 | Compute-Bound |
| 8192 | 8192 | 8192 | 2731 | Compute-Bound |

> **LLM 洞察：** Decode 阶段 batch=1 时，线性层的 GEMM 退化为 GEMV (M=1)，AI ≈ 1 FLOP/Byte，远低于任何 GPU 的 Ridge Point。即使 batch=64，AI 也只有约 62，仍低于 H100 的 296。直到 batch≈300+ 才跨越 Ridge Point 进入 compute-bound。这就是 Decode 阶段 GEMM 几乎总是 memory-bound 的根本原因。

### 3.3 Elementwise 操作的 AI

```
例: y = GELU(x)    x, y 各 n 个 FP16 元素
  FLOPs ≈ 8n  (x³, 系数乘, 加, tanh(SFU 单指令), 加 1, 乘 x, 乘 0.5)
  Bytes = 2n × 2 = 4n  (读 x + 写 y)
  AI = 8n / 4n = 2 FLOP/Byte

例: y = x_gate ⊙ SiLU(x_up)  (SwiGLU, 标准 2 输入)
  其中 SiLU(x) = x × sigmoid(x)
  FLOPs ≈ 6n  (sigmoid ~4 ops + 乘 x_up + 乘 x_gate)
  Bytes = 3 × 2n = 6n  (读 x_gate + 读 x_up + 写 y)
  AI = 6n / 6n = 1.0 FLOP/Byte
```

**结论：** 所有 elementwise 操作的 AI 都是 O(1) 常数，不随 n 增长，因此**永远是 memory-bound**。

### 3.4 Reduction 操作的 AI

```
例: RMSNorm   y_i = x_i × rsqrt(mean(x²) + ε) × γ_i
  每个元素: ~4 FLOPs (x²:1, 累加:1, ×rsqrt:1, ×γ:1; mean/rsqrt 共享全行均摊≈0)
  Bytes/元素: 读 x (2B) + 读 γ (2B) + 写 y (2B) = 6B
  AI ≈ 4/6 ≈ 0.67 FLOP/Byte

例: Softmax   y_i = exp(x_i - max) / sum(exp(x_j - max))
  需要 3 pass: max-reduce, exp + sum-reduce, 除法
  FLOPs ≈ 5n
  Bytes = 2n (读) + 2n (写) = 4n  (若 fused 单 pass)
  AI ≈ 5/4 ≈ 1.25 FLOP/Byte
```

### 3.5 Attention 的 AI

```
Self-Attention: softmax(Q·K^T / √d) · V
  Q, K, V: (B, H, S, d)   S=seq_len, d=head_dim

Prefill (S 很大):
  QK^T: 2×S²×d FLOPs, 读 Q+K: 4Sd bytes
  AI_QKT ≈ S/2    (S=2048 → AI=1024, compute-bound)

Decode (S_q=1, S_kv=S):
  QK^T: 2×1×S×d FLOPs, 读 K-cache: 2Sd bytes
  AI_QKT ≈ 1 FLOP/Byte    (始终 memory-bound)
```

### 3.6 AI 速查表

| 操作 | AI (FLOP/Byte) | 特性 |
|------|:--------------:|------|
| GEMM (M=N=K=4096, FP16) | ~1365 | 大矩阵 compute-bound |
| GEMV (M=1, N=K=4096, FP16) | ~1.0 | 始终 memory-bound |
| Batched GEMM (M=256, N=K=4096, FP16) | ~228 | 仍为 memory-bound (H100) |
| FlashAttention Prefill (S=2048) | ~1000+ | Compute-bound |
| FlashAttention Decode (S=1, S_kv=2048) | ~1 | Memory-bound |
| RMSNorm | ~0.67 | Memory-bound |
| Softmax | ~1.25 | Memory-bound |
| GELU / SiLU | ~2 | Memory-bound |
| SwiGLU | ~1.0 | Memory-bound |
| Residual Add | ~0.17 | Memory-bound (AI ≈ 1/6) |
| Embedding Lookup | ~0 | 纯访存 |

---

## 4. GPU 硬件参数与 Ridge Point

### 4.1 各架构硬件规格 (SXM / DC 版本)

| GPU | 显存 | 带宽 (TB/s) | FP32 CUDA (TFLOPS) | FP16/BF16 TC Dense (TFLOPS) | FP8 TC Dense (TFLOPS) |
|-----|:----:|:----------:|:------------------:|:--------------------------:|:--------------------:|
| **V100** | 32 GB HBM2 | 0.90 | 15.7 | 125 | — |
| **A100** | 80 GB HBM2e | 2.0 | 19.5 | 312 | — |
| **H100** | 80 GB HBM3 | 3.35 | 67 | 990 | 1,979 |
| **H200** | 141 GB HBM3e | 4.8 | 67 | 990 | 1,979 |
| **B200** | 180 GB HBM3e | 8.0 | 75 | 2,250 | 4,500 |
| **RTX 5090** | 32 GB GDDR7 | 1.79 | 105 | 210† | 419† |

> †RTX 5090 TC TFLOPS 为 FP32 累加器模式 (消费级 Blackwell FP32 累加半速)。FP16 累加器时: FP16 TC ~419, FP8 TC ~838 TFLOPS。

### 4.2 Ridge Point 对比

Ridge Point = Peak FLOP/s ÷ Peak Bandwidth

| GPU | FP32 CUDA Core Ridge | FP16 TC Dense Ridge | FP8 TC Dense Ridge |
|-----|:--------------------:|:-------------------:|:------------------:|
| **V100** | 17 | 139 | — |
| **A100** | 10 | 156 | — |
| **H100** | 20 | **296** | 591 |
| **H200** | 14 | **206** | 412 |
| **B200** | 9 | **281** | 563 |
| **RTX 5090** | 59 | **117** | 234 |

```
         Ridge Point 可视化 (FP16 TC Dense, FLOP/Byte)

5090  █████████████████████████████████████████ 117
V100  ████████████████████████████████████████████████████ 139
A100  ██████████████████████████████████████████████████████████ 156
H100  ██████████████████████████████████████████████████████████████████████████████████████████████████████████ 296
H200  ████████████████████████████████████████████████████████████████████████████ 206
B200  ████████████████████████████████████████████████████████████████████████████████████████████████████ 281
```

**关键观察：**
- RTX 5090 的 Ridge Point 最低 (117)——GDDR7 带宽 1.79 TB/s 与消费级 TC 半速 FP32 累加共同拉低了 Ridge Point，使绝大多数 Kernel 都能达到 compute-bound
- H200 的 Ridge Point 次低 (206)——因为它只增加了带宽而计算峰值不变（与 H100 同架构），使更多 Kernel 能进入 compute-bound 区域
- B200 虽然带宽翻倍 (8 TB/s)，但计算峰值增长更快 (2,250 vs 990 TFLOPS)，Ridge Point 回升至 281
- FP8 TC 的 Ridge Point 接近 600，意味着只有超大矩阵 GEMM 才能真正 compute-bound
- 代际趋势：计算能力增速 > 带宽增速 → Ridge Point 总体上升 → **更多 Kernel 变为 Memory-Bound**

### 4.3 实际 vs 理论峰值

理论峰值在实际中几乎无法完全达到，实测通常为：

| 指标 | 实际可达比例 | 说明 |
|------|:----------:|------|
| HBM 带宽 | 80–90% | 受 Cache Line 利用率、非合并访存影响 |
| FP32 CUDA Core | 70–85% | 受指令 Mix、分支分歧影响 |
| FP16 Tensor Core | 60–80% | 受 SMEM bank conflict、Pipeline bubble 影响 |
| FP8 Tensor Core | 50–75% | 数据搬运开销更突出 |

> **实践建议：** 计算 Ridge Point 时使用实测峰值 (而非理论峰值) 更有指导意义。Nsight Compute 的 Roofline 图默认使用经验值 (empirical peak)。

---

## 5. 三类性能瓶颈诊断

### 5.1 诊断流程

```
Kernel Performance 诊断
│
├─ 1. 测量 AI (Arithmetic Intensity)
│     ncu --set roofline ./my_app
│     或手动计算: FLOPs / DRAM_bytes
│
├─ 2. 比较 AI 与 Ridge Point
│     ├── AI < Ridge Point → 疑似 Memory-Bound
│     └── AI > Ridge Point → 疑似 Compute-Bound
│
├─ 3. 检查达到率
│     ├── Memory-Bound: 实际带宽 / 峰值带宽 → 带宽达到率
│     │    ├── > 80% → 已接近带宽极限 (需提高 AI)
│     │    └── < 50% → 带宽未饱和 → 可能是 Latency-Bound
│     │
│     └── Compute-Bound: 实际 FLOP/s / 峰值 FLOP/s → 计算达到率
│          ├── > 70% → 已接近计算极限 (需减少计算量)
│          └── < 50% → 计算单元空闲 → 可能是 Latency-Bound
│
└─ 4. Latency-Bound 确认
      若带宽达到率 < 60% 且计算达到率 < 60%
      → 大概率 Latency-Bound → 检查 Occupancy / 同步 / 分支
```

### 5.2 Nsight Compute 关键 Metrics

| 瓶颈类型 | 关键 Metric | 阈值 |
|---------|-----------|------|
| **Memory-Bound** | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | > 60% |
| | `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | 检查 L1 是否瓶颈 |
| **Compute-Bound** | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | > 60% |
| | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | TC 利用率 |
| **Latency-Bound** | `sm__warps_active.avg.pct_of_peak_sustained_active` | < 50% (Occupancy 不足) |
| | `sm__instruction_throughput.avg.pct_of_peak_sustained_elapsed` | < 30% (指令吞吐低) |
| | `smsp__warp_issue_stalled_*` | 哪类 stall 最多 |

### 5.3 三类瓶颈的典型特征

```
┌────────────────┬───────────────────┬──────────────────┬─────────────────┐
│     特征       │  Memory-Bound     │  Compute-Bound   │  Latency-Bound  │
├────────────────┼───────────────────┼──────────────────┼─────────────────┤
│ AI vs Ridge    │ AI < Ridge        │ AI > Ridge       │ 任意            │
│ 带宽达到率     │ > 60%             │ < 30%            │ < 60%           │
│ 计算达到率     │ < 30%             │ > 60%            │ < 60%           │
│ Occupancy      │ 通常足够          │ 通常足够          │ 通常不足        │
│ 主要 Stall     │ memory dependency │ execution dep    │ barrier/sync    │
│ 数据增大时     │ 时间线性增长       │ 时间线性增长      │ 时间增长缓慢    │
└────────────────┴───────────────────┴──────────────────┴─────────────────┘
```

---

## 6. Memory-Bound Kernel 优化策略

Memory-Bound 是 LLM 推理（特别是 Decode 阶段）中最常见的瓶颈。优化目标：**减少 DRAM 访问量** 或 **提高每字节访存的有效计算量 (AI)**。

### 6.1 减少访存量

#### 6.1.1 Kernel Fusion

消除中间结果写回 Global Memory 的开销：

```
融合前: 3 个 Kernel, 6 次 Global Memory 往返
┌─────────┐     ┌─────────┐     ┌─────────┐
│ LayerNorm│ ──→ │ Linear  │ ──→ │  GELU   │
│ 读 x     │     │ 读 y₁   │     │ 读 y₂   │
│ 写 y₁   │     │ 写 y₂   │     │ 写 y₃   │
└─────────┘     └─────────┘     └─────────┘
Global Memory: 读 x → 写 y₁ → 读 y₁ → 写 y₂ → 读 y₂ → 写 y₃

融合后: 1 个 Kernel, 2 次 Global Memory 往返
┌────────────────────────────────┐
│ Fused: LayerNorm + Linear + GELU │
│ 读 x                            │
│ (中间结果留在 Register / SMEM)   │
│ 写 y₃                           │
└────────────────────────────────┘
Global Memory: 读 x → 写 y₃
```

**AI 提升效果：**
- 融合前总 bytes: 6n × P (3 读 + 3 写)
- 融合后总 bytes: 2n × P (1 读 + 1 写)
- 访存减少 3×, AI 提升 3×

#### 6.1.2 量化 / 低精度

降低每个元素的字节数直接提高 AI：

| 精度 | 字节/元素 | 相对 FP16 的 AI 提升 |
|------|:--------:|:-------------------:|
| FP32 | 4 | 0.5× |
| FP16 / BF16 | 2 | 1× (基准) |
| FP8 (E4M3/E5M2) | 1 | 2× |
| INT8 | 1 | 2× |
| INT4 / FP4 | 0.5 | 4× |

> **GEMV 示例 (K=4096):** FP16 时 AI≈1.0，INT4 时 AI≈4.0。虽然仍远低于 Ridge Point，但带宽需求减少 4×，Decode 延迟直接降低 4×。

#### 6.1.3 向量化访存

使用宽指令减少指令数并提高 L1/L2 Cache 效率：

```
FP16 标量加载:   LDG.E.16   → 每条指令 2 Bytes
FP16 向量化:     LDG.E.128  → 每条指令 16 Bytes  (8× 效率)

Warp (32 threads) 每次访问的数据量:
  标量: 32 × 2B = 64 Bytes  (2 sectors, 128B cache line 填充率 50%)
  float4: 32 × 16B = 512 Bytes (16 sectors, 完美填充)
```

#### 6.1.4 数据重用与 Tiling

将频繁访问的数据缓存到 Shared Memory 或 Register File，避免重复读取 DRAM：

```
GEMM Tiling 中的数据重用:
  不做 Tiling: 每计算 C 的一个元素, 读 A 的一行 + B 的一列 → O(MNK) 次访存
  使用 Tiling: Block 共享 SMEM tile → 每个 tile 只从 DRAM 读一次

  Tile 大小 T×T 时:
    计算量: T × T × K FLOPs (per block)
    访存量: 2 × T × K bytes (读 A tile + B tile)
    AI ≈ T / 2    (T=128 → AI=64)

  → Tile 越大, AI 越高, 但受 SMEM 容量限制
```

### 6.2 提高带宽利用率

| 技术 | 方法 | 效果 |
|------|------|------|
| **合并访存** | Warp 内连续 thread 访问连续地址 | 1 个 128B 事务 vs 32 个独立事务 |
| **L2 持久化** | `cudaAccessPolicyWindow` 固定热数据 | KV Cache 的 K/V 驻留 L2 |
| **Prefetch** | `prefetch.global.L2` / TMA bulk prefetch | 隐藏 DRAM 延迟 |
| **Threadblock Swizzle** | 重排 Block 执行顺序 | L2 命中率提升可达 60% |
| **避免 SMEM Bank Conflict** | Padding / XOR Swizzle | SMEM 带宽不降级 |

### 6.3 重计算 (Recomputation) 替代存储

FlashAttention 的核心思想：宁可重算，不存中间结果：

```
标准 Attention:
  S = Q·K^T                    → 存 S (S×S 矩阵, 巨大!)
  P = softmax(S)              → 存 P
  O = P·V                     → 存 O
  反向传播时读 S, P           → 3 次 HBM 往返

FlashAttention:
  分块计算, 不存 S, P          → 只存 O + 少量 Online Softmax 状态 (max, sum)
  反向传播时重新计算 S, P     → 1 次 HBM 往返
  FLOP 增加 ~25%, 但 HBM 访问减少 ~90%
```

---

## 7. Compute-Bound Kernel 优化策略

Compute-Bound Kernel 在 LLM 的 Prefill 阶段和 Training 中最常见。优化目标：**减少不必要的计算** 或 **使用更高效的计算单元**。

### 7.1 使用 Tensor Core 替代 CUDA Core

| GPU | FP16 CUDA Core (TFLOPS) | FP16 Tensor Core (TFLOPS) | TC/CC 比 |
|-----|:----------------------:|:------------------------:|:--------:|
| V100 | 31.4 | 125 | 4× |
| A100 | 78 | 312 | 4× |
| H100 | 134 | 990 | 7.4× |
| B200 | 150 | 2,250 | 15× |

> **TC/CC 比逐代扩大**：从 V100 的 4× 到 B200 的 15×。这意味着不使用 Tensor Core 的 Kernel 在新硬件上的 "浪费" 越来越大。

### 7.2 减少指令数

#### 7.2.1 Fast Math Intrinsics

```
标准函数          → Intrinsic              → 效果
expf(x)          → __expf(x)              → SFU 单指令, ~10× 更快
a / b            → __fdividef(a,b)        → 快速浮点除法 (精度稍低, 2 ULP)
                   或 a * __frcp_rn(b)     → SFU 倒数 + 1 乘, 更快但精度更低
sinf/cosf        → __sincosf              → 1 次调用代替 2 次
```

#### 7.2.2 Loop Unrolling 减少控制开销

```
// 未展开: 每次迭代有循环控制开销 (比较 + 分支 + 计数)
for (int i = 0; i < N; i++) {
    acc += x[i] * w[i];
}

// 展开后: 减少循环控制指令, 暴露 ILP
#pragma unroll 4
for (int i = 0; i < N; i += 4) {
    acc += x[i]   * w[i];
    acc += x[i+1] * w[i+1];
    acc += x[i+2] * w[i+2];
    acc += x[i+3] * w[i+3];
}
```

#### 7.2.3 Packed Arithmetic (半精度打包)

```
// 标量: 2 条指令
half a1 = ..., a2 = ...;
half b1 = a1 + 1.0h;       // HADD
half b2 = a2 + 1.0h;       // HADD

// 打包: 1 条指令
half2 a = ...;
half2 b = __hadd2(a, __float2half2_rn(1.0f));  // HADD2 (2 个 FP16 同时计算)
```

### 7.3 提高指令级并行 (ILP)

```
// 低 ILP: 指令间有依赖链
acc = x[0] * w[0];          // 必须等这条完成
acc += x[1] * w[1];         // 才能执行这条 (RAW 依赖)
acc += x[2] * w[2];
acc += x[3] * w[3];

// 高 ILP: 独立累加器
acc0  = x[0] * w[0];        // 4 条乘法可以并行发射
acc1  = x[1] * w[1];
acc2  = x[2] * w[2];
acc3  = x[3] * w[3];
acc = acc0 + acc1 + acc2 + acc3;  // 最后合并
```

### 7.4 Warp Specialization (Hopper+)

将 Warp 分为 Producer (数据搬运) 和 Consumer (计算)，使计算流水线永不断流：

```
传统 Multistage Pipeline:
  同一 Warp 交替执行 Load 和 Compute → 寄存器被两种工作共享

Warp Specialization:
  Producer Warps: 只做 TMA Load, 寄存器需求极低 (~24 regs)
  Consumer Warps: 只做 WGMMA, 寄存器全部给累加器 (~232 regs)
  → 计算单元利用率从 ~60% 提升到 ~99%
```

### 7.5 Compute-Bound 优化总结

```
Compute-Bound 优化方向:
├── 使用更快的计算单元
│     └── CUDA Core → Tensor Core (4-15× 加速)
├── 减少总指令数
│     ├── Fast Math Intrinsics
│     ├── Loop Unrolling
│     ├── Packed Arithmetic (half2/bfloat162)
│     └── 避免冗余计算 (CSE 公共子表达式消除)
├── 提高指令级并行 (ILP)
│     ├── 多累加器 / Register Blocking
│     └── 解开数据依赖链
├── 提高 Pipeline 利用率
│     ├── Warp Specialization (Hopper+)
│     └── Multistage Async Pipeline
└── 算法优化
      ├── 降低计算复杂度 (O(n²) → O(n log n))
      └── 结构化稀疏 (跳过零值计算)
```

---

## 8. Latency-Bound 与延迟隐藏

### 8.1 Latency-Bound 的本质

当一个 Kernel 既没有达到计算峰值，也没有饱和内存带宽时，它通常是 **Latency-Bound**（也称 Overhead-Bound）。根本原因是 SM 上没有足够的并发操作来隐藏各种延迟。

### 8.2 Little's Law 与延迟隐藏

```
Little's Law:
  Required_Concurrency = Throughput × Latency

例: HBM 访问
  Latency = 400 cycles
  Throughput = 128 Bytes/cycle (per SM)
  Required_Concurrency = 400 × 128 = 51,200 Bytes in flight

  每个 Warp 若发 1 个 128B 请求:
  需要 51,200 / 128 = 400 个并发请求
  但每 SM 最多 ~48-64 Warps → 每 Warp 需发 ~6-8 个独立内存请求
```

### 8.3 影响延迟隐藏的因素

| 因素 | 影响 | 优化方向 |
|------|------|---------|
| **Occupancy** | 更多 Active Warps → 更多可切换的上下文 | 减少寄存器用量、SMEM 用量 |
| **ILP (指令级并行)** | 同一 Warp 内独立指令可掩盖延迟 | 多累加器、循环展开 |
| **TLP (线程级并行)** | 多 Warp 切换掩盖延迟 | 提高 Occupancy |
| **异步操作** | cp.async / TMA 不阻塞 Warp | Hopper+ 的 TMA + mbarrier |

### 8.4 Occupancy 与性能的非线性关系

```
性能
│                          ┌────── 性能饱和区
│                         /        (更多 Warp 不再有帮助)
│                        /
│                       /
│                      /  ← 拐点 (通常 50-75% Occupancy)
│                     /
│                    /
│                   /   ← 线性增长区
│                  /        (每增加 Warp 都能隐藏更多延迟)
│                 /
│                /
│               /
│──────────────────────────────── Occupancy (%)
0          25%        50%       75%       100%
```

**Volkov 的关键洞察：** 高 Occupancy 不是目的，足够的并发度才是。有时降低 Occupancy（让每线程使用更多寄存器）反而能提升性能——因为更多寄存器 → 更大 Tile → 更高 AI → 从 memory-bound 变为 compute-bound。

### 8.5 Latency-Bound 优化清单

```
Latency-Bound 诊断确认:
  sm__throughput < 60% 且 dram__throughput < 60%

优化方向:
├── 提高并发度
│     ├── 增加 Occupancy (减少寄存器/SMEM)
│     ├── 增加 ILP (循环展开/多累加器)
│     └── 使用异步操作 (cp.async/TMA)
│
├── 减少同步开销
│     ├── 减少 __syncthreads() 调用
│     ├── 使用 Warp-level 原语 (__syncwarp 替代 block sync)
│     └── 使用 mbarrier 细粒度同步 (Hopper+)
│
├── 减少指令开销
│     ├── 减少循环控制 (unroll)
│     ├── 减少地址计算 (向量化/预计算)
│     └── 减少分支分歧
│
└── 增大问题规模
      (若小问题 → Grid 太小 → 无法填满 GPU → 天然 latency-bound)
      ├── 增大 Batch Size
      ├── 合并多个小 Kernel (CUDA Graph)
      └── Persistent Kernel (长驻 SM)
```

---

## 9. 双 Roofline：CUDA Core vs Tensor Core

### 9.1 为什么需要双 Roofline

现代 GPU 拥有两套独立的计算子系统，各有不同的峰值性能：

```
         FLOP/s (log)
             │
TC Peak ═════╪═══════════════════════════════════ Tensor Core Roofline
             │                                    (FP16 TC: 990 TFLOPS on H100)
             │              /
             │             /
CC Peak ═════╪════════════/═══════════════════════ CUDA Core Roofline
             │           /                        (FP32: 67 TFLOPS on H100)
             │          /
             │         /    ★ GEMM (TC)  → 对标 TC Roofline
             │        /
             │       /      ★ GELU (CC)  → 对标 CC Roofline
             │      /
             │     /
             │    / ← 共用 Memory Roof (同一带宽)
             │   /
             │  /
             │ /
             │/
─────────────┼────────────────────────── AI (FLOP/Byte, log)
```

### 9.2 Kernel 到 Roofline 的映射

| Kernel 类型 | 对标 Roofline | 理由 |
|------------|:----------:|------|
| GEMM (mma.sync/wgmma) | TC Roofline | 核心计算在 Tensor Core 上 |
| FlashAttention | TC Roofline | QK^T 和 PV 用 Tensor Core |
| RMSNorm / Softmax | CC Roofline | 纯 FP32/FP16 标量运算 |
| GELU / SiLU / SwiGLU | CC Roofline | elementwise 运算 |
| 量化/反量化 | CC Roofline | 整数运算 + 缩放 |
| Fused Norm + Linear | 混合 | Norm 在 CC, Linear 在 TC |

### 9.3 混合 Kernel 的 Roofline 分析

对于 Fused Kernel (如 FlashAttention 融合了 GEMM + Softmax)：

```
实际场景: FlashAttention = GEMM (TC) + Softmax (CC) + GEMM (TC)

方法 1: 分别画两个点
  - GEMM 部分的 AI 和 FLOP/s → 对标 TC Roofline
  - Softmax 部分的 AI 和 FLOP/s → 对标 CC Roofline

方法 2: Nsight Compute 自动处理
  ncu 会分别统计 TC pipeline 和非 TC pipeline 的利用率
  → sm__pipe_tensor_op_hmma_cycles_active 显示 TC 利用率
  → sm__pipe_fma_cycles_active 显示 FMA (CC) 利用率
```

---

## 10. Hierarchical Roofline：多级缓存 Roofline

### 10.1 传统 Roofline 的局限

传统 Roofline 只考虑 DRAM 带宽。但如果 Kernel 的工作集驻留在 L2 或 L1 中，实际可用带宽远高于 DRAM：

```
L1 带宽 >> L2 带宽 >> DRAM 带宽

→ 同一 Kernel 在不同缓存层有不同的 "有效 AI"
→ 需要多层 Roofline 来准确定位瓶颈
```

### 10.2 Hierarchical Roofline 图

```
         FLOP/s (log)
             │
Compute ═════╪══════════════════════════════════════
             │            /        /        /
             │           /        /        /
             │          /        /        /
             │         / L1    / L2    / DRAM
             │        / Roof  / Roof  / Roof
             │       /        /        /
             │      /        /        /
             │     /        /   ★ Kernel (DRAM AI, actual perf)
             │    /        /        /
             │   /        /        /
             │  /        /        /
             │ /        /        /
─────────────┼──────────────────────────── AI (FLOP/Byte)
```

**多级带宽参考 (H100 SXM)：**

| 层级 | 每 SM 带宽 | 全芯片聚合带宽 | 与 DRAM 比 (聚合) |
|------|:----------:|:------------:|:----------------:|
| Register File | ~1 cycle 延迟, 非带宽受限 | — | — |
| L1 / SMEM | ~200 GB/s (128 B/cycle) | ~27 TB/s | ~8× |
| L2 | (全局共享) | ~12 TB/s | ~3.6× |
| DRAM (HBM3) | ~25 GB/s | 3.35 TB/s | 1× |

> **注：** L1/SMEM 带宽 = 128 Bytes/cycle × SM 时钟频率。per-SM DRAM 带宽 = 3.35 TB/s ÷ 132 SMs ≈ 25 GB/s。L1 相对 DRAM 的 per-SM 带宽比约为 200/25 = 8×。

### 10.3 实践意义

当 Kernel 的 AI 基于 DRAM 计算看起来是 memory-bound，但大部分数据命中 L2：

```
例: RMSNorm, hidden_dim=4096, batch=small
  DRAM AI ≈ 0.67 FLOP/Byte → memory-bound (对 DRAM roof)
  但 γ (4096 × 2B = 8 KB) 总是命中 L2
  → 实际只有 x 和 y 走 DRAM
  → "有效 AI" 更高, 性能比 DRAM-only 预测好
```

**Nsight Compute 的 Hierarchical Roofline：**
```bash
ncu --section SpeedOfLight_RooflineChart \
    --section SpeedOfLight_HierarchicalDoubleRooflineChart \
    ./my_app
```

---

## 11. Kernel Fusion 提升 Arithmetic Intensity

### 11.1 Fusion 的本质

Kernel Fusion 是**提高 Memory-Bound Kernel AI 的最有效手段**。其核心：消除中间数据对 Global Memory 的写回与重读。

### 11.2 Fusion 分类

```
Fusion 分类:
├── Vertical Fusion (纵向融合)
│     相邻的生产者-消费者 Kernel 合并
│     例: LayerNorm → Linear → GELU → 一个 Kernel
│     效果: 中间张量不写 DRAM
│
├── Horizontal Fusion (横向融合)
│     无依赖的并行 Kernel 合并
│     例: Q/K/V 三个 Linear 合并为一个
│     效果: 共享输入读取, 减少 Kernel Launch 开销
│
└── Mixed Fusion (混合融合)
      同时包含纵向和横向
      例: QKV Projection + RoPE + Split → 一个 Kernel
```

### 11.3 LLM 中常见的 Fusion Pattern

| Fusion Pattern | 融合前 AI | 融合后 AI | DRAM 访存减少 |
|---------------|:--------:|:--------:|:------------:|
| **Residual + LayerNorm** | ~0.17 + ~0.67 | ~1.0 | ~2× |
| **Linear + Bias + GELU** | 由 GEMM 决定 + ~1 | 由 GEMM 决定 | 中间结果不写回 |
| **SwiGLU = gate ⊙ SiLU(up)** | ~1.0 | ~2.5 | ~3× |
| **FlashAttention (全融合)** | S 矩阵写回 | ∞ (on-chip) | O(S²) → O(S) |
| **Fused MoE** | 独立 Expert GEMM | 共享 dispatch/combine | ~1.5× |

### 11.4 CUTLASS Epilogue Fusion

CUTLASS 提供了在 GEMM 结束后 (Epilogue 阶段) 融合自定义运算的能力：

```
标准 GEMM + 后处理:
  Kernel 1: D = A × B             (GEMM, write D to DRAM)
  Kernel 2: E = GELU(D + bias)     (读 D, 写 E)

CUTLASS Epilogue Fusion:
  Kernel 1: D_reg = A × B         (GEMM, D 在寄存器中)
             E = GELU(D_reg + bias) (直接在寄存器中后处理)
             write E to DRAM       (只写最终结果)

  → 省去 D 的 DRAM 往返, 常见加速 20-30%
```

### 11.5 Fusion 的限制

| 限制 | 说明 |
|------|------|
| **寄存器压力** | 融合更多操作需更多寄存器 → 可能降低 Occupancy |
| **SMEM 容量** | 多个操作共享 SMEM → 可能超限 |
| **代码复杂度** | Fused Kernel 更难调试和维护 |
| **编译时间** | 模板化 Fused Kernel 编译慢 (CUTLASS) |
| **形状依赖** | 不同 M/N/K 可能需要不同 Fusion 策略 |

---

## 12. LLM Kernel 实战：瓶颈分析与优化

### 12.1 Prefill vs Decode 全景分析

```
LLM 推理 Pipeline:
┌────────────────────────────────────────────────────────┐
│                    Prefill 阶段                        │
│  Input: [token₁, token₂, ..., token_S]  (S 个 token)  │
│                                                        │
│  QKV Projection: GEMM (S×D_model, D_model×3D_model)   │
│    → M=S, N=3D, K=D → 大矩阵 → Compute-Bound         │
│                                                        │
│  Self-Attention: FlashAttention(Q, K, V)               │
│    → O(S²d) FLOPs → Compute-Bound (S 大时)            │
│                                                        │
│  FFN: GEMM (S×D, D×4D) + SwiGLU + GEMM (S×4D, 4D×D)  │
│    → 大矩阵 → Compute-Bound                           │
├────────────────────────────────────────────────────────┤
│                    Decode 阶段                         │
│  Input: [new_token]  (每次 1 个 token)                 │
│                                                        │
│  QKV Projection: GEMV (1×D_model, D_model×3D_model)   │
│    → M=1 → AI≈1.0 → Memory-Bound (极端)               │
│                                                        │
│  Self-Attention: 扫描全部 KV-Cache                     │
│    → 读 S_kv×d 的 K,V → Memory-Bound                  │
│                                                        │
│  FFN: GEMV (1×D, D×4D) + SwiGLU + GEMV (1×4D, 4D×D)  │
│    → M=1 → Memory-Bound                               │
└────────────────────────────────────────────────────────┘
```

### 12.2 GEMM / GEMV 优化

**Prefill (Compute-Bound GEMM):**

```
优化重点: 最大化 Tensor Core 利用率
├── 使用 WGMMA (Hopper) / tcgen05 (Blackwell DC)
├── Tile 大小: 128×256×64 或更大
├── Warp Specialization: Producer + Consumer 流水线
├── Epilogue Fusion: 融合 bias + activation
└── 目标: >80% Tensor Core 利用率
```

**Decode (Memory-Bound GEMV):**

```
优化重点: 最大化带宽利用率
├── 权重量化: W4A16 / W4A8 / W8A8
│     → 权重 4× 更小 → 带宽需求 4× 更低
├── 向量化加载: LDG.E.128 每线程加载 16 Bytes
├── 共享权重读取: 多 Batch 共享权重 (增大 M)
│     → Continuous Batching: 凑更多请求一起推理
├── KV Cache 量化: FP8 / INT8 KV 存储
│     → 读取带宽减半
└── 目标: >80% DRAM 带宽利用率
```

### 12.3 FlashAttention 瓶颈分析

```
FlashAttention 的巧妙之处在于改变了 Roofline 上的位置:

标准 Attention:
  必须写 S 矩阵 (S×S) 到 HBM → 巨大的 IO
  AI ≈ O(1)  (FLOP 和 IO 都是 O(S²d))
  → Memory-Bound

FlashAttention:
  S 矩阵留在 SRAM (分块计算) → IO 只有 O(S×d)
  AI ≈ S×d (可以非常高)
  → 变为 Compute-Bound!

但 FlashAttention Decode (S_q=1):
  GEMM 退化为 GEMV → 回到 Memory-Bound
  → FlashDecoding: 沿 KV-Cache 维度并行化, 但瓶颈仍是带宽
```

### 12.4 Elementwise Fused Kernel 优化

```
RMSNorm + Residual Add + 存储:

未融合 (3 个 Kernel):
  K1: residual = x + residual       读 2×, 写 1×     AI ≈ 0.17
  K2: rms = sqrt(mean(x²))          读 1×             AI ≈ 0.67
  K3: y = x * rms_inv * gamma       读 2×, 写 1×     AI ≈ 0.67
  总计: 读 5× + 写 2× = 7× DRAM 往返

融合 (1 个 Kernel):
  y, residual_out = fused_rms_norm_residual(x, residual, gamma)
  读: x(1×) + residual(1×) + gamma(1×) = 3×
  写: y(1×) + residual_out(1×) = 2×
  总计: 5× DRAM 往返

  但 gamma 很小 (D_model × 2B, 常驻 L2)
  → 有效 DRAM 往返 ≈ 4× (vs 未融合的 7×)
  → 加速比 ≈ 1.75×
```

### 12.5 MoE (Mixture-of-Experts) 的特殊挑战

```
MoE 的瓶颈特殊性:
  Dense Model FFN: batch×D → batch×4D   (M=batch, AI 随 batch 增长)
  MoE: 每个 Expert 只处理 batch/num_experts 个 token
       → 有效 M = batch/num_experts
       → 每个 Expert 的 GEMM 更小、更 memory-bound

例: batch=256, num_experts=8, top_k=2, D=4096, FFN_dim=4D
  每 Expert 平均: M = 256×2/8 = 64  (远小于 Dense 的 256)
  → MoE FFN 的 AI ≈ 63 FLOP/Byte vs Dense FFN 的 ≈ 237 FLOP/Byte

优化:
  - Grouped GEMM: 将多个小 Expert GEMM 打包
  - Expert 权重量化: W4 减少每个 Expert 的读取量
  - Token Permutation: 重排 token 使连续 token 分到同一 Expert → 合并访存
```

### 12.6 Batch Size 对瓶颈的影响

```
            AI (FLOP/Byte)
              │
              │                                ┌─── Ridge Point (FP16 TC)
              │                                │
       1000 ─ │                                │     ← 大 batch Prefill GEMM
              │                                │
        100 ─ │                    ●           │     ← batch=2048
              │               ●                │     ← batch=512
         10 ─ │          ●                     │     ← batch=64
              │     ●                          │     ← batch=8
          1 ─ │●                               │     ← batch=1 (GEMV)
              │                                │
        0.1 ─ │                                │
              ├────────────────────────────────┼───── Batch Size
              1     8     64    512   2048     ∞

  → Batch Size 是 Decode 阶段从 Memory-Bound 转向 Compute-Bound 的关键
  → Continuous Batching / Speculative Decoding 本质上都是增大有效 Batch
```

---

## 13. Nsight Compute Roofline 诊断

### 13.1 基本用法

```bash
# 收集 Roofline 数据
ncu --set roofline -o profile ./my_app

# 收集完整数据 (含 Hierarchical Roofline)
ncu --set full -o profile ./my_app

# 只收集特定 Kernel
ncu --set roofline --kernel-name "flash_attn" -o profile ./my_app

# 命令行查看 Roofline Section
ncu --set roofline --page raw ./my_app
```

### 13.2 关键 Roofline Sections

| Section 名称 | 文件名 | 内容 |
|-------------|--------|------|
| **GPU Speed of Light Roofline Chart** | `SpeedOfLight_RooflineChart.section` | 单层 Roofline (DRAM) |
| **Hierarchical Roofline** | `SpeedOfLight_HierarchicalDoubleRooflineChart.section` | L1/L2/DRAM 三层 |
| **Roofline (all)** | `roofline` section set | 所有 Roofline 汇总 |

### 13.3 Roofline 图解读方法

```
Nsight Compute Roofline 图中:
┌─────────────────────────────────────────────────┐
│ Y 轴: GFLOP/s (对数坐标)                       │
│ X 轴: FLOP/Byte (Arithmetic Intensity, 对数坐标)│
│                                                  │
│ 蓝色区域: Memory-Bound                          │
│ 绿色区域: Compute-Bound                         │
│                                                  │
│ ★ Achieved Value: Kernel 实际性能               │
│ ─── Roofline: 理论性能上限                      │
│                                                  │
│ ★ 到 Roofline 的垂直距离 = 优化空间             │
│   距离小 → 已接近理论极限                       │
│   距离大 → 有大量优化空间                       │
└─────────────────────────────────────────────────┘
```

### 13.4 常用诊断 Metric

```bash
# 计算瓶颈诊断
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
  ./my_app
```

| Metric | 含义 | Memory-Bound | Compute-Bound | Latency-Bound |
|--------|------|:----------:|:-----------:|:------------:|
| `sm__throughput` | SM 计算吞吐 | 低 | **高** | 低 |
| `dram__throughput` | DRAM 带宽利用率 | **高** | 低 | 低 |
| `sm__pipe_tensor_op_hmma` | Tensor Core 利用率 | 低 | **高** (GEMM) | 低 |
| `sm__pipe_fma` | FMA 管线利用率 | 低 | **高** (elem) | 低 |
| `sm__warps_active` | Active Warps | 中-高 | 中-高 | **低** |
| `l1tex__throughput` | L1 吞吐 | 检查 L1 瓶颈 | 低 | 低 |

### 13.5 Baseline 对比跟踪优化进度

```bash
# 第一次 Profile (优化前)
ncu --set roofline -o baseline ./my_app

# 优化后 Profile
ncu --set roofline -o optimized ./my_app

# 在 GUI 中加载 baseline 对比
# File → Open → baseline.ncu-rep
# 右键 → Set as Baseline
# File → Open → optimized.ncu-rep
# → 两个点同时显示在 Roofline 图上, 可视化优化效果
```

---

## 14. 优化决策流程图与检查清单

### 14.1 决策流程图

```
               ┌─────────────────┐
               │  Profile Kernel  │
               │ (ncu --set full) │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ 计算 AI 并定位   │
               │ Roofline 图位置  │
               └────────┬────────┘
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────────┐
    │ AI < Ridge│ │ AI > Ridge│ │ 两者利用率   │
    │ & 带宽>60%│ │ & 计算>60%│ │ 均 < 60%    │
    │          │ │          │ │              │
    │ Memory   │ │ Compute  │ │ Latency      │
    │ Bound    │ │ Bound    │ │ Bound        │
    └────┬─────┘ └────┬─────┘ └──────┬───────┘
         │            │              │
         ▼            ▼              ▼
  ┌──────────┐ ┌──────────┐ ┌──────────────┐
  │减少访存量│ │减少指令数│ │提高并发度    │
  │• Fusion  │ │• TC 替代 │ │• ↑ Occupancy │
  │• 量化    │ │• Fast    │ │• ↑ ILP       │
  │• 向量化  │ │  Math    │ │• 异步操作    │
  │• Tiling  │ │• Unroll  │ │• 减少同步    │
  │提高带宽  │ │• Pack    │ │增大问题规模  │
  │• 合并访存│ │  half2   │ │• ↑ Batch     │
  │• L2 持久 │ │提高 ILP  │ │• Persistent  │
  │• Prefetch│ │• 多累加器│ │  Kernel      │
  │• Swizzle │ │• WarpSpec│ │• CUDA Graph  │
  └──────────┘ └──────────┘ └──────────────┘
         │            │              │
         └────────────┼──────────────┘
                      ▼
              ┌──────────────┐
              │ 重新 Profile  │
              │ 确认改善      │
              └──────────────┘
```

### 14.2 优化检查清单

#### Memory-Bound Checklist

- [ ] **访存量最小化**
  - [ ] 相邻 elementwise/norm 操作是否已 Fusion？
  - [ ] 是否可用低精度 (FP8/INT8/INT4) 减少传输量？
  - [ ] 是否使用向量化加载 (float4/uint4)？
  - [ ] 是否有冗余的 Global Memory 读写可消除？

- [ ] **带宽利用率最大化**
  - [ ] 全局内存访问是否合并 (coalesced)？
  - [ ] Shared Memory 是否存在 Bank Conflict？
  - [ ] 热数据是否适合 L2 持久化 (cudaAccessPolicyWindow)？
  - [ ] 是否利用 Prefetch 隐藏 DRAM 延迟？
  - [ ] Block 执行顺序是否优化了 L2 局部性 (Threadblock Swizzle)？

- [ ] **重计算 vs 存储**
  - [ ] 中间结果是否可以重算而非存储？(FlashAttention 模式)

#### Compute-Bound Checklist

- [ ] **计算单元最优化**
  - [ ] 矩阵运算是否使用 Tensor Core？(WMMA/mma.sync/WGMMA/tcgen05)
  - [ ] 标量运算是否使用 Fast Math Intrinsics？
  - [ ] 是否使用 Packed Arithmetic (half2/bfloat162)？

- [ ] **指令效率**
  - [ ] 热循环是否使用 `#pragma unroll`？
  - [ ] 是否有冗余计算可消除 (公共子表达式)？
  - [ ] 整数除法/取模是否可替换为位运算？

- [ ] **Pipeline 效率**
  - [ ] Tensor Core 利用率是否 > 70%？(否则检查 Pipeline 气泡)
  - [ ] 是否利用 Warp Specialization 实现计算-搬运重叠？(Hopper+)
  - [ ] 是否存在因 SMEM Bank Conflict 导致的 TC 喂不饱？

#### Latency-Bound Checklist

- [ ] **并发度**
  - [ ] Occupancy 是否足以隐藏延迟？(通常 ≥ 50%)
  - [ ] 若 Occupancy 受限，原因是寄存器还是 SMEM？
  - [ ] 是否可通过 ILP (循环展开/多累加器) 弥补低 Occupancy？

- [ ] **同步开销**
  - [ ] `__syncthreads()` 调用是否可减少？
  - [ ] Block 大小是否合理？(太小 → Grid 太大 → launch 开销; 太大 → Occupancy 下降)

- [ ] **问题规模**
  - [ ] 若 Grid < SM 数 → 考虑 Persistent Kernel 或合并 Kernel
  - [ ] 是否可通过 CUDA Graph 减少 Kernel Launch 开销？

---

> **参考来源：**
> - Williams, Waterman, Patterson. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." 2009.
> - Volkov. "Understanding Latency Hiding on GPUs." UC Berkeley, 2016.
> - Nsight Compute Profiling Guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/
> - NVIDIA Matrix Multiplication Background: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/
> - Tri Dao. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." 2022.
