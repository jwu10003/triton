# CUDA Fast Math 深度指南

> 面向 LLM 高性能 Kernel 开发的近似/快速数学函数与编译器选项参考
> 覆盖编译器 Flag、Intrinsic 函数、SFU 硬件、PTX 指令映射、精度-性能权衡

---

## 目录

1. [概述](#1-概述)
2. [编译器浮点选项](#2-编译器浮点选项)
3. [SFU 硬件架构](#3-sfu-硬件架构)
4. [标准函数 vs Intrinsic 函数](#4-标准函数-vs-intrinsic-函数)
5. [PTX 近似指令](#5-ptx-近似指令)
6. [FMA 融合与舍入控制](#6-fma-融合与舍入控制)
7. [FTZ：Flush-To-Zero 与 Denormal 处理](#7-ftzflush-to-zero-与-denormal-处理)
8. [除法与倒数近似](#8-除法与倒数近似)
9. [Half / BFloat16 数学函数](#9-half--bfloat16-数学函数)
10. [LLM 常用激活函数的 Fast Math 实现](#10-llm-常用激活函数的-fast-math-实现)
11. [Per-Kernel 精度控制策略](#11-per-kernel-精度控制策略)
12. [架构演进：SFU 瓶颈与 Blackwell Ultra](#12-架构演进sfu-瓶颈与-blackwell-ultra)
13. [诊断与验证](#13-诊断与验证)
14. [决策 Checklist](#14-决策-checklist)

---

## 1. 概述

### 1.1 Fast Math 的本质

CUDA Fast Math 是一组**以精度换取性能**的机制，包括：

```
Fast Math 机制
├── 编译器全局选项 ──── --use_fast_math (一键开启所有快速选项)
│   ├── --ftz=true           ─ 将 denormal 值刷为零
│   ├── --prec-div=false     ─ 近似除法 (非 IEEE 精确)
│   ├── --prec-sqrt=false    ─ 近似平方根 (非 IEEE 精确)
│   └── --fmad=true          ─ 启用 FMA 指令融合 (默认已开启)
├── 函数级 Intrinsic ── __sinf, __cosf, __expf, __logf, ... (手动调用)
├── PTX 近似指令 ────── sin.approx, ex2.approx, rcp.approx, ... (SFU 执行)
└── 舍入模式控制 ────── __fadd_rn, __fmul_rz, __fmaf_rd, ... (IEEE 精确但指定舍入)
```

### 1.2 三层数学函数体系

CUDA 数学函数分为三层，精度递减、性能递增：

| 层级 | 示例 | 精度 | 实现方式 | 使用场景 |
|------|------|------|---------|---------|
| **标准库函数** | `sinf(x)`, `expf(x)`, `logf(x)` | ≤2 ULP (IEEE 级) | 多条指令 (range reduction + 多项式) | 默认精度 |
| **Intrinsic 函数** | `__sinf(x)`, `__expf(x)`, `__logf(x)` | 2–8 ULP (近似) | 少量指令 (直接 SFU) | 精度不敏感的热点 |
| **舍入模式 Intrinsic** | `__fadd_rn(x,y)`, `__fmaf_rn(x,y,z)` | 0 ULP (IEEE 精确) | 单条指令 + 指定舍入 | 需要精确控制舍入 |

### 1.3 为什么 LLM Kernel 关注 Fast Math

| LLM 操作 | 涉及的数学函数 | Fast Math 可接受性 |
|----------|--------------|------------------|
| Softmax | `exp()`, `1/sum` | **高** — 相对值不变，绝对误差可容忍 |
| GELU 激活 | `tanh()`, `exp()` 或 `erf()` | **高** — 训练中梯度不敏感 |
| SiLU (Swish) | `sigmoid()` → `exp()` | **高** — 同 GELU |
| RMSNorm | `rsqrt()` | **中** — 已是单指令，误差很小 |
| RoPE 位置编码 | `sin()`, `cos()` | **中** — 小误差可能影响长序列精度 |
| Loss 计算 | `log()`, `exp()` | **低** — 梯度累积误差可能导致不收敛 |
| 精确 Attention Score | `exp()` 用于概率 | **中** — 需要验证 top-k 顺序不变 |

---

## 2. 编译器浮点选项

### 2.1 `--use_fast_math` 分解

`--use_fast_math` 是一个**组合开关**，等价于同时设置以下四个选项，并且将所有标准数学函数替换为对应的 intrinsic 版本：

```
--use_fast_math
   ≡ --ftz=true          ← 1. 刷新 denormal 为零
   + --prec-div=false     ← 2. 快速除法 (非 IEEE 精确)
   + --prec-sqrt=false    ← 3. 快速平方根 (非 IEEE 精确)
   + --fmad=true          ← 4. 启用 FMA 融合 (默认已开启)
   + (将 sinf→__sinf, cosf→__cosf, expf→__expf, logf→__logf, ...)
```

### 2.2 各选项详解

| 选项 | 默认值 | `--use_fast_math` | 影响 |
|------|--------|-------------------|------|
| `--ftz` | `false` | `true` | FP32 denormal 刷零，省去 denormal 慢路径 |
| `--prec-div` | `true` | `false` | 除法精度从 0.5 ULP (IEEE) 降为 ~2 ULP |
| `--prec-sqrt` | `true` | `false` | 开方精度从 0.5 ULP (IEEE) 降为 ~2 ULP |
| `--fmad` | `true` | `true` | 允许 `a*b+c` 融合为 FMA (默认就是开的) |

**关键细节：**

- `--fmad=true` 是**默认行为**，因此只有前三项和函数替换才是 `--use_fast_math` 的实质变化。
- Intrinsic 函数 (`__sinf` 等) 的行为**不受** `--prec-div` / `--prec-sqrt` 影响——它们有自己固定的精度特性。
- `--ftz=true` **是唯一影响 intrinsic 函数的编译器选项**，它改变 denormal 输入/输出的处理方式。

### 2.3 作用域：文件级而非 Kernel 级

```bash
# 整个编译单元都启用 fast math
nvcc --use_fast_math kernel.cu

# 仅针对特定子选项
nvcc --ftz=true --prec-div=false kernel.cu

# 保守默认 (最大 IEEE 兼容)
nvcc --ftz=false --prec-div=true --prec-sqrt=true kernel.cu
```

> **注意：** CUDA **没有** per-kernel 的 `#pragma fast_math`。要实现 per-kernel 精度控制，需要使用 §11 中的策略。

### 2.4 `-O` 优化级别与 Fast Math 的关系

| 优化级别 | Fast Math | 说明 |
|---------|-----------|------|
| `-O0` | 不启用 | 无优化 |
| `-O1` / `-O2` / `-O3` | **不启用** | 更高优化但不改变浮点语义 |
| `--use_fast_math` | 显式启用 | 独立于 -O 级别 |

`-O3` **不会自动启用** fast math。优化级别只影响寄存器分配、循环优化等，不改变浮点精度语义。

---

## 3. SFU 硬件架构

### 3.1 SFU 概述

Special Function Unit (SFU) 是 SM 中执行**超越函数**的专用硬件单元。它通过硬件查表 + 线性插值实现 ~2^-23 (单精度约 1 ULP) 的相对精度。

```
SM Sub-partition (1/4 SM)
├── FP32 CUDA Cores × 16      ← 算术主力 (Ampere+ 可含第二组)
├── INT32 Units × 16
├── Tensor Core × 1
├── Load/Store Units
└── SFU × 4                   ← 每 Sub-partition 4 个 SFU
     ↑
     └── 执行: sin, cos, rcp, rsqrt, ex2, lg2, tanh (Turing+)
```

### 3.2 SFU 吞吐量演进

| 架构 | SFU/SM | 结果数/周期/SM | FP32 Core 结果数/周期/SM | SFU:FP32 比例 |
|------|--------|--------------|------------------------|--------------|
| **Volta** (7.0) | 16 | 16 | 64 | 1:4 |
| **Turing** (7.5) | 16 | 16 | 64 | 1:4 |
| **Ampere** (8.0) | 16 | 16 | 64 | 1:4 |
| **Ampere** (8.6) | 16 | 16 | 128 | 1:8 |
| **Ada** (8.9) | 16 | 16 | 128 | 1:8 |
| **Hopper** (9.0) | 16 | 16 | 128 | 1:8 |
| **Blackwell** (10.0) | 16 | 16 | 128 | 1:8 |
| **Blackwell Ultra** (12.0) | 32 | **32** | 128 | **1:4** |

**关键趋势：** FP32 Core 吞吐从 Ampere 起翻倍 (FP32 专用 + FP32/INT32 双用)，但 SFU 数量保持 16/SM 不变长达 5 代——形成**非对称瓶颈**。

### 3.3 SFU 支持的操作

SFU 硬件原生支持以下操作 (在 SASS 层面统一为 `MUFU.*` 指令)：

| SASS 指令 | PTX 指令 | 数学操作 | 备注 |
|-----------|---------|---------|------|
| `MUFU.RCP` | `rcp.approx.f32` | 1/x | 倒数 |
| `MUFU.RSQ` | `rsqrt.approx.f32` | 1/√x | 倒数平方根 |
| `MUFU.SIN` | `sin.approx.f32` | sin(x·2π) | 输入单位: 周期 |
| `MUFU.COS` | `cos.approx.f32` | cos(x·2π) | 输入单位: 周期 |
| `MUFU.EX2` | `ex2.approx.f32` | 2^x | 以 2 为底指数 |
| `MUFU.LG2` | `lg2.approx.f32` | log₂(x) | 以 2 为底对数 |
| `MUFU.SQRT` | `sqrt.approx.f32` | √x | 平方根 |
| `MUFU.TANH` | `tanh.approx.f32` | tanh(x) | Turing+ 支持 |

> **注意：** SFU 原生函数使用**以 2 为底**的指数和对数，以及**以周期为单位**的三角函数。C 语言标准的 `exp()` (以 e 为底) 和 `sin()` (以弧度为单位) 需要额外的乘法做单位转换。

### 3.4 SFU 延迟分析

从 PTX 到 SASS 的映射通常不是单条指令——需要额外的 range reduction 和单位转换：

```
__expf(x) 的 PTX/SASS 展开:
  mul.f32     %f1, %f0, 0f3FB8AA3B;   // x × log₂(e) ≈ x × 1.4427
  ex2.approx.f32 %f2, %f1;            // 2^(x·log₂e) = e^x
  → SASS: FMUL + MUFU.EX2
  → 总延迟: ~18 cycles (Ampere 实测)

__sinf(x) 的展开:
  mul.f32     %f1, %f0, 0f3E22F983;   // x × (1/2π), 转为周期单位
  sin.approx.f32 %f2, %f1;            // sin(x/(2π) · 2π) = sin(x)
  → SASS: FSETP + FMUL + MUFU.SIN
  → 总延迟: ~18 cycles (Ampere 实测)

ex2.approx.f16 (FP16 版):
  → SASS: MUFU.EX2.F16
  → 总延迟: ~6 cycles (无需单位转换，直接 FP16 SFU)
```

### 3.5 标准函数 vs Intrinsic 的指令数对比

| 函数 | 标准版 PTX 指令数 | Intrinsic PTX 指令数 | 加速比 |
|------|------------------|--------------------|----|
| `sinf(x)` | ~40+ (range reduction + Cody-Waite + polynomial) | 2-3 (mul + sin.approx) | 5–8× |
| `cosf(x)` | ~40+ | 2-3 | 5–8× |
| `expf(x)` | ~20+ | 2 (mul + ex2.approx) | 3–5× |
| `logf(x)` | ~15+ | 2 (lg2.approx + mul) | 3–5× |
| `powf(x,y)` | ~200+ (完整实现) | 3-4 (lg2 + mul + ex2) | 10–50× |

> `powf` 的加速最显著：标准版需要处理所有 edge case (负底数、零、NaN 等)，intrinsic 版 `__powf(x,y) = 2^(y·log₂x)` 只有几条指令，但对负底数返回 NaN。

---

## 4. 标准函数 vs Intrinsic 函数

### 4.1 单精度标准函数 ULP 误差

标准数学函数在所有 IEEE 754 特殊值 (NaN, Inf, denormal, ±0) 上行为正确，精度接近 IEEE 要求：

| 标准函数 | 最大 ULP 误差 | 备注 |
|----------|-------------|------|
| `sinf(x)` | 2 | 全范围 |
| `cosf(x)` | 2 | 全范围 |
| `tanf(x)` | 4 | 全范围 |
| `sincosf(x, &s, &c)` | 2 (sin), 2 (cos) | 同时计算 sin 和 cos |
| `expf(x)` | 2 | 全范围 |
| `exp2f(x)` | 1 | 全范围 |
| `exp10f(x)` | 2 | 全范围 |
| `logf(x)` | 1 | 全范围 |
| `log2f(x)` | 1 | 全范围 |
| `log10f(x)` | 2 | 全范围 |
| `powf(x, y)` | 8 | 全范围 |
| `sqrtf(x)` | 0 | **IEEE 精确** (correctly rounded) |
| `rsqrtf(x)` | 2 | 全范围 |
| `cbrtf(x)` | 1 | 全范围 |
| `erff(x)` | 4 | 全范围 |
| `tanhf(x)` | 2 | 全范围 |

### 4.2 单精度 Intrinsic 函数 ULP 误差

Intrinsic 函数精度更低，但映射到更少的 native 指令：

| Intrinsic 函数 | 最大 ULP 误差 | 有效输入范围 | 备注 |
|---------------|-------------|------------|------|
| `__sinf(x)` | 2 | \|x\| < 48039.0f | 超出范围误差无界 |
| `__cosf(x)` | 2 | \|x\| < 48039.0f | 超出范围误差无界 |
| `__sincosf(x, &s, &c)` | 2 | \|x\| < 48039.0f | 同上 |
| `__tanf(x)` | 4 | \|x\| < 48039.0f | 内部为 `__sinf/__cosf` 相除 |
| `__expf(x)` | 2 + floor(abs(1.16·x)) | 全范围 | 大 \|x\| 误差增长 |
| `__exp10f(x)` | 2 + floor(abs(2.95·x)) | 全范围 | 大 \|x\| 误差增长 |
| `__logf(x)` | 1 | 全范围 | denormal 输入刷零 |
| `__log2f(x)` | 1 | 全范围 | denormal 输入刷零 |
| `__log10f(x)` | 2 | 全范围 | denormal 输入刷零 |
| `__powf(x, y)` | 8 | x > 0 | 负底数返回 **NaN** |
| `__fdividef(x, y)` | 2 | 2^-126 ≤ \|y\| ≤ 2^126 | 超出 y 范围返回 0 或 NaN |

### 4.3 关键差异总结

```
                        标准函数                          Intrinsic 函数
──────────────────────────────────────────────────────────────────────────
精度                    ≤2 ULP (多数)                     2–8 ULP (可能随输入增大)
特殊值处理              IEEE 754 完全兼容                  简化处理 (可能返回 NaN/0)
指令数                  10–200+ PTX 指令                  2–4 PTX 指令
受 --ftz 影响           否 (保留 denormal)                 是 (ftz=true 时 denormal 刷零)
受 --prec-div 影响      是                                否 (自有精度)
受 --use_fast_math 影响  被替换为 intrinsic                行为不变 (本身就是 intrinsic)
__powf 负底数           正确处理                          返回 NaN
```

### 4.4 `--use_fast_math` 的函数替换列表

以下标准函数会被替换为对应 intrinsic (来自 CUDA Programming Guide Table 9)：

| 标准函数 | 替换为 | 标准函数 | 替换为 |
|----------|-------|----------|-------|
| `sinf` | `__sinf` | `logf` | `__logf` |
| `cosf` | `__cosf` | `log2f` | `__log2f` |
| `tanf` | `__tanf` | `log10f` | `__log10f` |
| `sincosf` | `__sincosf` | `powf` | `__powf` |
| `expf` | `__expf` | `x / y` | `__fdividef(x, y)` |
| `exp10f` | `__exp10f` | `sqrtf(x)` | → prec-sqrt 影响 |

> **不被替换的函数：** `erff`, `tanhf`, `cbrtf`, `acosf`, `asinf`, `atanf`, `atan2f` 等不在替换列表中——即使开启 `--use_fast_math`，这些函数仍使用标准版实现。

---

## 5. PTX 近似指令

### 5.1 PTX `.approx` 指令精度规格

PTX ISA 定义的近似指令直接在 SFU 上执行：

| PTX 指令 | 数学操作 | 精度 | SASS 映射 |
|---------|---------|------|----------|
| `rcp.approx.f32` | 1/x | ~2^-23 相对误差 (≈1 ULP) | `MUFU.RCP` |
| `rcp.approx.ftz.f32` | 1/x (FTZ) | ~2^-23 | `MUFU.RCP` |
| `rsqrt.approx.f32` | 1/√x | ~2^-23 (≈1 ULP) | `MUFU.RSQ` |
| `sin.approx.f32` | sin(x·2π) | ~2^-21 (≈4 ULP) | `MUFU.SIN` |
| `cos.approx.f32` | cos(x·2π) | ~2^-21 (≈4 ULP) | `MUFU.COS` |
| `ex2.approx.f32` | 2^x | ~2^-23 (≈1 ULP) | `MUFU.EX2` |
| `lg2.approx.f32` | log₂(x) | ~2^-23 (≈1 ULP) | `MUFU.LG2` |
| `sqrt.approx.f32` | √x | ~2^-23 | `MUFU.SQRT` |
| `tanh.approx.f32` | tanh(x) | ~2^-23 | `MUFU.TANH` (Turing+) |

### 5.2 `.approx` vs `.rn` (IEEE 精确) 对比

| 方面 | `.approx` 版 | `.rn` 版 |
|------|------------|---------|
| 执行单元 | SFU | FP32 Core (多次迭代) |
| 延迟 | 1 MUFU 周期/warp | ~8 周期/warp |
| 吞吐 | 16 结果/周期/SM | 64 结果/周期/SM (但多周期) |
| 精度 | ~2^-21 到 ~2^-23 | 0.5 ULP (IEEE) |

### 5.3 PTX 中如何实现常见数学函数

```ptx
// ═══════════════════════════════════════════════════════
// exp(x) = 2^(x · log₂(e))
// ═══════════════════════════════════════════════════════
mul.f32         %f1, %f0, 0f3FB8AA3B;   // x × log₂(e), 常数 ≈ 1.44269504
ex2.approx.f32  %f2, %f1;               // 2^(x·log₂e) = e^x

// ═══════════════════════════════════════════════════════
// log(x) = log₂(x) / log₂(e) = log₂(x) · ln(2)
// ═══════════════════════════════════════════════════════
lg2.approx.f32  %f1, %f0;               // log₂(x)
mul.f32         %f2, %f1, 0f3F317218;   // × ln(2) ≈ 0.693147

// ═══════════════════════════════════════════════════════
// pow(x, y) = 2^(y · log₂(x))
// ═══════════════════════════════════════════════════════
lg2.approx.f32  %f1, %f0;               // log₂(x)  ← x > 0 required!
mul.f32         %f2, %f1, %fy;           // y · log₂(x)
ex2.approx.f32  %f3, %f2;               // 2^(y·log₂x) = x^y

// ═══════════════════════════════════════════════════════
// sin(x) — 输入为弧度，需转换为周期
// ═══════════════════════════════════════════════════════
mul.f32         %f1, %f0, 0f3E22F983;   // x × (1/2π) ≈ x × 0.15915494
sin.approx.f32  %f2, %f1;               // sin(x/(2π) · 2π) = sin(x)

// ═══════════════════════════════════════════════════════
// Softmax: exp(x - max) / sum
// ═══════════════════════════════════════════════════════
sub.f32         %f_diff, %f_elem, %f_max;
mul.f32         %f_scaled, %f_diff, 0f3FB8AA3B;  // (x - max) × log₂(e)
ex2.approx.f32  %f_exp, %f_scaled;                // exp(x - max)
// ... 累加 sum ...
rcp.approx.f32  %f_inv, %f_sum;                   // 1/sum
mul.f32         %f_out, %f_exp, %f_inv;            // exp(x-max) / sum
```

---

## 6. FMA 融合与舍入控制

### 6.1 FMA 融合 (`--fmad`)

**Fused Multiply-Add (FMA)** 将 `a*b + c` 在单条指令中完成，只做**一次**舍入 (而非分开的两次舍入)：

```
非融合:   round(round(a × b) + c)    ← 两次舍入，可能损失精度
FMA融合:  round(a × b + c)           ← 一次舍入，结果更精确
```

| 行为 | `--fmad=true` (默认) | `--fmad=false` |
|------|---------------------|----------------|
| `a*b + c` | 编译为 **FFMA** (一次舍入) | 编译为 **FMUL + FADD** (两次舍入) |
| 精度 | 通常**更高** (减少中间舍入) | 严格匹配 CPU 标量行为 |
| 性能 | 1 条指令 | 2 条指令 (~30% 慢) |
| 结果可复现性 | 可能与 CPU 不一致 | 与 CPU 行为一致 |

### 6.2 显式 FMA 函数

```cuda
// 显式 FMA — 不受 --fmad 选项影响，始终执行 FMA
float r = fmaf(a, b, c);           // 标准库 FMA (0 ULP, IEEE 精确)
float r = __fmaf_rn(a, b, c);      // intrinsic FMA, round-to-nearest

// 显式 非融合 — 阻止编译器融合为 FMA
float r = __fadd_rn(__fmul_rn(a, b), c);  // 强制分开: MUL + ADD
```

### 6.3 舍入模式 Intrinsic (IEEE 精确, 0 ULP)

这些 intrinsic 精度为 0 ULP (correctly rounded)，主要用途是**选择舍入模式**和**阻止 FMA 融合**：

| Intrinsic | 操作 | 舍入模式 |
|-----------|------|---------|
| `__fadd_rn(x,y)` | x + y | round-to-nearest (ties to even) |
| `__fadd_rz(x,y)` | x + y | round-towards-zero |
| `__fadd_ru(x,y)` | x + y | round-towards-+∞ |
| `__fadd_rd(x,y)` | x + y | round-towards-−∞ |
| `__fmul_rn(x,y)` | x × y | round-to-nearest |
| `__fmul_rz(x,y)` | x × y | round-towards-zero |
| `__fdiv_rn(x,y)` | x / y | round-to-nearest (IEEE 精确) |
| `__fsqrt_rn(x)` | √x | round-to-nearest (IEEE 精确) |
| `__fmaf_rn(x,y,z)` | x×y+z | round-to-nearest (FMA, 单次舍入) |

**双精度版本：** `__dadd_rn`, `__dmul_rn`, `__ddiv_rn`, `__dsqrt_rn`, `__fma_rn` 等，语义完全对应。

### 6.4 `__fmaf_ieee_rn` — 忽略 FTZ 的 FMA

```cuda
// 即使编译时 --ftz=true，此函数仍保留 denormal 处理
float r = __fmaf_ieee_rn(a, b, c);   // FMA 且不刷 denormal
```

用途：在 `--use_fast_math` 编译的文件中，需要个别操作精确处理 denormal 时。

---

## 7. FTZ：Flush-To-Zero 与 Denormal 处理

### 7.1 什么是 Denormal (Subnormal)

FP32 中，denormal 是指数部分为全零、尾数非零的浮点数：

```
Normal FP32:    1.mmm...m × 2^(e-127),   1 ≤ e ≤ 254
Denormal FP32:  0.mmm...m × 2^(-126),    e = 0, m ≠ 0

最小 normal:    ±1.0 × 2^(-126) ≈ ±1.175e-38
最大 denormal:  ±0.999... × 2^(-126) ≈ ±1.175e-38
最小 denormal:  ±2^(-149) ≈ ±1.401e-45
```

Denormal 数实现了**渐进下溢 (gradual underflow)**，避免微小值突然跳变为零。

### 7.2 FTZ 的行为

| 场景 | `--ftz=false` (默认) | `--ftz=true` |
|------|--------------------|----|
| 输入为 denormal | 正常参与计算 | 视为 ±0.0 |
| 结果为 denormal | 保留 denormal 值 | 刷为 ±0.0 (保留符号) |
| 性能 | 可能走慢路径 (部分操作) | 无慢路径 |

### 7.3 性能影响

在 NVIDIA GPU 上，denormal 的性能影响**因操作而异**：

| 操作 | Denormal 处理方式 | 性能影响 |
|------|-----------------|---------|
| FMA (`fmaf`) | **硬件处理** (CC ≥ 2.0) | 无 (全速) |
| ADD / MUL / SUB | **硬件处理** (CC ≥ 2.0) | 无 (全速) |
| RCP / RSQRT / SQRT | 可能走**微码慢路径** | ~10-20% 慢 |
| SFU 函数 (sin/cos/ex2) | 通常刷零 | 无 |
| Atomic FP32 (Global) | **始终 FTZ**，无论编译选项 | — |
| Atomic FP32 (Shared) | **始终保留 denormal** | — |

**实测案例：** Tesla K20c 上 N-body 模拟开启 `--ftz=true` 后加速 ~20%。

### 7.4 FTZ 对 LLM Kernel 的影响

| 场景 | 是否安全开启 FTZ | 原因 |
|------|-----------------|------|
| GEMM 累加器 (FP32) | **安全** | 中间值远离 denormal 范围 |
| Softmax exp() | **安全** | exp(x-max) ≥ 0, 极小值本该舍去 |
| LayerNorm rsqrt() | **通常安全** | 方差极少为 denormal |
| 损失函数 log(p) | **需要注意** | p→0 时 log(p) 输入可能 denormal |
| 梯度累积 | **需要注意** | 小梯度可能进入 denormal 范围 |

### 7.5 只影响 FP32

`--ftz` **仅影响单精度 (FP32)**。FP64 的 denormal 始终按 IEEE 754 处理，不受此选项影响。FP16 / BF16 / FP8 有独立的 denormal 行为，不受 `--ftz` 控制。

---

## 8. 除法与倒数近似

### 8.1 除法的三种精度级别

```
                    精度递增 →
                    性能递减 →
┌─────────────────┬─────────────────────┬──────────────────────────┐
│  __fdividef(x,y) │  x / y              │  __fdiv_rn(x, y)         │
│  --prec-div=false │  (默认 --prec-div=true) │  intrinsic 指定舍入    │
│                  │                      │                          │
│  ~2 ULP          │  0.5 ULP (IEEE)      │  0 ULP (correctly rounded)│
│  1 MUFU + 1 FMUL │  ~8 cycles (Newton-  │  同左但指定舍入模式       │
│  (最快)          │   Raphson 迭代)      │                          │
│                  │                      │                          │
│  y∈[2^-126,2^126]│  全范围 IEEE 兼容    │  全范围 IEEE 兼容         │
│  否则返回 0/NaN  │                      │                          │
└─────────────────┴─────────────────────┴──────────────────────────┘
```

### 8.2 `__fdividef` 的 2^126 陷阱

`__fdividef(x, y)` 内部使用 SFU 计算 `rcp.approx(y)` 然后乘以 x。当 |y| > 2^126 时，倒数结果下溢为 denormal 或零：

```cuda
// 安全: |y| 在 [2^-126, 2^126] 范围内
float r = __fdividef(x, 100.0f);     // OK, 精度 ~2 ULP

// 危险: |y| > 2^126
float r = __fdividef(x, 1e38f);      // 返回 0!
float r = __fdividef(INFINITY, 1e38f); // 返回 NaN!

// 正确处理大 y:
float r = x / y;                      // 使用标准除法
```

### 8.3 `--prec-div=false` 的行为

当 `--prec-div=false` (或 `--use_fast_math`) 时，编译器将普通除法 `x / y` 也编译为近似版本，效果等同于 `__fdividef`：

```cuda
float a = x / y;  // --prec-div=true:  IEEE 精确, 0.5 ULP
                   // --prec-div=false: 近似, ~2 ULP, 有 2^126 限制
```

### 8.4 倒数 (`rcp`) 的使用模式

```cuda
// 计算 1/x 的三种方式
float r1 = 1.0f / x;              // 标准除法 (IEEE)
float r2 = __frcp_rn(x);          // intrinsic, IEEE 精确, 0 ULP, 指定舍入
                                   // (注意: 这不是 SFU 近似版!)

// PTX 层面:
// rcp.rn.f32    → IEEE 精确倒数 (~8 cycles)
// rcp.approx.f32 → SFU 近似倒数 (1 cycle/warp, ~2^-23 误差)
```

---

## 9. Half / BFloat16 数学函数

### 9.1 FP16 标量数学函数

CUDA 提供 `cuda_fp16.h` 中的 half 精度标量数学函数，使用 `h` 前缀命名：

| 函数 | 说明 | 内部实现 |
|------|------|---------|
| `hsin(x)` | sin(x) | 提升为 float → `sinf()` → 截回 half |
| `hcos(x)` | cos(x) | 提升为 float → `cosf()` → 截回 half |
| `hexp(x)` | e^x | 提升为 float → `expf()` → 截回 half |
| `hexp2(x)` | 2^x | 提升为 float → `exp2f()` → 截回 half |
| `hexp10(x)` | 10^x | 提升为 float → `exp10f()` → 截回 half |
| `hlog(x)` | ln(x) | 提升为 float → `logf()` → 截回 half |
| `hlog2(x)` | log₂(x) | 提升为 float → `log2f()` → 截回 half |
| `hlog10(x)` | log₁₀(x) | 提升为 float → `log10f()` → 截回 half |
| `hsqrt(x)` | √x | 提升为 float → `sqrtf()` → 截回 half |
| `hrsqrt(x)` | 1/√x | 提升为 float → `rsqrtf()` → 截回 half |
| `hrcp(x)` | 1/x | 倒数 |

### 9.2 FP16 向量数学函数 (half2)

同时处理两个 half 值，使用 `h2` 前缀：

| 函数 | 说明 |
|------|------|
| `h2sin(v)` | 向量 sin |
| `h2cos(v)` | 向量 cos |
| `h2exp(v)` | 向量 exp |
| `h2exp2(v)` | 向量 2^x |
| `h2exp10(v)` | 向量 10^x |
| `h2log(v)` | 向量 ln |
| `h2log2(v)` | 向量 log₂ |
| `h2log10(v)` | 向量 log₁₀ |
| `h2sqrt(v)` | 向量 √x |
| `h2rsqrt(v)` | 向量 1/√x |
| `h2rcp(v)` | 向量 1/x |

### 9.3 FP16 数学函数与 Fast Math 的交互

**重要：** FP16 数学函数内部提升为 float 后调用对应的 float 函数。因此 `--use_fast_math` 会影响 FP16 数学函数的精度——因为内部调用的 `sinf()` 被替换为 `__sinf()`。

```cuda
// 不使用 --use_fast_math:
hsin(x) → 内部: sinf((float)x) → 精确 sinf → 截回 half

// 使用 --use_fast_math:
hsin(x) → 内部: __sinf((float)x) → 近似 __sinf → 截回 half
                                    ↑ 精度下降!
```

### 9.4 BFloat16 数学函数

`cuda_bf16.h` 提供类似的 BF16 数学函数，命名规则相同 (`hsin`/`h2sin` 等但操作 `__nv_bfloat16` 类型)，行为也相同——内部提升为 float 再截回 BF16。

### 9.5 PTX FP16 SFU 指令

从 SM 7.5+ 开始，PTX 支持 FP16 近似指令，**直接在 SFU 执行**而不提升为 FP32：

```ptx
// FP16 SFU 指令 (SM 7.5+) — 直接执行，无 FP32 提升
ex2.approx.f16      %h1, %h0;       // 2^x (FP16)
ex2.approx.f16x2    %hh1, %hh0;     // 同时计算两个 2^x (FP16x2)
// → SASS: MUFU.EX2.F16
// → 延迟: ~6 cycles (vs FP32 的 ~18 cycles)
```

这些指令在性能关键路径上比 `hexp()` (提升为 FP32) 更快。

---

## 10. LLM 常用激活函数的 Fast Math 实现

### 10.1 Softmax

Softmax 是 LLM 中最频繁使用 `exp()` 的操作：

```cuda
// ═══════════════════════════════════════════════════════
// Softmax: 数值稳定版本 + Fast Math
// ═══════════════════════════════════════════════════════
__global__ void softmax_fast(const float* input, float* output, int N) {
    // Step 1: 找最大值 (无需 fast math)
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, input[i]);
    }
    max_val = warpReduceMax(max_val);  // warp 级归约

    // Step 2: exp(x - max) 求和 — 使用 __expf
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += __expf(input[i] - max_val);   // ← Fast math exp
    }
    sum = warpReduceSum(sum);

    // Step 3: 归一化 — 使用 __fdividef 或 rcp
    float inv_sum = __frcp_rn(sum);  // 或: 1.0f / sum
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        output[i] = __expf(input[i] - max_val) * inv_sum;
    }
}
```

**精度分析：** `__expf` 在 Softmax 中的误差影响很小——因为 softmax 输出是**相对概率**，所有值除以同一个 sum。2 ULP 的相对误差在归一化后基本不可见。

### 10.2 GELU 激活函数

GELU 有两种实现——精确版 (erf) 和 tanh 近似：

```cuda
// ═══════════════════════════════════════════════════════
// GELU 精确版: x · 0.5 · (1 + erf(x / √2))
// erff 没有 intrinsic 版本，不受 --use_fast_math 替换
// ═══════════════════════════════════════════════════════
__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));  // 1/√2
}

// ═══════════════════════════════════════════════════════
// GELU tanh 近似: x · 0.5 · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
// PyTorch 中 approximate='tanh' 使用此公式
// ═══════════════════════════════════════════════════════
__device__ __forceinline__ float gelu_tanh(float x) {
    const float c = 0.7978845608f;  // √(2/π)
    const float k = 0.044715f;
    float inner = c * (x + k * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// ═══════════════════════════════════════════════════════
// GELU fast: 使用 __expf 实现 tanh 近似
// tanh(x) = 1 - 2/(1 + exp(2x))   或  (exp(2x)-1)/(exp(2x)+1)
// ═══════════════════════════════════════════════════════
__device__ __forceinline__ float gelu_fast(float x) {
    const float c = 0.7978845608f;
    const float k = 0.044715f;
    float inner = c * (x + k * x * x * x);
    float e = __expf(2.0f * inner);   // ← Fast math
    float t = (e - 1.0f) / (e + 1.0f);  // tanh
    return 0.5f * x * (1.0f + t);
}
```

> **注意：** Turing+ 架构的 SFU 有 `MUFU.TANH` 硬件指令 (~6 cycles)，`tanhf()` 会自动利用——这比手写 `__expf` 实现 tanh 可能更快。

### 10.3 SiLU (Swish) 激活函数

```cuda
// SiLU(x) = x · sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float silu_fast(float x) {
    float e = __expf(-x);                // Fast exp
    return x / (1.0f + e);               // 标准除法 (安全)
    // 或: return x * __frcp_rn(1.0f + e); // 用倒数替代除法
}

// half2 版本 — 向量化
__device__ __forceinline__ half2 silu_fast_h2(half2 x) {
    half2 neg_x = __hneg2(x);
    half2 e = h2exp(neg_x);              // exp(-x)
    half2 one = __float2half2_rn(1.0f);
    half2 denom = __hadd2(one, e);
    return __h2div(x, denom);            // x / (1 + exp(-x))
}
```

### 10.4 RMSNorm

```cuda
// RMSNorm: x_i / sqrt(mean(x²) + eps)
// rsqrtf 已经是单指令 (MUFU.RSQ)，无需额外 fast math
__device__ __forceinline__ void rmsnorm_fast(
    float* output, const float* input, const float* weight,
    float inv_rms,  // 预计算的 1/sqrt(mean(x²)+eps)
    int hidden_dim
) {
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

// inv_rms 的计算:
float sum_sq = /* warp reduction of x² */;
float inv_rms = rsqrtf(sum_sq / hidden_dim + eps);
// rsqrtf 编译为 MUFU.RSQ (1 SFU 指令)，精度 ~2 ULP
// 对 RMSNorm 完全够用
```

### 10.5 RoPE 位置编码

```cuda
// RoPE: 需要 sin/cos，可使用 __sincosf 同时计算
__device__ __forceinline__ void rope_fast(
    float& x0, float& x1,  // 相邻的两个特征维度
    float freq              // θ_i = 10000^(-2i/d) · pos
) {
    float s, c;
    __sincosf(freq, &s, &c);   // Fast sin + cos 同时计算
    float new_x0 = x0 * c - x1 * s;
    float new_x1 = x0 * s + x1 * c;
    x0 = new_x0;
    x1 = new_x1;
}
// 注意: __sincosf 在 |freq| < 48039.0f 时精度 2 ULP
// 对于 LLM 的位置编码，freq 值通常远小于此阈值
```

### 10.6 FlashAttention 中的 Softmax

FlashAttention 使用 **online softmax**，在流式处理 KV 块时需要反复调用 `exp()` 和缩放修正：

```cuda
// FlashAttention online softmax 核心循环
float m_new = max(m_prev, row_max_current);    // 更新最大值
float correction = __expf(m_prev - m_new);      // ← Fast exp 修正因子
float exp_val = __expf(score - m_new);           // ← Fast exp 新 exp 值
l_new = correction * l_prev + exp_val;           // 更新归一化因子

// 修正之前的 O 向量
O = O * correction + exp_val * V_current;
```

> **Flash Attention 4 创新：** 在 Blackwell 上 SFU 成为瓶颈后，FA4 引入了**CUDA Core 多项式近似 exp**来替代部分 SFU 调用——在 FMA 单元上用三次多项式逼近 2^x，与 SFU `MUFU.EX2` 混合使用，减轻 SFU 压力。

---

## 11. Per-Kernel 精度控制策略

### 11.1 问题：`--use_fast_math` 是文件级的

CUDA 没有 per-kernel 的 `#pragma fast_math`。如果某些 kernel 需要精确数学，而另一些需要快速数学，必须使用以下策略。

### 11.2 策略 1：手动调用 Intrinsic (推荐)

不使用 `--use_fast_math`，而是在需要的地方手动调用 intrinsic 函数：

```cuda
// 精度敏感的 kernel — 使用标准函数
__global__ void loss_kernel(const float* logits, float* loss) {
    float p = expf(logits[i] - max_val);   // 标准精确 exp
    loss[i] = -logf(p / sum);               // 标准精确 log
}

// 性能敏感的 kernel — 手动使用 intrinsic
__global__ void softmax_kernel(const float* input, float* output) {
    float e = __expf(input[i] - max_val);   // 快速近似 exp
    output[i] = e * __frcp_rn(sum);         // 快速倒数
}
```

**优点：** 最细粒度控制，无需拆分编译单元。
**缺点：** 需要手动替换每个函数调用。

### 11.3 策略 2：分离编译单元

将不同精度需求的 kernel 放入不同的 `.cu` 文件：

```bash
# fast_kernels.cu — 包含 softmax, activation 等
nvcc --use_fast_math -c fast_kernels.cu -o fast_kernels.o

# precise_kernels.cu — 包含 loss, gradient 等
nvcc -c precise_kernels.cu -o precise_kernels.o

# 链接
nvcc fast_kernels.o precise_kernels.o -o program

# 使用 Device LTO 恢复跨文件优化性能
nvcc -dlto --use_fast_math -c fast_kernels.cu -o fast_kernels.o
nvcc -dlto -c precise_kernels.cu -o precise_kernels.o
nvcc -dlto fast_kernels.o precise_kernels.o -o program
```

**优点：** 整个文件的函数都自动替换。
**缺点：** 分离编译可能损失跨文件优化；需要 `-dlto` 恢复。

### 11.4 策略 3：`__forceinline__` 封装

将 fast math 操作封装为 `__device__ __forceinline__` 函数，在不同 kernel 中选择性使用：

```cuda
// math_helpers.cuh
__device__ __forceinline__ float fast_exp(float x) { return __expf(x); }
__device__ __forceinline__ float safe_exp(float x) { return expf(x); }

__device__ __forceinline__ float fast_log(float x) { return __logf(x); }
__device__ __forceinline__ float safe_log(float x) { return logf(x); }

__device__ __forceinline__ void fast_sincos(float x, float* s, float* c) {
    __sincosf(x, s, c);
}
__device__ __forceinline__ void safe_sincos(float x, float* s, float* c) {
    sincosf(x, s, c);
}
```

**优点：** 零额外开销 (inline)，代码可读性好。
**缺点：** 需要定义两套函数。

### 11.5 策略 4：模板参数化

```cuda
template <bool UseFastMath>
__device__ __forceinline__ float my_exp(float x) {
    if constexpr (UseFastMath) return __expf(x);
    else return expf(x);
}

template <bool UseFastMath>
__global__ void attention_kernel(const float* Q, const float* K, ...) {
    // ...
    float score = my_exp<UseFastMath>(logit - max_val);
    // ...
}

// 实例化两个版本
template __global__ void attention_kernel<true>(...);
template __global__ void attention_kernel<false>(...);
```

---

## 12. 架构演进：SFU 瓶颈与 Blackwell Ultra

### 12.1 非对称硬件扩展问题

Tensor Core 吞吐每代翻倍，但 SFU 吞吐保持不变——形成 Softmax 瓶颈：

```
架构      Tensor Core 吞吐(BF16)    SFU 吞吐/SM    比例
────────────────────────────────────────────────────────
Ampere    312 TFLOPS               16 result/cyc   基准
Hopper    990 TFLOPS (3.2×)        16 result/cyc   瓶颈加剧
Blackwell 2250 TFLOPS (7.2×)       16 result/cyc   严重瓶颈
B-Ultra   2250 TFLOPS              32 result/cyc   瓶颈缓解 (2× SFU)
```

### 12.2 Attention 循环中的 SFU 瓶颈可视化

```
典型 Attention 循环 (Hopper/Blackwell):

BMM1 (QK^T)     Softmax (SFU)          BMM2 (PV)
████████████   ████████████████████   ████████████
  Tensor Core     SFU (瓶颈!)           Tensor Core
  ~高利用率        ~低利用率              ~高利用率
              ↑                    ↑
              Tensor Core 空闲      Tensor Core 空闲

Blackwell Ultra (2× SFU):

BMM1 (QK^T)     Softmax (SFU)    BMM2 (PV)
████████████   ██████████████   ████████████
  Tensor Core     SFU (2×快)      Tensor Core
              ↑ 等待时间减半 ↑
```

### 12.3 FlashAttention 4 的 SFU 规避策略

FA4 在 Blackwell (SFU 未翻倍) 上使用**软件多项式近似**替代部分 SFU 调用：

```
标准 exp 流程 (SFU):
  x → mul(x, log₂e) → MUFU.EX2 → result
  延迟: ~18 cycles, 受限于 SFU 吞吐

FA4 多项式近似 (CUDA Core):
  将 x 分为整数部分 n 和小数部分 f: x = n + f
  2^n: 直接位操作设置 FP32 指数字段
  2^f: 三次多项式 P(f) ≈ c₀ + c₁f + c₂f² + c₃f³ (4 条 FMA)
  result = 2^n × 2^f
  延迟: ~几条 FMA, 不占用 SFU

FA4 混合策略:
  部分迭代用 SFU (MUFU.EX2)，部分用 CUDA Core 多项式
  可调比例，平衡 SFU 和 FMA 管线利用率
```

### 12.4 Blackwell Ultra 的 SFU 翻倍效果

实测结果 (GB300 vs GB200)：

| 指标 | GB200 (SM 10.0) | GB300 (SM 10.3) | 提升 |
|------|----------------|----------------|------|
| `MUFU.EX2` FLOPS | 基准 | 2× | 2.0× |
| Softmax 块宽度 | 基准 | ~50% | 瓶颈缩短 |
| FP8 前向 Attention 吞吐 | 基准 | +35% | 显著 |
| BMM1-BMM2 间隙 | 长 | 短 | TC 利用率提升 |

---

## 13. 诊断与验证

### 13.1 编译时诊断

```bash
# 查看 PTX 中的近似指令使用情况
nvcc -ptx kernel.cu -o kernel.ptx
grep -c "approx" kernel.ptx              # 计数近似指令
grep -c "MUFU" kernel.sass 2>/dev/null   # SASS 层面的 SFU 调用

# 对比 fast math vs 标准版的 PTX 差异
nvcc -ptx kernel.cu -o kernel_precise.ptx
nvcc -ptx --use_fast_math kernel.cu -o kernel_fast.ptx
diff kernel_precise.ptx kernel_fast.ptx
```

### 13.2 Nsight Compute 关键 Metrics

| Metric | 含义 | 目标 |
|--------|------|------|
| `sm__pipe_fma_cycles_active` | FMA 管线活跃周期 | 对比 SFU |
| `sm__pipe_xu_cycles_active` | SFU (XU) 管线活跃周期 | 如果 >> FMA 则 SFU 是瓶颈 |
| `sm__inst_executed_pipe_xu` | SFU 指令执行数 | SFU 负载量 |
| `sm__inst_executed_pipe_fma` | FMA 指令执行数 | FMA 负载量 |
| `smsp__sass_thread_inst_executed_op_mufu_pct` | MUFU 指令占比 | SFU 密集度 |

```bash
# 收集 SFU vs FMA 管线利用率
ncu --metrics sm__pipe_xu_cycles_active,sm__pipe_fma_cycles_active \
    ./my_program

# 收集 MUFU 指令占比
ncu --metrics smsp__sass_thread_inst_executed_op_mufu_pct \
    ./my_program
```

### 13.3 精度验证方法

```cuda
// 对比 fast math vs 标准版的输出差异
__global__ void verify_precision(const float* input, float* diff, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float precise = expf(input[i]);
        float fast = __expf(input[i]);
        diff[i] = fabsf(precise - fast) / fabsf(precise);  // 相对误差
    }
}

// Host 端统计
float max_rel_error = *std::max_element(diff, diff + N);
float avg_rel_error = std::accumulate(diff, diff + N, 0.0f) / N;
printf("Max relative error: %e\n", max_rel_error);
printf("Avg relative error: %e\n", avg_rel_error);
```

### 13.4 检查 Denormal 影响

```cuda
// 检测计算中是否产生 denormal
__global__ void check_denormals(const float* data, int* denormal_count, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float val = data[i];
        int bits = __float_as_int(val);
        int exp_bits = (bits >> 23) & 0xFF;
        if (exp_bits == 0 && (bits & 0x7FFFFF) != 0) {
            atomicAdd(denormal_count, 1);  // 这是一个 denormal
        }
    }
}
```

---

## 14. 决策 Checklist

### 14.1 是否使用 `--use_fast_math`

```
开始
 │
 ├── 这是推理 (inference) 代码？
 │   ├── 是 → 对精度容忍度高，倾向使用
 │   └── 否 (训练) → 谨慎使用
 │
 ├── 是否有 loss/gradient 计算？
 │   ├── 是 → 不要对该部分使用 fast math
 │   └── 否 → 可以使用
 │
 ├── 是否涉及 powf(负底数)？
 │   ├── 是 → __powf 会返回 NaN，不能使用
 │   └── 否 → 安全
 │
 ├── 是否有大 |y| 的除法 (|y| > 2^126)？
 │   ├── 是 → __fdividef 返回 0，不能使用
 │   └── 否 → 安全
 │
 └── 结果是否需要跨平台一致 (GPU vs CPU)？
     ├── 是 → 不使用 fast math
     └── 否 → 使用 fast math
```

### 14.2 Per-Function 决策表

| LLM 操作 | 建议 | 函数选择 |
|----------|------|---------|
| Softmax exp | ✅ 使用 `__expf` | 精度足够，显著加速 |
| Softmax 归一化 | ✅ 使用 `rcp.approx` | 单指令倒数 |
| GELU (tanh 近似) | ✅ 使用 `tanhf` (Turing+ MUFU.TANH) | 硬件加速 |
| GELU (erf 精确) | ⚠️ 保持 `erff` | 无 intrinsic 替代 |
| SiLU exp(-x) | ✅ 使用 `__expf` | 精度足够 |
| RMSNorm rsqrt | ✅ `rsqrtf` 本身即 MUFU | 已是单指令 |
| RoPE sin/cos | ⚠️ 按需选择 `__sincosf` | 确认 freq < 48039 |
| Loss log(p) | ❌ 保持 `logf` | 需要精确 |
| Gradient 计算 | ❌ 保持标准函数 | 误差累积敏感 |
| 量化 scale 计算 | ❌ 保持标准函数 | 影响量化精度 |

### 14.3 快速检查清单

- [ ] **标识热点**：用 Nsight 确认数学函数是否是瓶颈 (SFU 利用率)
- [ ] **选择粒度**：全局 `--use_fast_math` vs 手动 intrinsic vs 分离编译
- [ ] **检查 edge case**：`__powf` 负底数、`__fdividef` 大除数、`__sinf/__cosf` 大输入
- [ ] **验证精度**：与标准版对比相对误差，确认 top-k 结果不变
- [ ] **检查 denormal**：如果计算涉及极小值，谨慎开启 `--ftz=true`
- [ ] **考虑 FMA**：`--fmad=true` 通常有益，仅在需要跨平台一致性时关闭
- [ ] **FP16 注意**：`--use_fast_math` 间接影响 half 数学函数精度
- [ ] **SFU 瓶颈**：如果 SFU 利用率 > 50%，考虑多项式近似替代 SFU

---

## 参考资源

- [CUDA Programming Guide — Mathematical Functions Appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
- [CUDA Math API Reference — Single Precision Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html)
- [CUDA Math API Reference — Single Precision Standard](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html)
- [NVIDIA Floating Point and IEEE 754 Compliance](https://docs.nvidia.com/cuda/floating-point/index.html)
- [CUDA Pro Tip: Flush Denormals with Confidence](https://developer.nvidia.com/blog/cuda-pro-tip-flush-denormals-confidence/)
- [NVCC Compiler Driver Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
- [Demystifying the NVIDIA Ampere Architecture (arXiv:2208.11174)](https://arxiv.org/pdf/2208.11174)
- [Making Softmax More Efficient with NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/making-softmax-more-efficient-with-nvidia-blackwell-ultra/)
- [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)
- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html)
- [CUDA Half Math Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html)
- [CUDA BFloat16 Math Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html)

---

*本文档作为 LLM Kernel Agent 的 Fast Math 参考。配合 `cuda-core.md`（SFU 架构）和 `official-std/` 目录下的文档共同使用。*
