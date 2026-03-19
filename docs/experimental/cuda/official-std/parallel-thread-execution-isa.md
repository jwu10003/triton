# PTX ISA (Parallel Thread Execution) 参考指南

> 面向 LLM 高性能 Kernel 开发的 PTX 中间表示与指令集速查手册
> 基于 PTX ISA 8.x（CUDA 12.x），覆盖 SM 8.0 – SM 10.0 / SM 12.0

---

## 目录

1. [PTX 概述与定位](#1-ptx-概述与定位)
2. [编程模型映射](#2-编程模型映射)
3. [状态空间 (State Spaces)](#3-状态空间-state-spaces)
4. [基本类型系统](#4-基本类型系统)
5. [寄存器与操作数](#5-寄存器与操作数)
6. [指令集总览](#6-指令集总览)
7. [数据移动与转换指令](#7-数据移动与转换指令)
8. [算术与数学指令](#8-算术与数学指令)
9. [比较与谓词指令](#9-比较与谓词指令)
10. [内存访问指令](#10-内存访问指令)
11. [异步拷贝与 TMA 指令](#11-异步拷贝与-tma-指令)
12. [原子操作指令](#12-原子操作指令)
13. [Warp 级指令](#13-warp-级指令)
14. [Tensor Core 指令](#14-tensor-core-指令)
15. [屏障与同步指令](#15-屏障与同步指令)
16. [特殊寄存器](#16-特殊寄存器)
17. [指令级指示与编译指令](#17-指令级指示与编译指令)
18. [CUDA 中内联 PTX](#18-cuda-中内联-ptx)
19. [LLM Kernel 常用 PTX 模式](#19-llm-kernel-常用-ptx-模式)
20. [架构演进与指令可用性](#20-架构演进与指令可用性)

---

## 1. PTX 概述与定位

### 1.1 什么是 PTX

PTX (Parallel Thread Execution) 是 NVIDIA GPU 的**虚拟指令集架构 (Virtual ISA)**。它是一种稳定的、面向并行线程的底层中间表示，位于 CUDA C++ 与硬件原生指令 (SASS) 之间：

```
CUDA C++ → (nvcc/clang) → PTX → (ptxas) → SASS (机器码)
```

### 1.2 为何 LLM Kernel 开发需要 PTX

| 场景 | 说明 |
|------|------|
| **Tensor Core 编程** | WMMA API 封装有限，`mma.sync`、`wgmma`、`tcgen05` 需通过 PTX 或内联 PTX 调用 |
| **异步数据移动** | `cp.async`、`cp.async.bulk`、TMA 描述符操作需 PTX 级控制 |
| **极致内存优化** | 缓存提示 (`.ca/.cg/.cs/.lu/.cv`)、向量化 `ld.global.v4` 等无法从高级语言直接指定 |
| **Warp 级原语** | `shfl.sync`、`redux.sync`、`match.sync` 细粒度控制 |
| **性能调试** | 阅读 `cuobjdump --dump-ptx` 输出，理解编译器行为，指导优化 |
| **前沿硬件特性** | 新架构指令通常先以 PTX 形式暴露，后续才有 CUDA 内建函数封装 |

### 1.3 PTX 版本与目标架构

```
.version 8.5        // PTX ISA 版本
.target sm_90a      // 目标架构 (a = 加速特性)
.address_size 64    // 64 位地址
```

**版本对应关系（常用）：**

| PTX 版本 | CUDA 版本 | 新增关键特性 |
|----------|-----------|-------------|
| 7.0 | CUDA 11.0 | sm_80 (A100)，`mma.sync` 扩展 |
| 7.1 | CUDA 11.1 | `redux.sync`，`cp.async` |
| 7.8 | CUDA 11.8 | sm_89 (Ada)，sm_90 (Hopper) |
| 8.0 | CUDA 12.0 | `wgmma`，TMA，`mbarrier`，Distributed Shared Memory |
| 8.3 | CUDA 12.3 | sm_90a 加速特性 |
| 8.4 | CUDA 12.4 | `tcgen05` 预览 |
| 8.5 | CUDA 12.5+ | sm_100 (Blackwell 数据中心)，`tcgen05` 完整支持 |
| 8.7 | CUDA 12.8+ | sm_120 (Blackwell 消费级 RTX 50 系列)，`mma.sync` 扩展 (FP4/FP6/FP8) |

---

## 2. 编程模型映射

PTX 程序与 CUDA 编程模型直接对应：

| CUDA 概念 | PTX 对应 |
|-----------|---------|
| `__global__` 函数 | `.entry` 指示 |
| `__device__` 函数 | `.func` 指示 |
| `threadIdx.x` | `%tid.x` 特殊寄存器 |
| `blockIdx.x` | `%ctaid.x` 特殊寄存器 |
| `blockDim.x` | `%ntid.x` 特殊寄存器 |
| `gridDim.x` | `%nctaid.x` 特殊寄存器 |
| `__shared__` 变量 | `.shared` 状态空间 |
| `__syncthreads()` | `bar.sync 0;` |
| Warp 中的 lane | `%laneid` 特殊寄存器 |

**PTX Kernel 骨架示例：**

```
.version 8.0
.target sm_90
.address_size 64

.entry vector_add(
    .param .u64 param_A,
    .param .u64 param_B,
    .param .u64 param_C,
    .param .u32 param_N
)
{
    .reg .u32   %r<10>;
    .reg .u64   %rd<10>;
    .reg .f32   %f<5>;
    .reg .pred  %p<3>;

    // 加载参数
    ld.param.u64    %rd0, [param_A];
    ld.param.u64    %rd1, [param_B];
    ld.param.u64    %rd2, [param_C];
    ld.param.u32    %r0,  [param_N];

    // 计算全局线程 ID
    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ntid.x;
    mov.u32         %r3, %ctaid.x;
    mad.lo.u32      %r4, %r3, %r2, %r1;  // tid = blockIdx.x * blockDim.x + threadIdx.x

    // 边界检查
    setp.ge.u32     %p0, %r4, %r0;
    @%p0 bra        EXIT;

    // 地址计算 (float = 4 bytes)
    mul.wide.u32    %rd3, %r4, 4;
    add.u64         %rd4, %rd0, %rd3;     // &A[tid]
    add.u64         %rd5, %rd1, %rd3;     // &B[tid]
    add.u64         %rd6, %rd2, %rd3;     // &C[tid]

    // 加载、计算、存储
    ld.global.f32   %f0, [%rd4];
    ld.global.f32   %f1, [%rd5];
    add.f32         %f2, %f0, %f1;
    st.global.f32   [%rd6], %f2;

EXIT:
    ret;
}
```

---

## 3. 状态空间 (State Spaces)

PTX 定义 8 种状态空间，对应不同硬件存储层次：

### 3.1 状态空间总表

| 状态空间 | 关键字 | 硬件位置 | 作用域 | 读写 | LLM Kernel 用途 |
|----------|--------|---------|--------|------|----------------|
| **寄存器** | `.reg` | 寄存器文件 | 线程私有 | R/W | 所有中间计算，累加器 |
| **特殊寄存器** | `.sreg` | 硬件只读 | 线程私有 | R | 线程/块/网格 ID |
| **局部内存** | `.local` | 设备内存 (L1 缓存) | 线程私有 | R/W | 寄存器溢出 (spill) |
| **共享内存** | `.shared` | 片上 SRAM | Block 内 | R/W | 数据分块、Warp 间通信 |
| **全局内存** | `.global` | HBM / GDDR | 所有线程 | R/W | 输入/输出张量 |
| **常量内存** | `.const` | 设备内存 (专用缓存) | 所有线程 | R | 超参数、查找表 |
| **参数空间** | `.param` | 寄存器/常量内存 | Kernel 参数 | R | Kernel 入参传递 |
| **纹理内存** | `.tex` | 设备内存 (专用缓存) | 所有线程 | R | (LLM 场景极少使用) |

### 3.2 共享内存详解

```ptx
// 静态分配
.shared .align 16 .b8 smem_buf[16384];   // 16 KB，16 字节对齐

// 动态分配 (通过 extern)
.extern .shared .align 32 .b8 dyn_smem[]; // 大小在 launch 时指定
```

**Hopper Distributed Shared Memory (SM 9.0+)：**
```ptx
.shared::cta    .b8 local_smem[8192];     // 本 CTA 的共享内存
.shared::cluster .b8 remote_smem[8192];   // 集群内可跨 CTA 访问
```

### 3.3 全局内存访问规则

- 事务粒度：32、64 或 128 字节，必须自然对齐
- 单指令访问条件：数据大小 ∈ {1, 2, 4, 8, 16} 字节 **且** 自然对齐
- 非对齐或非标准大小 → 编译器拆分为多条指令，性能下降

---

## 4. 基本类型系统

### 4.1 基本类型

| 类别 | 类型 | 宽度 | 说明 |
|------|------|------|------|
| **谓词** | `.pred` | 1 bit | 条件/分支控制 |
| **无类型位** | `.b8` `.b16` `.b32` `.b64` `.b128` | 8–128 bit | 原始位操作、内存拷贝 |
| **无符号整数** | `.u8` `.u16` `.u32` `.u64` | 8–64 bit | 地址计算、索引 |
| **有符号整数** | `.s8` `.s16` `.s32` `.s64` | 8–64 bit | 偏移量、带符号运算 |
| **浮点** | `.f16` `.f16x2` `.bf16` `.bf16x2` `.f32` `.f64` | 16–64 bit | 核心计算类型 |
| **特殊浮点** | `.tf32` `.e4m3` `.e5m2` | 19/8/8 bit | Tensor Core 专用 |

### 4.2 LLM 常用类型映射

| 精度策略 | PTX 计算类型 | PTX 存储类型 | 对应 C++ |
|----------|-------------|-------------|----------|
| FP32 训练 | `.f32` | `.f32` / `.b32` | `float` |
| FP16 推理 | `.f16` / `.f16x2` | `.b16` / `.b32` | `half` / `half2` |
| BF16 训练/推理 | `.bf16` / `.bf16x2` | `.b16` / `.b32` | `__nv_bfloat16` |
| TF32 Tensor Core | `.tf32` | `.b32` | N/A (MMA 内部) |
| FP8 推理 (Hopper+) | `.e4m3` / `.e5m2` | `.b8` | `__nv_fp8_e4m3` |
| INT8 量化 | `.s8` / `.u8` | `.b8` | `int8_t` |
| INT4 量化 | `.s4` / `.u4` | `.b32` (packed) | 需手动 pack/unpack |

### 4.3 向量类型

PTX 支持 `.v2` 和 `.v4` 向量修饰：

```ptx
.reg .f32 %f<4>;
.reg .v4.f32 %vec;       // 4 个 f32 组成的向量寄存器

ld.global.v4.f32 %vec, [%rd0];   // 128-bit 向量加载 (单条指令)
```

向量宽度约束：`.v2` 可用于所有基本类型，`.v4` 仅用于 32-bit 及以下类型。

---

## 5. 寄存器与操作数

### 5.1 寄存器声明

```ptx
.reg .u32    %r<32>;       // 32 个 u32 寄存器 (%r0 – %r31)
.reg .f32    %f<16>;       // 16 个 f32 寄存器
.reg .pred   %p<8>;        // 8 个谓词寄存器
.reg .u64    %rd<8>;       // 8 个 u64 寄存器 (指针)
.reg .b128   %q<4>;        // 4 个 128-bit 寄存器 (用于 MMA fragment)
```

### 5.2 操作数语法

```ptx
// 立即数
add.f32 %f0, %f1, 1.0;         // 浮点立即数
add.u32 %r0, %r1, 256;         // 整数立即数

// 地址 + 偏移
ld.global.f32 %f0, [%rd0+16];  // 基址 + 常量偏移

// 谓词条件执行
@%p0 add.f32 %f0, %f1, %f2;    // 仅当 %p0 为 true 时执行
@!%p0 bra SKIP;                 // %p0 为 false 时跳转
```

### 5.3 寄存器压力与 LLM Kernel

LLM kernel（特别是 GEMM、FlashAttention）通常需要大量寄存器来保存累加器矩阵 fragment：

| Kernel 类型 | 典型寄存器需求 | 关注点 |
|------------|--------------|--------|
| GEMM（Tensor Core） | 128–256 | 累加器 fragment + 地址计算 |
| FlashAttention | 160–255 | Q/K/V fragment + softmax 中间值 |
| Fused Softmax | 64–96 | 行最大值 + 求和累加器 |
| Elementwise | 16–32 | 向量化加载/存储 |

使用 `.maxnreg` 指示或 nvcc `-maxrregcount` 控制寄存器使用上限。寄存器不足时数据 spill 到 `.local` 空间（性能代价大）。

---

## 6. 指令集总览

PTX 指令按功能分为以下大类：

```
指令集
├── 数据移动与转换 ─── ld / st / mov / cvt / shfl / prmt
├── 算术与数学 ──────── add / sub / mul / fma / mad / div / rcp / sqrt / rsqrt / ex2 / lg2
├── 比较与谓词 ──────── setp / selp / set
├── 逻辑与位操作 ────── and / or / xor / not / shl / shr / bfe / bfi / brev / popc / clz / fns
├── 内存访问 ─────────── ld.global / st.global / ld.shared / st.shared（含缓存提示）
├── 异步拷贝与 TMA ──── cp.async / cp.async.bulk / cp.async.bulk.tensor
├── 原子操作 ─────────── atom / red
├── Warp 级 ─────────── shfl.sync / vote.sync / match.sync / redux.sync
├── Tensor Core ─────── mma.sync / wmma / wgmma / tcgen05
├── 屏障与同步 ──────── bar.sync / mbarrier / fence / membar
└── 控制流 ──────────── bra / call / ret / exit / @pred
```

**指令格式通用模式：**

```
opcode{.modifier}* .type  dst, src1 [, src2 [, src3]];
```

---

## 7. 数据移动与转换指令

### 7.1 加载 / 存储 (ld / st)

```ptx
// 基本格式
ld{.space}{.cop}{.vec}.type   dst, [addr];
st{.space}{.cop}{.vec}.type   [addr], src;
```

**空间 (space)：**
- `.global` `.shared` `.local` `.const` `.param`
- 省略时为 generic（由地址自动判定空间）

**缓存操作 (cop) — 仅 `.global`：**

| 修饰符 | 含义 | LLM 场景 |
|--------|------|---------|
| `.ca` | Cache all levels (默认) | 通用读取 |
| `.cg` | Cache in L2 only，绕过 L1 | 流式大张量读取 |
| `.cs` | Cache streaming，最低优先驱逐 | 一次性访问的大数据 |
| `.lu` | Last use，提示缓存可驱逐 | 消费后不再访问的数据 |
| `.cv` | Cache volatile，每次绕过缓存 | 多 Kernel 间共享的 flag |

**向量化加载/存储：**

```ptx
// 128-bit 向量加载 (4 × float)
ld.global.v4.f32  {%f0, %f1, %f2, %f3}, [%rd0];

// 128-bit 向量存储
st.global.v4.f32  [%rd0], {%f0, %f1, %f2, %f3};

// 64-bit 向量加载 (2 × half2 = 4 × half)
ld.global.v2.b32  {%r0, %r1}, [%rd0];

// 256-bit 加载 (Hopper+，用于 WGMMA fragment)
// 通过 2 × v4.b32 或 ld.global.b128
```

### 7.2 mov — 寄存器间移动

```ptx
mov.u32     %r0, %r1;           // 寄存器复制
mov.u32     %r0, %tid.x;        // 读取特殊寄存器
mov.b64     %rd0, {%r0, %r1};   // 两个 32-bit 合并为 64-bit
mov.b32     {%r0, %r1}, %rd0;   // 64-bit 拆分为两个 32-bit (仅限 b32 合并对)
```

### 7.3 cvt — 类型转换

```ptx
// 浮点精度转换
cvt.f32.f16     %f0, %h0;      // FP16 → FP32 (上转)
cvt.rn.f16.f32  %h0, %f0;      // FP32 → FP16 (下转, 最近偶数舍入)
cvt.rz.bf16.f32 %b0, %f0;      // FP32 → BF16 (截断舍入)

// 整数 ↔ 浮点
cvt.rn.f32.s32  %f0, %r0;      // int32 → float
cvt.rzi.s32.f32 %r0, %f0;      // float → int32 (截断)

// 宽度转换
cvt.u64.u32     %rd0, %r0;     // u32 → u64 (零扩展)
cvt.s64.s32     %rd0, %r0;     // s32 → s64 (符号扩展)
```

**舍入模式：**

| 后缀 | 含义 | 用途 |
|------|------|------|
| `.rn` | 最近偶数 (Round to Nearest Even) | 默认精度转换 |
| `.rz` | 向零截断 (Round toward Zero) | BF16 下转（保持训练稳定性） |
| `.rm` | 向负无穷 | 区间算术下界 |
| `.rp` | 向正无穷 | 区间算术上界 |

### 7.4 prmt — 字节排列 (Permute)

```ptx
prmt.b32 %r0, %r1, %r2, %r3;   // 从 r1:r2 的 8 字节中按 r3 的控制选取 4 字节
```

用于 INT8/INT4 量化中的 pack/unpack 操作，以及 FP8 数据的字节重排。

---

## 8. 算术与数学指令

### 8.1 整数算术

```ptx
add.u32     %r0, %r1, %r2;       // r0 = r1 + r2
sub.s32     %r0, %r1, %r2;       // r0 = r1 - r2
mul.lo.u32  %r0, %r1, %r2;       // r0 = (r1 * r2) 低 32 位
mul.hi.u32  %r0, %r1, %r2;       // r0 = (r1 * r2) 高 32 位
mul.wide.u32 %rd0, %r1, %r2;     // rd0 = r1 * r2 (32×32 → 64)
mad.lo.u32  %r0, %r1, %r2, %r3;  // r0 = r1 * r2 + r3 (低 32 位)
```

**地址计算常见模式：**
```ptx
// offset = tid * stride  (32→64 位扩展乘法)
mul.wide.u32  %rd0, %r_tid, %r_stride;
add.u64       %rd1, %rd_base, %rd0;
```

### 8.2 浮点算术

```ptx
add.f32     %f0, %f1, %f2;           // f32 加
mul.f32     %f0, %f1, %f2;           // f32 乘
fma.rn.f32  %f0, %f1, %f2, %f3;      // f0 = f1 * f2 + f3 (FP32 fused multiply-add)

// FP16 打包运算 (2-way SIMD)
add.f16x2   %r0, %r1, %r2;           // 同时对两个 FP16 求和
mul.f16x2   %r0, %r1, %r2;           // 同时对两个 FP16 求积
fma.rn.f16x2 %r0, %r1, %r2, %r3;     // 打包 FMA

// BF16 打包运算
add.bf16x2  %r0, %r1, %r2;
fma.rn.bf16x2 %r0, %r1, %r2, %r3;

// FP64
fma.rn.f64  %fd0, %fd1, %fd2, %fd3;
```

### 8.3 数学函数（快速近似 vs 精确）

| 指令 | 精度 | 吞吐 (SM 8.0) | LLM 用途 |
|------|------|---------------|---------|
| `rcp.approx.f32` | ~2^-23 相对误差 | 1 cyc/warp | 除法近似 (1/x) |
| `rcp.rn.f32` | IEEE 精确 | ~8 cyc/warp | 精确除法 |
| `rsqrt.approx.f32` | ~2^-23 | 1 cyc/warp | LayerNorm 中 1/√x |
| `sqrt.rn.f32` | IEEE 精确 | ~8 cyc/warp | 精确平方根 |
| `ex2.approx.f32` | ~2^-23 | 1 cyc/warp | exp 近似 (2^x) |
| `lg2.approx.f32` | ~2^-23 | 1 cyc/warp | log 近似 (log2) |
| `sin.approx.f32` | ~2^-21 | 1 cyc/warp | 激活函数 |
| `cos.approx.f32` | ~2^-21 | 1 cyc/warp | 位置编码 |

**LLM 常用数学模式：**

```ptx
// exp(x) = 2^(x * log2(e))
mul.f32         %f1, %f0, 0f3FB8AA3B;   // x * log2(e), log2(e) ≈ 1.4427
ex2.approx.f32  %f2, %f1;               // 2^(x*log2e) ≈ exp(x)

// 1/sqrt(x) for LayerNorm
rsqrt.approx.f32 %f1, %f0;              // 快速倒数平方根

// tanh(x) 近似: 通过 exp 实现
// tanh(x) = 1 - 2/(exp(2x)+1)
```

### 8.4 min / max / abs / neg

```ptx
min.f32     %f0, %f1, %f2;   // ReLU 裁剪
max.f32     %f0, %f1, 0f00000000;  // ReLU: max(x, 0)
abs.f32     %f0, %f1;
neg.f32     %f0, %f1;

// FP16 打包
min.f16x2   %r0, %r1, %r2;
max.f16x2   %r0, %r1, %r2;
```

---

## 9. 比较与谓词指令

### 9.1 setp — 设置谓词

```ptx
setp.lt.f32     %p0, %f0, %f1;         // p0 = (f0 < f1)
setp.ge.u32     %p0, %r0, %r_N;        // p0 = (r0 >= N) — 边界检查
setp.eq.s32     %p0, %r0, 0;           // p0 = (r0 == 0)
setp.ne.and.s32 %p0, %r0, 0, %p1;      // p0 = (r0 != 0) AND p1
```

**NaN 处理比较操作符：**

| 操作符 | NaN 结果 | 用途 |
|--------|---------|------|
| `.lt` `.le` `.gt` `.ge` `.eq` `.ne` | 有序比较 (NaN → false) | 通用比较 |
| `.ltu` `.leu` `.gtu` `.geu` `.equ` `.neu` | 无序比较 (NaN → true) | NaN 检测 |
| `.num` | 两操作数均非 NaN | NaN 过滤 |
| `.nan` | 至少一个是 NaN | NaN 检测 |

### 9.2 selp — 条件选择 (无分支)

```ptx
selp.f32    %f0, %f1, %f2, %p0;    // f0 = p0 ? f1 : f2
selp.b32    %r0, %r1, %r2, %p0;    // 整数条件选择
```

**LLM 中避免 Warp 分歧的关键技巧：** 用 `selp` 替代条件分支。

```ptx
// 不良: 分支 (warp divergence)
@%p0 bra TRUE_BRANCH;
mov.f32 %f0, 0f00000000;
bra END;
TRUE_BRANCH:
mov.f32 %f0, %f1;
END:

// 优良: 无分支条件选择
selp.f32 %f0, %f1, 0f00000000, %p0;
```

### 9.3 谓词执行 (Predicated Execution)

PTX 中任何指令都可以加谓词前缀：

```ptx
@%p0 ld.global.f32  %f0, [%rd0];     // 仅 p0=true 时执行加载
@!%p1 st.global.f32 [%rd1], %f0;     // 仅 p1=false 时执行存储
```

谓词执行将条件控制转换为数据依赖，消除分支，避免 warp 分歧。这是 PTX 中实现高效条件逻辑的首选方式。

---

## 10. 内存访问指令

### 10.1 全局内存加载

```ptx
// 标量加载
ld.global.f32       %f0, [%rd0];
ld.global.u8        %r0, [%rd0];       // 加载 1 字节，零扩展到 32-bit

// 向量化加载 (关键性能优化)
ld.global.v2.f32    {%f0, %f1}, [%rd0];         // 64-bit
ld.global.v4.f32    {%f0, %f1, %f2, %f3}, [%rd0]; // 128-bit (推荐)
ld.global.v4.b32    {%r0, %r1, %r2, %r3}, [%rd0]; // 128-bit 无类型

// 带缓存提示
ld.global.cg.v4.f32 {%f0, %f1, %f2, %f3}, [%rd0]; // 绕过 L1
ld.global.cs.f32    %f0, [%rd0];                    // 流式，优先驱逐
```

### 10.2 共享内存加载/存储

```ptx
ld.shared.f32       %f0, [%r_smem_addr];
st.shared.v4.f32    [%r_smem_addr], {%f0, %f1, %f2, %f3};

// 共享内存偏移计算
// smem_addr = base + threadIdx.x * element_size
shl.b32     %r0, %tid.x, 2;          // tid * 4 (float)
add.u32     %r1, %r_smem_base, %r0;
ld.shared.f32 %f0, [%r1];
```

### 10.3 Prefetch 预取

```ptx
prefetch.global.L1  [%rd0];      // 预取到 L1
prefetch.global.L2  [%rd0];      // 预取到 L2
prefetchu.L1        [%rd0];      // 统一地址空间预取
```

### 10.4 LDG / STG 非缓存加载 (Legacy)

```ptx
// SM 3.5+ 纹理路径只读加载 (通过 __ldg() 或 const __restrict__)
ld.global.nc.f32  %f0, [%rd0];   // .nc = non-coherent, 走纹理缓存路径
```

从 SM 8.0 起，L1 缓存统一后 `.nc` 修饰符的性能优势减弱，但在特定访问模式下仍有用。

---

## 11. 异步拷贝与 TMA 指令

### 11.1 cp.async — 异步全局→共享拷贝 (SM 8.0+)

```ptx
// 绕过寄存器，直接从 Global → Shared
cp.async.ca.shared.global [%r_smem], [%rd_gmem], 16;     // 拷贝 16 字节
cp.async.cg.shared.global [%r_smem], [%rd_gmem], 16;     // L2 only

// 可选：带源大小和填充（用于边界处理）
cp.async.ca.shared.global [%r_smem], [%rd_gmem], 16, 12;
// 拷贝 12 字节实际数据 + 4 字节零填充 = 16 字节

// 提交异步组
cp.async.commit_group;

// 等待异步完成 (N = 还允许未完成的组数)
cp.async.wait_group N;
// cp.async.wait_group 0;  // 等待所有组完成
// cp.async.wait_all;       // 等价于 wait_group 0
```

**双缓冲 Pipeline 模式 (LLM GEMM 核心)：**

```ptx
// Stage 0: 启动第一批异步拷贝
cp.async.ca.shared.global [smem_buf0], [gmem_tile0], 16;
cp.async.commit_group;

// Stage 1: 启动第二批异步拷贝
cp.async.ca.shared.global [smem_buf1], [gmem_tile1], 16;
cp.async.commit_group;

// 主循环
LOOP:
    cp.async.wait_group 1;         // 等待 buf0 就绪（允许 1 组未完成）
    bar.sync 0;                    // 确保所有线程看到数据

    // 计算 buf0 中的数据
    // ...

    // 启动下一批异步拷贝到 buf0
    cp.async.ca.shared.global [smem_buf0], [gmem_next_tile], 16;
    cp.async.commit_group;

    // 交换 buffer 指针，继续循环
```

### 11.2 cp.async.bulk — 批量异步拷贝 (SM 9.0+)

```ptx
// 一次拷贝大块数据 (以 mbarrier 同步)
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
    [%r_smem_dst], [%rd_gmem_src], %r_size, [%r_mbar];

// 全局 → 共享，由 mbarrier 追踪完成
cp.async.bulk.shared.global.mbarrier::complete_tx::bytes
    [%r_smem], [%rd_gmem], 4096, [%r_mbar];
```

### 11.3 TMA — Tensor Memory Accelerator (SM 9.0+)

TMA 是 Hopper 架构引入的硬件张量拷贝引擎，支持多维张量描述符驱动的异步拷贝。

```ptx
// 创建 TMA 描述符 (Host 端通过 CUDA Driver API)
// cuTensorMapEncode*()

// 1D TMA 加载
cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [%r_smem], [%rd_tma_desc, {%r_coord0}], [%r_mbar];

// 2D TMA 加载 (GEMM 中的矩阵 tile)
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [%r_smem], [%rd_tma_desc, {%r_coord0, %r_coord1}], [%r_mbar];

// 3D TMA 加载 (Attention 中的 Q/K/V)
cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [%r_smem], [%rd_tma_desc, {%r_batch, %r_head, %r_seq}], [%r_mbar];

// TMA 存储 (Shared → Global)
cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group
    [%rd_tma_desc, {%r_coord0, %r_coord1}], [%r_smem];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0;
```

**TMA 对 LLM Kernel 的价值：**

| 特性 | 优势 |
|------|------|
| 硬件多维寻址 | 自动处理 stride、padding，减少地址计算指令 |
| 绕过寄存器 | Global → Shared 不占用寄存器带宽 |
| 边界检查 | 硬件自动处理越界 (out-of-bounds → 零填充) |
| 仅 1 个线程发起 | 减少指令冗余 (其他线程做计算) |
| Multicast | 一次拷贝广播到 Cluster 中多个 CTA |

---

## 12. 原子操作指令

### 12.1 atom — 原子读-改-写

```ptx
// 原子加
atom.global.add.f32     %f0, [%rd0], %f1;   // *addr += val, 返回旧值
atom.shared.add.u32     %r0, [%r_smem], %r1;

// 原子 CAS (Compare-And-Swap)
atom.global.cas.b32     %r0, [%rd0], %r1, %r2;  // if (*addr == r1) *addr = r2

// 原子 min/max (用于 Softmax 的行最大值)
atom.global.max.f32     %f0, [%rd0], %f1;

// 原子位操作
atom.global.or.b32      %r0, [%rd0], %r1;
atom.global.and.b32     %r0, [%rd0], %r1;
```

### 12.2 red — 归约 (不返回旧值)

```ptx
red.global.add.f32      [%rd0], %f0;   // *addr += val (不返回旧值，可能更快)
red.shared.add.u32      [%r_smem], %r0;
```

### 12.3 原子操作的精度支持

| 操作 | `.u32` | `.s32` | `.f32` | `.f64` | `.f16x2` |
|------|--------|--------|--------|--------|----------|
| `add` | ✓ | ✓ | ✓ (SM 2.0+) | ✓ (SM 6.0+) | ✓ (SM 7.0+) |
| `min/max` | ✓ | ✓ | ✓ (SM 2.0+) | ✗ | ✗ |
| `cas` | ✓ (`.b32`) | — | — | ✓ (`.b64`) | — |
| `and/or/xor` | ✓ (`.b32`) | — | — | — | — |

**LLM 中的原子使用场景：**
- 跨 block 归约（AllReduce 最终阶段）
- 动态 token 计数
- 分布式 Softmax 的全局 max/sum

---

## 13. Warp 级指令

### 13.1 shfl.sync — Warp Shuffle

```ptx
// 通用格式
shfl.sync.mode.b32  dst, src, offset_or_lane, mask_and_clamp, membermask;

// Shuffle Down (用于 warp 内归约)
shfl.sync.down.b32  %r0, %r1, 16, 0x1f, 0xFFFFFFFF;
// 从 lane + 16 读取 (实现 16-wide 归约)

// Shuffle Up
shfl.sync.up.b32    %r0, %r1, 1, 0, 0xFFFFFFFF;
// 从 lane - 1 读取 (前缀和)

// Shuffle XOR (蝴蝶归约)
shfl.sync.bfly.b32  %r0, %r1, 8, 0x1f, 0xFFFFFFFF;
// 从 lane ^ 8 读取

// Shuffle Index (广播)
shfl.sync.idx.b32   %r0, %r1, 0, 0x1f, 0xFFFFFFFF;
// 从 lane 0 广播
```

**Warp 内归约 (Softmax 行求和)：**

```ptx
// 假设 %f0 中有每个 lane 的局部和
shfl.sync.bfly.b32 %f1, %f0, 16, 0x1f, 0xFFFFFFFF;
add.f32            %f0, %f0, %f1;
shfl.sync.bfly.b32 %f1, %f0, 8, 0x1f, 0xFFFFFFFF;
add.f32            %f0, %f0, %f1;
shfl.sync.bfly.b32 %f1, %f0, 4, 0x1f, 0xFFFFFFFF;
add.f32            %f0, %f0, %f1;
shfl.sync.bfly.b32 %f1, %f0, 2, 0x1f, 0xFFFFFFFF;
add.f32            %f0, %f0, %f1;
shfl.sync.bfly.b32 %f1, %f0, 1, 0x1f, 0xFFFFFFFF;
add.f32            %f0, %f0, %f1;
// 结果: 所有 lane 中 %f0 = warp 内所有值之和
```

### 13.2 vote.sync — Warp 投票

```ptx
vote.sync.all.pred  %p1, %p0, 0xFFFFFFFF;  // p1 = warp 内所有 lane 的 p0 均为 true
vote.sync.any.pred  %p1, %p0, 0xFFFFFFFF;  // p1 = warp 内任一 lane 的 p0 为 true
vote.sync.uni.pred  %p1, %p0, 0xFFFFFFFF;  // p1 = warp 内所有 lane 的 p0 相同
vote.sync.ballot.b32 %r0, %p0, 0xFFFFFFFF; // r0 = 32-bit bitmask，每位对应一个 lane
```

### 13.3 match.sync — 值匹配 (SM 7.0+)

```ptx
match.any.sync.b32  %r0, %r1, 0xFFFFFFFF;
// r0 = 与当前 lane 的 r1 值相同的所有 lane 的 bitmask

match.all.sync.b32  %r0, %p0, %r1, 0xFFFFFFFF;
// r0 = 当所有 lane 的 r1 值相同时的值, p0 = 是否全部相同
```

### 13.4 redux.sync — Warp 归约 (SM 8.0+)

```ptx
redux.sync.add.u32  %r0, %r1, 0xFFFFFFFF;  // r0 = warp 内所有 r1 的和
redux.sync.min.s32  %r0, %r1, 0xFFFFFFFF;  // r0 = warp 内最小值
redux.sync.max.u32  %r0, %r1, 0xFFFFFFFF;  // r0 = warp 内最大值
redux.sync.or.b32   %r0, %r1, 0xFFFFFFFF;  // r0 = 按位 OR
redux.sync.and.b32  %r0, %r1, 0xFFFFFFFF;  // r0 = 按位 AND
```

> **注意：** `redux.sync` 仅支持整数类型。浮点归约仍需使用 `shfl.sync` 循环实现。

---

## 14. Tensor Core 指令

### 14.1 指令演进

| 指令 | 引入架构 | 特点 | LLM 适用性 |
|------|---------|------|-----------|
| `wmma` | SM 7.0 (V100) | CUDA C++ API + PTX，简单 | 入门，灵活性受限 |
| `mma.sync` | SM 7.5+ | PTX only，精确控制 fragment 布局 | 高性能 GEMM |
| `mma.sync` (扩展) | SM 12.0 (RTX 5090) | 同上 + FP4/FP6/FP8 新类型 | 消费级 Blackwell |
| `wgmma` | SM 9.0 (Hopper) | 异步，warpgroup (128 线程)，直接从 Shared Mem 输入 | 最高性能 GEMM |
| `tcgen05` | SM 10.0 (B200) | 新一代，Tensor Memory，更大 tile，仅数据中心 | 下一代 LLM |

> **SM 10.0 vs SM 12.0：** 同为 Blackwell 品牌但 Tensor Core 编程模型完全不同。SM 10.0 (B200/B100) 使用 `tcgen05` + TMEM；SM 12.0 (RTX 5090) 使用 `mma.sync` 扩展版 (SASS 层面为 HMMA.16816)，**不支持 wgmma 和 tcgen05**，TMEM 硬件不存在。需要维护三套独立代码路径 (Hopper / 数据中心 Blackwell / 消费级 Blackwell)。

### 14.2 mma.sync — 同步矩阵乘累加 (SM 7.5+)

```ptx
// 格式: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//       M×N×K    A布局 B布局 D类型 A类型 B类型 C类型
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%f0, %f1, %f2, %f3},                           // D fragment (4×f32)
    {%r0, %r1, %r2, %r3},                           // A fragment (4×b32 = 8×f16)
    {%r4, %r5},                                      // B fragment (2×b32 = 4×f16)
    {%f4, %f5, %f6, %f7};                           // C fragment (4×f32)
```

**常用 MMA 形状与类型组合：**

| MMA Shape | A/B Type | C/D Type | 算力需求 |
|-----------|----------|----------|---------|
| `m16n8k8` | `.f16` | `.f32` | SM 7.5+ |
| `m16n8k16` | `.f16` | `.f32` | SM 8.0+ |
| `m16n8k16` | `.bf16` | `.f32` | SM 8.0+ |
| `m16n8k8` | `.tf32` | `.f32` | SM 8.0+ |
| `m16n8k32` | `.s8` / `.u8` | `.s32` | SM 8.0+ |
| `m16n8k32` | `.e4m3` / `.e5m2` | `.f32` | SM 8.9+ |
| `m16n8k64` | `.e4m3` | `.f32` | SM 8.9+ |

**Fragment 寄存器布局 (m16n8k16.f16→f32 示例)：**

每个 warp (32 线程) 协作执行一个 MMA：
- **A fragment**: 每线程 4 个 b32 寄存器 (存储 8 个 f16 元素)
- **B fragment**: 每线程 2 个 b32 寄存器 (存储 4 个 f16 元素)
- **C/D fragment**: 每线程 4 个 f32 寄存器

线程到矩阵元素的映射由硬件规定，需按文档中的映射表进行数据布局。

### 14.3 wgmma — Warpgroup MMA (SM 9.0+)

```ptx
// Warpgroup = 4 个 warp = 128 线程
// 形状更大: m64nNk16 (N = 8, 16, 24, ..., 256)
// A 操作数可直接从 shared memory 读取

// 异步 WGMMA: A 从 shared memory, B 从 shared memory
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
    {%f0, ..., %f63},              // D: 每线程 64 个 f32 累加器
    [%r_smem_A],                    // A: shared memory 描述符
    [%r_smem_B],                    // B: shared memory 描述符
    p,                              // 谓词 (scale D)
    1, 1,                           // scale A, scale B
    0,                              // negate A
    0;                              // negate B

// 提交与等待
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;
```

**WGMMA 对比 mma.sync 的优势 (LLM GEMM)：**

| 特性 | mma.sync | wgmma |
|------|----------|-------|
| 参与线程 | 32 (1 warp) | 128 (4 warps) |
| 最大 tile | m16n8k16 | m64n256k16 |
| A 操作数来源 | 寄存器 | Shared memory / 寄存器 |
| 执行模式 | 同步 | 异步 (可与数据加载重叠) |
| 寄存器压力 | 较高 (A 需载入寄存器) | 较低 (A 直接从 smem 读) |

### 14.4 tcgen05 — 第五代 Tensor Core (SM 10.0, 仅数据中心 Blackwell)

> **注意：** tcgen05 / TMEM 仅在数据中心级 Blackwell (SM 10.0, B200/B100) 上可用。消费级 Blackwell RTX 5090 (SM 12.0) **不支持** tcgen05、TMEM 和 wgmma，其 Tensor Core 使用 `mma.sync` 扩展版。

```ptx
// Blackwell 引入的新一代 Tensor Core 指令
// 使用 Tensor Memory (tmem) 存储累加器
// 更大的 tile size，更高吞吐

tcgen05.mma.cta_group::1.kind::f16
    [%r_tmem_addr],                 // 累加器在 Tensor Memory 中的地址
    [%r_smem_A_desc],               // A 矩阵描述符 (shared memory)
    [%r_smem_B_desc],               // B 矩阵描述符 (shared memory)
    %r_idesc,                       // 立即描述符
    enable_input_d,                 // 是否使用 D 作为输入累加
    flags;

// 提交/等待
tcgen05.commit;
tcgen05.wait;

// 从 Tensor Memory 搬运结果
tcgen05.st.sync.aligned.16x256b.x64.b32
    [%r_smem_dst], [%r_tmem_src];   // tmem → shared memory

// 或直接存回全局内存
```

**tcgen05 关键特性：**

| 特性 | 说明 |
|------|------|
| Tensor Memory (tmem) | 专用存储，不占用通用寄存器文件 |
| CTA Group | 多个 CTA 协作执行单个 MMA |
| 支持类型 | FP16, BF16, TF32, FP8 (E4M3/E5M2), INT8 |
| 更大 tile | MMA tile 可达 256×256 |

---

## 15. 屏障与同步指令

### 15.1 bar — Block 级屏障

```ptx
bar.sync    0;                      // 等价于 __syncthreads()
bar.sync    0, %r_thread_count;     // 仅等待指定数量的线程到达
bar.arrive  0, %r_thread_count;     // 到达但不等待
bar.red.or.pred %p0, 0, %p1;       // 屏障 + 谓词归约
```

### 15.2 mbarrier — 异步屏障 (SM 9.0+)

mbarrier 是 Hopper 引入的异步屏障机制，用于协调异步数据传输与计算：

```ptx
// 在 shared memory 中初始化 mbarrier
mbarrier.init.shared.b64 [%r_mbar], %r_expected_count;

// 线程到达 (减少计数)
mbarrier.arrive.shared.b64 %r_state, [%r_mbar];

// 异步拷贝完成时自动到达 (TMA 配合)
// cp.async.bulk ... mbarrier::complete_tx::bytes ... [%r_mbar];

// 等待 mbarrier 完成
WAIT_LOOP:
    mbarrier.try_wait.shared.b64 %p0, [%r_mbar], %r_state;
    @!%p0 bra WAIT_LOOP;

// Phase-based: mbarrier 可多次复用，通过 phase 位区分
mbarrier.arrive.expect_tx.shared.b64 %r_state, [%r_mbar], %r_tx_bytes;
```

**mbarrier 在 LLM Kernel 中的角色：**

```
Pipeline 阶段:
  Stage k:   TMA load → mbarrier.arrive(tx_bytes) → ... → mbarrier.try_wait → compute
  Stage k+1: TMA load → mbarrier.arrive(tx_bytes) → ... → mbarrier.try_wait → compute
  ...
```

### 15.3 fence — 内存栅栏

```ptx
fence.sc.gpu;           // Sequential Consistency，GPU 范围
fence.acq_rel.sys;      // Acquire-Release，系统范围 (含 CPU)
fence.sc.cta;           // CTA 范围

fence.proxy.async;      // 异步代理栅栏 (确保异步操作对后续指令可见)
fence.proxy.async.shared::cta;  // 异步拷贝到 shared memory 后的可见性保证
```

### 15.4 membar — 内存屏障 (Legacy)

```ptx
membar.cta;     // Block 内内存可见性 (类似 __threadfence_block())
membar.gl;      // 全局内存可见性 (类似 __threadfence())
membar.sys;     // 系统级 (含 CPU) 可见性 (类似 __threadfence_system())
```

> SM 9.0+ 推荐使用 `fence` 替代 `membar`。

---

## 16. 特殊寄存器

### 16.1 线程/Block/Grid 标识

| 寄存器 | 含义 | CUDA 等价 |
|--------|------|----------|
| `%tid.x/y/z` | 线程在 Block 内的索引 | `threadIdx.x/y/z` |
| `%ntid.x/y/z` | Block 维度 | `blockDim.x/y/z` |
| `%ctaid.x/y/z` | Block 在 Grid 内的索引 | `blockIdx.x/y/z` |
| `%nctaid.x/y/z` | Grid 维度 | `gridDim.x/y/z` |

### 16.2 Warp/SM 标识

| 寄存器 | 含义 | 典型用途 |
|--------|------|---------|
| `%laneid` | 线程在 Warp 内的位置 (0–31) | Shuffle 控制、MMA fragment 映射 |
| `%warpid` | Warp 在 Block 内的 ID | Warpgroup 划分 |
| `%nwarpid` | Block 内 Warp 总数 | 负载均衡 |
| `%smid` | 当前 SM 编号 | 调试、性能分析 |
| `%nsmid` | SM 总数 | — |

### 16.3 时钟与性能计数

| 寄存器 | 含义 | 用途 |
|--------|------|------|
| `%clock` | 32-bit SM 周期计数器 | 微基准测试 |
| `%clock64` | 64-bit SM 周期计数器 | 长时间测量 |
| `%globaltimer` | 64-bit 全局纳秒计时器 | 跨 SM 时间同步 |

```ptx
// 性能测量示例
mov.u64     %rd_start, %clock64;
// ... 被测代码 ...
mov.u64     %rd_end, %clock64;
sub.u64     %rd_elapsed, %rd_end, %rd_start;
```

### 16.4 其他有用的特殊寄存器

| 寄存器 | 含义 |
|--------|------|
| `%lanemask_eq` | 仅当前 lane 的 bitmask |
| `%lanemask_le` | 当前 lane 及更低 lane 的 bitmask |
| `%lanemask_lt` | 低于当前 lane 的 bitmask |
| `%lanemask_ge` | 当前 lane 及更高 lane 的 bitmask |
| `%lanemask_gt` | 高于当前 lane 的 bitmask |
| `%dynamic_smem_size` | 动态共享内存大小 (字节) |
| `%total_smem_size` | 总共享内存大小 (静态+动态) |

---

## 17. 指令级指示与编译指令

### 17.1 核心指示 (Directives)

```ptx
.version 8.5                    // PTX ISA 版本
.target sm_90a                  // 目标架构 (a = 架构特定加速)
.address_size 64                // 地址宽度

.entry kernel_name(...)  { }    // Kernel 入口 (__global__)
.func  device_func(...)  { }   // 设备函数 (__device__)

.extern .func ext_func(...);    // 外部函数声明 (链接时解析)
.weak   .func weak_func(...);   // 弱符号
```

### 17.2 性能调优指示

```ptx
// 限制每线程最大寄存器数
.maxnreg 128

// 限制每 Block 最大线程数
.maxntid 256, 1, 1

// 要求最少 Block 数/SM (用于 occupancy 控制)
.minnctapersm 2

// 对齐要求
.align 128

// Pragma
.pragma "nounroll";             // 禁止循环展开
```

### 17.3 性能调优组合示例

```ptx
.entry flash_attention_fwd(...)
.maxnreg 255          // 允许最大寄存器使用
.maxntid 128          // 128 线程/Block
.minnctapersm 1       // 至少 1 个 Block/SM
{
    // kernel body
}
```

---

## 18. CUDA 中内联 PTX

### 18.1 基本语法

```cpp
// asm volatile("PTX指令" : 输出操作数 : 输入操作数 : clobber列表);
asm volatile("add.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
```

**操作数约束符：**

| 约束 | 含义 | PTX 类型 |
|------|------|---------|
| `"r"` | 32-bit 整数寄存器 | `.u32` / `.s32` / `.b32` |
| `"l"` | 64-bit 整数寄存器 | `.u64` / `.s64` / `.b64` |
| `"f"` | 32-bit 浮点寄存器 | `.f32` |
| `"d"` | 64-bit 浮点寄存器 | `.f64` |
| `"h"` | 16-bit 整数寄存器 | `.u16` / `.s16` |
| `"n"` | 编译时整数常量 | 立即数 |
| `"="` | 输出 (write-only) | — |
| `"+"` | 输入+输出 (read-write) | — |

### 18.2 常用内联 PTX 模式

**cp.async 异步拷贝：**

```cpp
__device__ void cp_async_4B(void* smem, const void* gmem) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(gmem)
    );
}

__device__ void cp_async_16B(void* smem, const void* gmem) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem)
    );
}

__device__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" :::);
}

__device__ void cp_async_wait_group(int n) {
    // n 必须是编译时常量
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}
```

**LDGSTS (全局加载直存共享)：**

```cpp
// 128-bit 全局→共享 (绕过寄存器)
__device__ void ldgsts_128(void* smem, const void* gmem) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem)
    );
}
```

**mma.sync 内联调用：**

```cpp
__device__ void mma_m16n8k16_f16_f32(
    float* d, uint32_t* a, uint32_t* b, float* c)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}
```

**Warp Shuffle：**

```cpp
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        asm volatile(
            "{"
            ".reg .f32 tmp;"
            "shfl.sync.bfly.b32 tmp, %0, %1, 0x1f, 0xFFFFFFFF;"
            "add.f32 %0, %0, tmp;"
            "}"
            : "+f"(val) : "r"(offset)
        );
    }
    return val;
}
```

**缓存提示加载：**

```cpp
// 绕过 L1 的全局加载
__device__ float ldg_cg(const float* addr) {
    float val;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(val) : "l"(addr));
    return val;
}

// 128-bit 向量加载
__device__ float4 ldg_128(const float* addr) {
    float4 val;
    asm volatile(
        "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
        : "l"(addr)
    );
    return val;
}
```

### 18.3 地址空间转换

```cpp
// Generic → Shared (用于 cp.async 等需要 .shared 地址的指令)
__device__ uint32_t cvta_to_shared(void* ptr) {
    uint32_t addr;
    asm("cvta.to.shared.u32 %0, %1;" : "=r"(addr) : "l"(ptr));
    return addr;
    // 或直接使用内置: __cvta_generic_to_shared(ptr)
}

// Generic → Global
__device__ uint64_t cvta_to_global(void* ptr) {
    uint64_t addr;
    asm("cvta.to.global.u64 %0, %1;" : "=l"(addr) : "l"(ptr));
    return addr;
}
```

---

## 19. LLM Kernel 常用 PTX 模式

### 19.1 高效 GEMM Pipeline (Hopper)

```
┌─────────────────────────────────────────────────┐
│  Stage 0: TMA Load Tile A₀, B₀ → smem_buf[0]   │
│           mbarrier.arrive(tx_bytes)              │
├─────────────────────────────────────────────────┤
│  Stage 1: TMA Load Tile A₁, B₁ → smem_buf[1]   │
│           mbarrier.arrive(tx_bytes)              │
│           mbarrier.try_wait(buf[0])              │
│           WGMMA: C += A₀ × B₀                   │
├─────────────────────────────────────────────────┤
│  Stage 2: TMA Load Tile A₂, B₂ → smem_buf[2]   │
│           mbarrier.arrive(tx_bytes)              │
│           mbarrier.try_wait(buf[1])              │
│           WGMMA: C += A₁ × B₁                   │
├─────────────────────────────────────────────────┤
│  ...  (滚动 pipeline，N-stage 缓冲)              │
├─────────────────────────────────────────────────┤
│  Epilogue: WGMMA wait → fence → TMA Store C     │
└─────────────────────────────────────────────────┘
```

关键 PTX 指令序列：

```ptx
// 1. TMA 加载
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [smem_A], [tma_desc_A, {coord_m, coord_k}], [mbar];
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [smem_B], [tma_desc_B, {coord_n, coord_k}], [mbar];

// 2. 等待数据就绪
mbarrier.try_wait.shared.b64 %p0, [mbar_prev], %state;

// 3. WGMMA 计算
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
    {acc0..acc63}, [smem_A_desc], [smem_B_desc], ...;
wgmma.commit_group.sync.aligned;

// 4. 等待计算完成
wgmma.wait_group.sync.aligned 0;

// 5. 存储结果
fence.proxy.async;
cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group
    [tma_desc_C, {coord_m, coord_n}], [smem_C];
```

### 19.2 FlashAttention 核心模式

```
FlashAttention 在 PTX 层面的关键操作:

1. Q tile 加载 (一次): TMA/cp.async → smem_Q
2. K tile 循环加载: TMA/cp.async → smem_K (每个 K block)
3. S = Q × K^T:  mma.sync / wgmma
4. 行最大值:     shfl.sync.bfly (warp 归约) + shared memory (跨 warp)
5. exp(S - max): ex2.approx.f32 (2^(x * log2e))
6. 行求和:       shfl.sync.bfly + shared memory
7. V tile 加载:  TMA/cp.async → smem_V
8. O = P × V:   mma.sync / wgmma
9. 在线 rescale: fma.rn.f32 (O_old *= scale_correction)
10. 存储 O:      st.global / TMA store
```

### 19.3 Fused Softmax PTX 关键序列

```ptx
// --- 行最大值 (Online) ---
// 每线程处理 row_len/warp_size 个元素
max.f32         %f_max, %f_max, %f_elem;       // 局部 max

// Warp 内归约
shfl.sync.bfly.b32 %f_tmp, %f_max, 16, 0x1f, 0xFFFFFFFF;
max.f32         %f_max, %f_max, %f_tmp;
shfl.sync.bfly.b32 %f_tmp, %f_max, 8, 0x1f, 0xFFFFFFFF;
max.f32         %f_max, %f_max, %f_tmp;
// ... (shfl bfly 4, 2, 1)

// 跨 Warp 归约 (通过 shared memory)
st.shared.f32   [smem_max + warp_id * 4], %f_max;
bar.sync 0;
// lane 0 of warp 0 归约所有 warp 的 max
// ...

// --- 指数与求和 ---
sub.f32         %f_diff, %f_elem, %f_max;       // x - max
mul.f32         %f_scaled, %f_diff, 0f3FB8AA3B;  // (x - max) * log2(e)
ex2.approx.f32  %f_exp, %f_scaled;               // exp(x - max)
add.f32         %f_sum, %f_sum, %f_exp;          // 累加

// ... warp 归约 sum (同 max 模式) ...

// --- 归一化 ---
rcp.approx.f32  %f_inv_sum, %f_sum;              // 1/sum
mul.f32         %f_out, %f_exp, %f_inv_sum;      // softmax = exp/sum
```

### 19.4 量化/反量化 PTX 模式

```ptx
// INT8 反量化: output = (int8_val - zero_point) * scale
// 加载 4 个 INT8 (packed in b32)
ld.global.b32   %r0, [%rd_int8_ptr];

// 解包 (使用 prmt 或位操作)
bfe.s32         %r1, %r0, 0, 8;     // 提取 byte 0 (带符号)
bfe.s32         %r2, %r0, 8, 8;     // 提取 byte 1
bfe.s32         %r3, %r0, 16, 8;    // 提取 byte 2
bfe.s32         %r4, %r0, 24, 8;    // 提取 byte 3

// 转换为 float
cvt.rn.f32.s32  %f0, %r1;
cvt.rn.f32.s32  %f1, %r2;
cvt.rn.f32.s32  %f2, %r3;
cvt.rn.f32.s32  %f3, %r4;

// 反量化
sub.f32         %f0, %f0, %f_zp;     // - zero_point
mul.f32         %f0, %f0, %f_scale;  // * scale
// (对 f1, f2, f3 同理)
```

### 19.5 RoPE 位置编码

```ptx
// cos(θ * pos) 和 sin(θ * pos) 的近似
// θ = base^(-2i/d), 通常预计算后从 global memory 加载

// 旋转: [x0, x1] → [x0*cos - x1*sin, x0*sin + x1*cos]
ld.global.v2.f32    {%f_x0, %f_x1}, [%rd_input];
ld.global.v2.f32    {%f_cos, %f_sin}, [%rd_rope_table];

// x0_new = x0 * cos - x1 * sin
mul.f32     %f_t0, %f_x0, %f_cos;
fma.rn.f32  %f_out0, %f_x1, %f_sin_neg, %f_t0;   // fma(-sin, x1, x0*cos)

// x1_new = x0 * sin + x1 * cos
mul.f32     %f_t1, %f_x1, %f_cos;
fma.rn.f32  %f_out1, %f_x0, %f_sin, %f_t1;        // fma(sin, x0, x1*cos)

st.global.v2.f32    [%rd_output], {%f_out0, %f_out1};
```

---

## 20. 架构演进与指令可用性

### 20.1 各架构 PTX 指令可用性矩阵

| 指令 / 特性 | SM 8.0 (A100) | SM 8.6 (3090) | SM 8.9 (4090) | SM 9.0 (H100) | SM 10.0 (B200) | SM 12.0 (RTX 5090) |
|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| `mma.sync.m16n8k16.f16` | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| `mma.sync.m16n8k8.tf32` | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| `mma.sync.m16n8k32.s8` | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| `mma.sync.*.e4m3/e5m2` (FP8) | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ |
| `mma.sync` 扩展 (FP4/FP6) | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| `cp.async` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `redux.sync` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `wgmma` | ✗ | ✗ | ✗ | ✓ | ✓ | **✗** |
| TMA (`cp.async.bulk.tensor`) | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| `mbarrier` | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| Distributed Shared Memory | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| Thread Block Clusters | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| `fence.proxy.async` | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| `tcgen05` | ✗ | ✗ | ✗ | ✗ | ✓ | **✗** |
| Tensor Memory (tmem) | ✗ | ✗ | ✗ | ✗ | ✓ | **✗** |
| FP4 MMA (`mma` 或 `tcgen05`) | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |

> **关键观察：** SM 10.0 不支持 `mma.sync`，SM 12.0 不支持 `wgmma`/`tcgen05`。两者虽同为 Blackwell 品牌，但 Tensor Core 指令集**互不兼容**。SM 12.0 的 SASS 层面使用 `HMMA.16816` (与 Ampere 同源)，SM 10.0 使用 `UTCHMMA` (全新指令)。

### 20.2 Target 架构后缀说明

| 后缀 | 含义 | 示例 |
|------|------|------|
| (无) | 标准架构 | `sm_90` |
| `a` | 架构特定加速 (不保证前向兼容) | `sm_90a` (WGMMA, TMA 等 Hopper 专属) |
| `f` | Flash 执行 (嵌入式) | `sm_87f` (Orin) |

**Blackwell 的 target 分裂：**

| Target | GPU | Tensor Core 指令 |
|--------|-----|-----------------|
| `sm_100` / `sm_100a` | B200, B100 (数据中心) | `tcgen05` + TMEM |
| `sm_120` / `sm_120a` | RTX 5090, 5080, 5070 Ti (消费级) | `mma.sync` 扩展版 |

```ptx
// 使用 Hopper 专有指令必须指定 sm_90a
.target sm_90a
// 否则 wgmma, TMA 等指令在 ptxas 阶段会报错

// 数据中心 Blackwell: tcgen05 需要 sm_100a
.target sm_100a

// 消费级 Blackwell: 使用 mma.sync 扩展, 不支持 wgmma/tcgen05
.target sm_120a
// 以下指令在 sm_120 上会报错:
//   wgmma.fence          → "not supported on .target 'sm_120'"
//   tcgen05.mma          → "not supported on .target 'sm_120'"
//   mma with block scale → "not supported on .target 'sm_120'"
```

### 20.3 PTX 兼容性策略

```
// 方式 1: JIT 编译 (前向兼容)
// 编译为 PTX，运行时由驱动编译为目标 SASS
nvcc -arch=compute_80 -code=compute_80 kernel.cu   // 仅输出 PTX

// 方式 2: fatbin (即时 + 预编译)
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_90a,code=sm_90a \
     -gencode arch=compute_100a,code=sm_100a \
     -gencode arch=compute_120a,code=sm_120a \
     -gencode arch=compute_120a,code=compute_120a \  // PTX 后备
     kernel.cu

// 方式 3: NVRTC 运行时编译
// 在运行时检测 GPU 架构，选择合适的 PTX 代码路径
```

> **Blackwell 多路径编译：** 由于 SM 10.0 和 SM 12.0 的 Tensor Core 指令不兼容，必须分别编译。同一份使用 `wgmma` 的 Hopper kernel 源码**不能**直接编译到 sm_120 target。

---

## 附录 A: PTX 指令速查表 (LLM Kernel 高频)

| 类别 | 指令 | 典型用途 |
|------|------|---------|
| **加载** | `ld.global.v4.f32` | 128-bit 全局向量加载 |
| **存储** | `st.global.v4.f32` | 128-bit 全局向量存储 |
| **异步拷贝** | `cp.async.cg.shared.global` | Global→Shared 直接拷贝 |
| **TMA** | `cp.async.bulk.tensor.2d` | 硬件张量拷贝 (Hopper) |
| **FMA** | `fma.rn.f32` / `fma.rn.f16x2` | 融合乘加 |
| **MMA** | `mma.sync.aligned.m16n8k16` | Tensor Core 矩阵乘 |
| **WGMMA** | `wgmma.mma_async` | 异步 Warpgroup MMA |
| **Shuffle** | `shfl.sync.bfly.b32` | Warp 内蝴蝶归约 |
| **Redux** | `redux.sync.add.u32` | Warp 内整数归约 |
| **exp** | `ex2.approx.f32` | 快速 exp (Softmax) |
| **rsqrt** | `rsqrt.approx.f32` | LayerNorm 的 1/√x |
| **类型转换** | `cvt.rn.f16.f32` | FP32→FP16 下转 |
| **谓词** | `setp` / `selp` | 无分支条件逻辑 |
| **屏障** | `bar.sync` / `mbarrier` | 线程同步 |
| **地址转换** | `cvta.to.shared` | Generic→Shared 地址 |

---

## 附录 B: 内联 PTX 约束速查

| C++ 类型 | 约束符 | PTX 类型 | 宽度 |
|----------|--------|---------|------|
| `int` / `unsigned` / `uint32_t` | `"r"` | `.u32` / `.s32` / `.b32` | 32-bit |
| `long long` / `uint64_t` / 指针 | `"l"` | `.u64` / `.b64` | 64-bit |
| `float` | `"f"` | `.f32` | 32-bit |
| `double` | `"d"` | `.f64` | 64-bit |
| `short` / `uint16_t` / `half` | `"h"` | `.u16` / `.b16` | 16-bit |
| 编译时常量 | `"n"` | 立即数 | — |

---

## 附录 C: 常见错误与调试

| 错误 | 原因 | 解决 |
|------|------|------|
| `ptxas error: Feature not available on target` | 使用了目标架构不支持的指令 | 检查 `.target` 是否匹配，如 `wgmma` 需 `sm_90a` |
| `ptxas error: Misaligned address` | 向量加载地址未对齐 | 确保 `ld.v4.f32` 的地址是 16 字节对齐 |
| `error: unknown constraint` | 内联 PTX 约束符错误 | 检查类型与约束对应关系 (见附录 B) |
| `error: output constraint must start with '='` | 输出操作数缺少 `=` 或 `+` | 添加 `"=f"` 或 `"+r"` 前缀 |
| `warning: register spill to local memory` | 寄存器使用超出分配 | 减少活跃变量，使用 `.maxnreg`，或降低 tile 大小 |
| `mbarrier deadlock` | 异步屏障期望计数与实际不符 | 检查 `mbarrier.init` 的 count 与到达次数一致 |
| `wgmma shape mismatch` | shared memory 描述符布局与 wgmma 声明不匹配 | 验证 swizzle 模式和 leading dimension |

---

*本文档作为 LLM Kernel Agent 的 PTX ISA 技能参考。配合 `cuda-c-programming-guide.md`（编程模型）、`cuda-cpp-best-practices-guide.md`（优化策略）、`cuda-compiler-driver-nvcc.md`（编译构建）共同构成完整的 CUDA kernel 开发知识体系。*
