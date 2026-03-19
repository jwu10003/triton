# MMA / WMMA / WGMMA 编程接口指南

> 面向 LLM 高性能 Kernel 开发的 Tensor Core 编程接口深度解析
> 覆盖 WMMA C++ API、mma.sync PTX、WGMMA (Hopper)、tcgen05 (Blackwell)

---

## 目录

1. [接口总览与演进](#1-接口总览与演进)
2. [WMMA C++ API](#2-wmma-c-api)
3. [mma.sync PTX 指令](#3-mmasync-ptx-指令)
4. [ldmatrix — 协作矩阵加载](#4-ldmatrix--协作矩阵加载)
5. [WGMMA — Warpgroup MMA (Hopper)](#5-wgmma--warpgroup-mma-hopper)
6. [tcgen05 — 第五代 MMA (Blackwell)](#6-tcgen05--第五代-mma-blackwell)
7. [Fragment 寄存器映射详解](#7-fragment-寄存器映射详解)
8. [性能对比与选型](#8-性能对比与选型)
9. [常见问题与调试](#9-常见问题与调试)

---

## 1. 接口总览与演进

### 1.1 四代编程接口

| 接口 | 引入版本 | 层级 | 线程范围 | 执行模式 | 目标架构 |
|------|---------|------|---------|---------|---------|
| **WMMA** | CUDA 9.0 | C++ API | 32 线程 (Warp) | 同步 | SM 7.0+ |
| **mma.sync** | CUDA 10.1 | PTX 内联汇编 | 32 线程 (Warp) | 同步 | SM 7.0+ |
| **wgmma** | CUDA 12.0 | PTX 内联汇编 | 128 线程 (Warpgroup) | 异步 | SM 9.0 (仅 Hopper) |
| **tcgen05** | CUDA 12.4+ | PTX 内联汇编 | 单线程发射 | 异步 | SM 10.0 (仅数据中心 Blackwell) |

### 1.2 核心差异

```
WMMA:     简单   →  opaque fragment  →  编译器决定布局  →  性能 ~60% cuBLAS
mma.sync: 中等   →  显式寄存器映射    →  开发者控制布局  →  性能 ~95% cuBLAS
wgmma:    复杂   →  Shared Mem 描述符 →  异步 + 流水线   →  性能 ~100% cuBLAS
tcgen05:  最复杂 →  TMEM + 跨 2SM    →  单线程指令发射  →  性能 Blackwell 峰值
```

### 1.3 选择建议

```
快速原型 / 学习                     → WMMA
高性能 Ampere/Ada kernel           → mma.sync
Hopper 生产级 kernel               → wgmma (via CUTLASS/CuTe)
RTX 5090 (SM 12.0)                → mma.sync 扩展版 (不支持 wgmma/tcgen05)
数据中心 Blackwell (SM 10.0)        → tcgen05 (via CUTLASS 4.x)
需跨架构可移植                      → WMMA 或 CUTLASS 抽象层
```

---

## 2. WMMA C++ API

### 2.1 核心概念

WMMA 提供 4 个核心操作：

| 操作 | 函数 | 作用 |
|------|------|------|
| **声明 Fragment** | `wmma::fragment<...>` | 在 Warp 的寄存器中分配矩阵存储 |
| **加载** | `wmma::load_matrix_sync()` | 从 Global/Shared Memory 加载到 Fragment |
| **计算** | `wmma::mma_sync()` | 执行 D = A×B + C |
| **存储** | `wmma::store_matrix_sync()` | 从 Fragment 写回 Global/Shared Memory |
| **填充** | `wmma::fill_fragment()` | 将 Fragment 初始化为常量 |

### 2.2 Fragment 模板参数

```cpp
#include <mma.h>
using namespace nvcuda;

wmma::fragment<
    Use,        // matrix_a | matrix_b | accumulator
    M, N, K,    // MMA 维度 (如 16, 16, 16)
    T,          // 数据类型 (half, __nv_bfloat16, int8_t, ...)
    Layout      // row_major | col_major (仅 matrix_a/b 需要)
> frag;
```

**Fragment 是不透明的 (Opaque)：** 矩阵元素如何分布在 Warp 的 32 个线程中由编译器决定，程序员不应假设映射关系。

### 2.3 支持的 MMA 形状

| A/B 类型 | M×N×K | 累加器类型 | 最低 SM |
|----------|-------|----------|--------|
| `half` | 16×16×16 | `half` / `float` | 7.0 |
| `half` | 32×8×16 | `half` / `float` | 7.0 |
| `half` | 8×32×16 | `half` / `float` | 7.0 |
| `__nv_bfloat16` | 16×16×16 | `float` | 8.0 |
| `int8_t` / `uint8_t` | 16×16×16 | `int32_t` | 7.5 |
| `int8_t` / `uint8_t` | 32×8×16 | `int32_t` | 7.5 |
| `int8_t` / `uint8_t` | 8×32×16 | `int32_t` | 7.5 |
| `experimental::precision::b1` | 8×8×128 | `int32_t` | 7.5 |

### 2.4 完整 HGEMM 示例

```cpp
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void hgemm_wmma(
    const half* __restrict__ A,  // [M, K]
    const half* __restrict__ B,  // [K, N]
    float*      __restrict__ C,  // [M, N]
    int M, int N, int K)
{
    // 每个 Warp 负责 C 的一个 16×16 tile
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * WMMA_M;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * WMMA_N;

    // 声明 fragment
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累加器为 0
    wmma::fill_fragment(c_frag, 0.0f);

    // K 维度循环
    for (int k = 0; k < K; k += WMMA_K) {
        // 从 Global Memory 加载 A 和 B 的 tile
        wmma::load_matrix_sync(a_frag, A + warpM * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warpN, N);

        // Tensor Core MMA
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 存回 C
    wmma::store_matrix_sync(C + warpM * N + warpN, c_frag, N,
                            wmma::mem_row_major);
}
```

### 2.5 Shared Memory 优化版

```cpp
__global__ void hgemm_wmma_smem(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ C,
    int M, int N, int K)
{
    __shared__ half As[BLOCK_M][WMMA_K];
    __shared__ half Bs[WMMA_K][BLOCK_N];

    int warpId = threadIdx.y;
    int laneId = threadIdx.x;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        // 协作加载 A, B tile 到 Shared Memory (合并访问)
        // ... (省略加载逻辑)
        __syncthreads();

        // 从 Shared Memory 加载 fragment
        wmma::load_matrix_sync(a_frag,
            &As[warpId * WMMA_M][0], WMMA_K);
        wmma::load_matrix_sync(b_frag,
            &Bs[0][warpId * WMMA_N], BLOCK_N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(
        C + (blockIdx.y * BLOCK_M + warpId * WMMA_M) * N
          + blockIdx.x * BLOCK_N,
        c_frag, N, wmma::mem_row_major);
}
```

### 2.6 WMMA 的优缺点

| 优点 | 缺点 |
|------|------|
| C++ 级 API，简单直观 | Fragment 布局不透明，无法精确控制 |
| 跨架构兼容 (SM 7.0–9.0) | 性能仅达 cuBLAS 的 ~50–80% |
| 编译器自动处理寄存器映射 | 加载路径不可定制 (无法用 `ldmatrix`) |
| 无需内联 PTX | 不支持 FP8、wgmma 等新特性 |
| 适合教学和快速原型 | 底层实际由 2 条 HMMA 16×8×16 执行 |

---

## 3. mma.sync PTX 指令

### 3.1 指令格式

```
mma.sync.aligned.m{M}n{N}k{K}.{Alayout}.{Blayout}.{Dtype}.{Atype}.{Btype}.{Ctype}
    {D_regs}, {A_regs}, {B_regs}, {C_regs};
```

示例：
```
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%f0,%f1,%f2,%f3},        // D: 4×f32
    {%r0,%r1,%r2,%r3},        // A: 4×b32 (8×f16)
    {%r4,%r5},                // B: 2×b32 (4×f16)
    {%f4,%f5,%f6,%f7};        // C: 4×f32
```

### 3.2 支持的所有形状与类型

#### 浮点类型

| A/B Type | Shape | A Regs/Thread | B Regs/Thread | C/D Regs/Thread | 最低 SM |
|----------|-------|-------------|-------------|---------------|--------|
| **f16** | m8n8k4 | 2×b32 | 2×b32 | 4×f16x2 或 8×f32 | 7.0 |
| **f16** | m16n8k8 | 2×b32 | 1×b32 | 2×f16x2 或 4×f32 | 7.5 |
| **f16** | m16n8k16 | 4×b32 | 2×b32 | 2×f16x2 或 4×f32 | 8.0 |
| **bf16** | m16n8k8 | 2×b32 | 1×b32 | 4×f32 | 8.0 |
| **bf16** | m16n8k16 | 4×b32 | 2×b32 | 4×f32 | 8.0 |
| **f64** | m8n8k4 | 1×f64 | 1×f64 | 2×f64 | 8.0 |

#### 整数类型

| A/B Type | Shape | A Regs/Thread | B Regs/Thread | C/D Regs/Thread | 最低 SM |
|----------|-------|-------------|-------------|---------------|--------|
| **s8/u8** | m8n8k16 | 1×b32 | 1×b32 | 2×s32 | 7.5 |
| **s8/u8** | m16n8k16 | 2×b32 | 1×b32 | 4×s32 | 8.0 |
| **s8/u8** | m16n8k32 | 4×b32 | 2×b32 | 4×s32 | 8.0 |
| **s4/u4** | m8n8k32 | 1×b32 | 1×b32 | 2×s32 | 7.5 |
| **s4/u4** | m16n8k32 | 2×b32 | 1×b32 | 4×s32 | 8.0 |
| **s4/u4** | m16n8k64 | 4×b32 | 2×b32 | 4×s32 | 8.0 |

#### FP8 类型 (SM 8.9+)

| A/B Type | Shape | A Regs/Thread | B Regs/Thread | C/D Regs/Thread | 最低 SM |
|----------|-------|-------------|-------------|---------------|--------|
| **e4m3/e5m2** | m16n8k16 | 2×b32 | 1×b32 | 4×f32 | 8.9 |
| **e4m3/e5m2** | m16n8k32 | 4×b32 | 2×b32 | 4×f32 | 8.9 |

#### Block-Scaled MMA 类型 (SM 12.0+, `kind::mxf8f6f4` / `kind::mxf4nvf4`)

SM 12.0 (RTX 5090) 引入带 **Block Scaling** 的 `mma.sync` 扩展，支持 FP8/FP6/FP4 数据类型 + 每 16/32 元素一个缩放因子：

| A/B Type | Shape | A Regs | B Regs | C/D Regs | SF_A Regs | SF_B Regs | Kind | Scale Type | 最低 SM |
|----------|-------|--------|--------|----------|-----------|-----------|------|-----------|--------|
| **e4m3/e5m2** | m16n8k32 | 4×b32 | 2×b32 | 4×f32 | 1×u8 | 1×u8 | `mxf8f6f4` | `ue8m0` | 12.0 |
| **e3m2/e2m3** | m16n8k32 | 4×b32 | 2×b32 | 4×f32 | 1×u8 | 1×u8 | `mxf8f6f4` | `ue8m0` | 12.0 |
| **e2m1** (FP4) | m16n8k64 | 4×b32 | 2×b32 | 4×f32 | 1×u8 | 1×u8 | `mxf4nvf4` | `ue4m3` | 12.0 |
| **e2m1** (FP4) | m16n8k64 | 4×b32 | 2×b32 | 4×f32 | 1×u8 | 1×u8 | `mxf4nvf4` | `ue8m0` | 12.0 |

> SF_A / SF_B = Scale Factor 寄存器，每线程 1 字节 (uint8_t)。详见 [3.5 Block Scaling](#35-block-scaling-for-mmasync)。

### 3.3 内联 PTX 调用模式

#### m16n8k16 FP16→FP32 (Ampere 主力)

```cpp
__device__ void mma_m16n8k16_f16_f32(
    float d[4],              // 输出累加器
    const uint32_t a[4],     // A fragment (4 × b32 = 8 × f16)
    const uint32_t b[2],     // B fragment (2 × b32 = 4 × f16)
    const float c[4])        // 输入累加器
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

#### m16n8k16 BF16→FP32

```cpp
__device__ void mma_m16n8k16_bf16_f32(
    float d[4],
    const uint32_t a[4],     // 4 × b32 = 8 × bf16
    const uint32_t b[2],     // 2 × b32 = 4 × bf16
    const float c[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
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

#### m16n8k32 INT8→INT32

```cpp
__device__ void mma_m16n8k32_s8_s32(
    int32_t d[4],
    const uint32_t a[4],     // 4 × b32 = 16 × s8
    const uint32_t b[2],     // 2 × b32 = 8 × s8
    const int32_t c[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}
```

### 3.4 mma.sync 的优势

| 特性 | 说明 |
|------|------|
| **显式寄存器映射** | 程序员完全控制数据在线程间的分布 |
| **配合 ldmatrix** | 可消除加载时的 Bank Conflict |
| **灵活的形状选择** | m16n8k16 是主力，也可用 m16n8k8 等 |
| **更高性能** | 相比 WMMA 达 cuBLAS 的 91–98% |
| **精确的指令调度** | 可精确控制 MMA 与数据加载的交错 |

### 3.5 Block Scaling for mma.sync (SM 12.0+)

#### 3.5.1 概述

SM 12.0 (RTX 5090) 引入 **Block-Scaled MMA**，在 `mma.sync` 指令中集成了逐块缩放因子 (Scale Factor)，实现低精度 (FP8/FP6/FP4) 计算时的精度补偿：

```
D = (A × SF_A) × (B × SF_B) + C
```

- **SF_A / SF_B** 是缩放因子矩阵，每 `VEC_SIZE` 个 K 维度元素共享 1 个缩放因子
- 缩放因子类型：`ue8m0` (8-bit 无符号, 纯指数) 或 `ue4m3` (7-bit 无符号)
- 硬件在 Tensor Core 内部自动执行乘缩放操作，无额外指令开销

#### 3.5.2 PTX 指令格式

```
mma.sync.aligned.kind::{KIND}.block_scale.scale_vec::{VEC}X
    .m{M}n{N}k{K}.row.col.f32.{Atype}.{Btype}.f32.{SFtype}
    {D_regs}, {A_regs}, {B_regs}, {C_regs}, {SF_A_reg}, {SF_B_reg};
```

| 参数 | 说明 |
|------|------|
| `KIND` | `mxf8f6f4` (FP8/FP6) 或 `mxf4nvf4` (NVFP4) |
| `VEC` | `1` (每 32 元素 1 个 SF) 或 `4` (每 16 元素 1 个 SF, 仅 FP4) |
| `Atype / Btype` | `e4m3`, `e5m2`, `e3m2`, `e2m3`, `e2m1` |
| `SFtype` | `ue8m0` 或 `ue4m3` |

#### 3.5.3 Scale Factor 数据格式

| 格式 | 位宽 | 结构 | 说明 |
|------|------|------|------|
| **ue8m0** | 8-bit | 8 exp + 0 mantissa (无符号) | 纯指数, 表示 2^(val-127)，不支持 NaN/Inf |
| **ue4m3** | 7-bit | 4 exp + 3 mantissa (无符号, MSB 补零) | 更高精度的缩放因子, `.b8` 类型存储 |

> `ue8m0` 只能表示 2 的幂次方缩放值，适合 FP8；`ue4m3` 可表示非 2^n 值，适合精度要求更高的 FP4。

#### 3.5.4 scale_vec 与缩放粒度

| `scale_vec` | VEC_SIZE | K 维度每组元素数 | SF_A 尺寸 | SF_B 尺寸 | 适用场景 |
|-------------|----------|---------------|----------|----------|---------|
| `1X` | 32 | 32 | M × ⌈K/32⌉ | N × ⌈K/32⌉ | FP8 (e4m3/e5m2), FP6 |
| `4X` | 16 | 16 | M × ⌈K/16⌉ | N × ⌈K/16⌉ | FP4 (e2m1), 更细粒度 |

> NVFP4 的 "两级缩放" 中，`ue4m3` 微块缩放 (每 16 值) 对应此处的 `scale_vec::4X`。

#### 3.5.5 Scale Factor 寄存器映射

每个线程持有 **1 字节** 的缩放因子 (uint8_t)，32 个线程协作覆盖整个 MMA tile 的缩放：

```
SF_A (scale_vec::1X, m16n8k32):
  每线程 1 个 ue8m0 值, 对应 A 矩阵中该线程负责的行的 K 维度缩放因子
  SF_A[groupID] 和 SF_A[groupID+8] 分别对应上/下半 M 的缩放

SF_B (scale_vec::1X, m16n8k32):
  每线程 1 个 ue8m0 值, 对应 B 矩阵中该线程负责的列的 K 维度缩放因子
  SF_B[groupID] 对应 N 维度的缩放

SF_A (scale_vec::4X, m16n8k64 FP4):
  每线程 1 个 ue4m3 值, 打包了多个 16 元素组的缩放因子
```

#### 3.5.6 Scale Factor 内存布局

为保证 Tensor Core 快速访问，缩放因子需要特殊的内存布局 (非简单行优先)：

```
逻辑形状 (row-major): SF_A[M, K // VEC_SIZE]

实际存储布局 (packed block layout):
  SF_A[M // 128, K // VEC_SIZE // 4, 32, 4, 4]

从 PyTorch Tensor 转换:
  sf_a.reshape(M//128, 4, 32, K//VEC_SIZE//4, 4)
      .permute(0, 3, 2, 1, 4)
      .contiguous()
```

#### 3.5.7 内联 PTX 示例

**FP8 Block-Scaled MMA (m16n8k32, SM 12.0)：**

```cpp
__device__ void mma_m16n8k32_fp8_block_scaled(
    float d[4],              // 输出累加器
    const uint32_t a[4],     // A fragment (4 × b32 = 16 × e4m3)
    const uint32_t b[2],     // B fragment (2 × b32 = 8 × e4m3)
    const float c[4],        // 输入累加器
    uint8_t sf_a,            // A 的缩放因子 (ue8m0)
    uint8_t sf_b)            // B 的缩放因子 (ue8m0)
{
    uint32_t sf_a_32 = sf_a;
    uint32_t sf_b_32 = sf_b;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "{%14}, {%15};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sf_a_32), "r"(sf_b_32)
    );
}
```

**NVFP4 Block-Scaled MMA (m16n8k64, SM 12.0)：**

```cpp
__device__ void mma_m16n8k64_fp4_block_scaled(
    float d[4],              // 输出累加器
    const uint32_t a[4],     // A fragment (4 × b32 = 32 × e2m1)
    const uint32_t b[2],     // B fragment (2 × b32 = 16 × e2m1)
    const float c[4],        // 输入累加器
    uint8_t sf_a,            // A 的缩放因子 (ue4m3)
    uint8_t sf_b)            // B 的缩放因子 (ue4m3)
{
    uint32_t sf_a_32 = sf_a;
    uint32_t sf_b_32 = sf_b;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "{%14}, {%15};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sf_a_32), "r"(sf_b_32)
    );
}
```

#### 3.5.8 Block Scaling vs 传统量化

| 方式 | 缩放粒度 | 硬件支持 | 延迟 | 精度 |
|------|---------|---------|------|------|
| Per-tensor scaling | 整个张量 1 个 scale | 软件 | 无 | 低 |
| Per-channel scaling | 每行/列 1 个 scale | 软件 | 额外乘法指令 | 中 |
| **Block scaling (mma.sync)** | 每 16/32 元素 1 个 scale | **硬件集成** | **零** (融合在 MMA 中) | 高 |

---

## 4. ldmatrix — 协作矩阵加载

### 4.1 指令作用

`ldmatrix` 是 NVIDIA 为 Tensor Core 设计的**协作加载指令**。它在 Warp 级别从 Shared Memory 加载矩阵数据到寄存器，自动将数据分配到 `mma.sync` 期望的线程-寄存器映射中。

```ptx
ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r0,%r1,%r2,%r3}, [%r_smem_addr];
```

### 4.2 为什么需要 ldmatrix

`mma.sync` 的 Fragment 布局是**不连续的**——矩阵元素在 32 个线程间的分布方式复杂。如果手动用 `ld.shared` 逐元素加载，需要复杂的索引计算且容易产生 Bank Conflict。

`ldmatrix` 让每个线程提供一个 Shared Memory 地址（指向 8 个 half 元素），硬件自动将数据交叉分发到正确的线程和寄存器。

### 4.3 变体

| 指令 | 加载矩阵数 | 每线程寄存器 | 总数据量 |
|------|----------|-----------|---------|
| `ldmatrix.x1` | 1 个 8×8 | 1 × b32 | 128 B |
| `ldmatrix.x2` | 2 个 8×8 | 2 × b32 | 256 B |
| `ldmatrix.x4` | 4 个 8×8 | 4 × b32 | 512 B |

### 4.4 内联 PTX 用法

```cpp
// 加载 A fragment: ldmatrix.x4 (m16n8k16 的 A 需要 4 × b32/thread)
__device__ void ldmatrix_x4(uint32_t dst[4], const void* smem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(smem_addr)
    );
}

// 加载 B fragment: ldmatrix.x2 (m16n8k16 的 B 需要 2 × b32/thread)
__device__ void ldmatrix_x2(uint32_t dst[2], const void* smem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(dst[0]), "=r"(dst[1])
        : "r"(smem_addr)
    );
}

// 转置加载 (用于 B 矩阵列优先布局)
__device__ void ldmatrix_x2_trans(uint32_t dst[2], const void* smem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(dst[0]), "=r"(dst[1])
        : "r"(smem_addr)
    );
}
```

### 4.5 ldmatrix 与 Bank Conflict

`ldmatrix` 按 Phase (每 Phase 8 个线程) 执行，每线程访问 16 字节 (4 个连续 Bank)。如果 Shared Memory 未做 Swizzle，同一 Phase 的 8 个线程可能访问同一 SuperBank → **8-way Bank Conflict**。

**必须配合 Swizzle** 使用以消除冲突（参见 `conflict-free-accesses.md`）。

### 4.6 ldmatrix 地址计算

每个线程提供的地址指向 Shared Memory 中一行 8 个 half 的起始位置。对于 m16n8k16 的 A 矩阵 (16×16, row-major)：

```cpp
// 线程 lane_id 提供的 smem 地址
// ldmatrix.x4 加载 4 个 8×8 矩阵 (合计 16×16)
int row = lane_id % 16;  // 实际映射取决于 ldmatrix 规范
const half* addr = &smem_A[row][0];  // 指向该行的 8 个 half
```

具体映射需参考 PTX ISA 文档中 `ldmatrix` 的线程-地址对应表。

---

## 5. WGMMA — Warpgroup MMA (Hopper)

### 5.1 概述

WGMMA (Warp Group Matrix Multiply-Accumulate) 是 Hopper (SM 9.0) 引入的新一代 Tensor Core 接口：

| 特性 | mma.sync (Ampere) | wgmma (Hopper) |
|------|-------------------|----------------|
| 参与线程 | 32 (1 Warp) | 128 (4 Warps = Warpgroup) |
| 执行模式 | 同步 | **异步** |
| A 操作数来源 | 寄存器 | **Shared Memory** 或 寄存器 |
| B 操作数来源 | 寄存器 | **Shared Memory** (必须) |
| C/D 累加器 | 寄存器 | 寄存器 |
| 最大 tile | m16n8k16 | **m64n256k16** |

### 5.2 指令格式

```ptx
wgmma.mma_async.sync.aligned.m64n{N}k{K}.{Dtype}.{Atype}.{Btype}
    {D_regs},
    {A_desc_or_regs},
    {B_desc},
    scale_D, imm_scale_A, imm_scale_B, imm_neg_A, imm_neg_B;
```

### 5.3 支持的形状

| A/B Type | M | N (可选) | K |
|----------|---|---------|---|
| f16 / bf16 | 64 | 8,16,24,...,256 | 16 |
| e4m3 / e5m2 | 64 | 8,16,...,256 | 32 |
| s8 / u8 | 64 | 8,16,...,256 | 32 |

> M 固定为 64（Warpgroup 的自然分片粒度）。N 的最大值为 256。

### 5.4 Shared Memory 描述符 (Matrix Descriptor)

WGMMA 不直接传递 Shared Memory 指针，而是使用 **64-bit 描述符**：

```
描述符结构 (uint64_t):
├── Start Address (14 bits): SMEM 起始地址
├── Leading Dimension Byte Offset (14 bits): 主维度步长
├── Stride Dimension Byte Offset (14 bits): 次维度步长
├── Base Offset (3 bits): 基础偏移
├── Layout Type (1 bit): Swizzle 模式
└── Reserved
```

**在 CUTLASS/CuTe 中，** 描述符通过 `make_smem_desc()` 或 `cute::make_tma_copy()` 自动生成。

### 5.5 操作数模式

| 模式 | A 来源 | B 来源 | 用途 | CUTLASS 记法 |
|------|--------|--------|------|-------------|
| **SS** | Shared Memory 描述符 | Shared Memory 描述符 | 标准 GEMM | `MMA_64xNxK_..._SS` |
| **RS** | 寄存器 | Shared Memory 描述符 | A 在寄存器中已有时 | `MMA_64xNxK_..._RS` |

### 5.6 同步原语

WGMMA 异步执行需要精确的同步：

```cpp
// Step 1: Fence — 在首次 wgmma 前，确保寄存器写入可见
warpgroup_fence_operand(acc);  // → wgmma.fence.sync.aligned;

// Step 2: 发射 WGMMA 指令
wgmma.mma_async ...;          // 异步发射

// Step 3: Commit — 提交当前批次
warpgroup_commit_batch();       // → wgmma.commit_group.sync.aligned;

// Step 4: Wait — 等待完成
warpgroup_wait<0>();            // → wgmma.wait_group.sync.aligned 0;

// 注意: 连续多个 wgmma 调用之间不需要额外 fence
// (硬件对同形状、同累加器的连续 MMA 有特殊优化)
```

### 5.7 Pipeline 模式 (Producer-Consumer)

Hopper 上最高效的模式是 **Warp 专化 (Warp Specialization)**：

```
Warpgroup 0 (Producer):
  循环: TMA 加载 A[k], B[k] → smem_buf[k % N_STAGES]
        mbarrier.arrive(tx_bytes)

Warpgroup 1 (Consumer):
  循环: mbarrier.try_wait(buf[k])
        wgmma.fence
        wgmma.mma_async (A_desc[k], B_desc[k], acc)
        wgmma.commit_group
        wgmma.wait_group<0>

→ 数据加载与计算完全重叠
```

### 5.8 SMEM 布局约束

WGMMA 对 Shared Memory 布局有严格要求：

| 约束 | 值 |
|------|-----|
| M 维度 | 必须是 64 的倍数 |
| N 维度 | 必须是所选 N 的倍数 (8 的倍数) |
| K 维度 | 必须是 16 (FP16/BF16) 或 32 (FP8/INT8) |
| Swizzle | 必须使用 128B swizzle 模式 |
| 对齐 | 描述符地址必须按 swizzle 周期对齐 |

### 5.9 内联 PTX 示例 (m64n16k16 BF16→F32)

```cpp
__device__ void wgmma_m64n16k16_bf16_f32(
    float acc[8],           // 每线程 8 个 f32 累加器
    uint64_t desc_a,        // A 的 SMEM 描述符
    uint64_t desc_b,        // B 的 SMEM 描述符
    int scale_D)            // 0: 清零累加器, 1: 累加
{
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "%8, %9, "
        "%10, 1, 1, 0, 0;\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),
          "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7])
        : "l"(desc_a), "l"(desc_b),
          "r"(scale_D)
    );
}
```

---

## 6. tcgen05 — 第五代 MMA (Blackwell 数据中心, SM 10.0)

> **注意：** tcgen05 / TMEM / CTA Pair 仅在**数据中心级 Blackwell (SM 10.0)** 上可用 (B200/B100 等)。消费级 Blackwell RTX 5090 (SM 12.0) **既不支持 tcgen05 也不支持 wgmma**，其 Tensor Core 使用 `mma.sync` 扩展版 (Ampere 编程模型 + FP8/FP4/FP6 新类型)。

### 6.1 核心变化

| 特性 | wgmma (Hopper) | tcgen05 (Blackwell) |
|------|----------------|-------------------|
| 发射线程 | 128 (Warpgroup) | **单线程** |
| A 来源 | SMEM / 寄存器 | **SMEM** |
| B 来源 | SMEM | **SMEM** |
| C/D 累加器 | 寄存器 | **Tensor Memory (TMEM)** |
| 跨 SM | 不支持 | **CTA Pair (2 SM 协作)** |
| 新精度 | — | FP4, FP6 |

### 6.2 Tensor Memory (TMEM) 管理

```cpp
// 分配 TMEM (必须由单个 Warp 调用)
uint32_t tmem_addr;
asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
    : "=r"(tmem_addr) : "r"(num_columns));

// MMA 计算 (结果写入 TMEM)
asm volatile("tcgen05.mma.cta_group::1.kind::f16 ..."
    : : "r"(tmem_addr), "r"(smem_a_desc), "r"(smem_b_desc), ...);

// 从 TMEM 读取结果到寄存器 (后处理)
asm volatile("tcgen05.ld.sync.aligned.16x256b.x64.b32 %0, [%1];\n"
    : "=r"(reg_val) : "r"(tmem_addr));

// 释放 TMEM
asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
    :: "r"(tmem_addr), "r"(num_columns));
```

### 6.3 CTA Group 模式

| 参数 | 单 CTA | 双 CTA (CTA Pair) |
|------|--------|-------------------|
| `cta_group::1` | 1 个 SM 的 TC | 2 个 SM 的 TC 协作 |
| `cta_group::2` | — | M 维度翻倍 |
| Shared Memory | 1 个 SM | 等效 2 个 SM (各提供一半 A) |

### 6.4 支持的数据类型

| A/B Type | Accumulator | K 值 | 备注 |
|----------|-------------|------|------|
| f16 / bf16 | f32 / f16 | 16 | 与 Hopper 类似 |
| e4m3 / e5m2 | f32 | 32 | FP8 |
| e3m2 (FP6) | f32 | 32 | Blackwell 新增 |
| e2m1 (FP4) | f32 | 64 | Blackwell 新增，需 Block Scaling |

---

## 7. Fragment 寄存器布局与线程映射详解

> **本节是 Tensor Core 编程的核心参考。** 理解每个线程持有矩阵哪些元素，是正确使用 `mma.sync`、`ldmatrix`、手动构造 Fragment 的基础。

### 7.1 通用映射规则与约定

#### 线程分组

所有 `mma.sync.m16n8k*` 指令使用相同的线程分组方式：

```
groupID        = lane_id >> 2      // lane_id / 4, 取值 0..7 (共 8 组)
tid_in_group   = lane_id & 3       // lane_id % 4, 取值 0..3 (组内 4 个线程)
```

> `lane_id` = `threadIdx.x % 32`，即 Warp 内线程编号 (0..31)。

#### 寄存器打包规则

不同位宽的数据在 32-bit 寄存器中的打包方式：

| 数据类型 | 位宽 | 每寄存器打包数 | 打包方式 |
|----------|------|-------------|---------|
| FP16 / BF16 | 16-bit | **2** | `{low_f16, high_f16}` packed in 1×b32 |
| INT8 / FP8 | 8-bit | **4** | `{b0, b1, b2, b3}` packed in 1×b32 |
| FP32 (累加器) | 32-bit | **1** | 1×f32 per register |
| FP64 | 64-bit | **1** | 1×f64 per register (64-bit reg) |
| INT32 (累加器) | 32-bit | **1** | 1×s32 per register |

#### 矩阵布局约定

- **A 矩阵**: `row-major` (行优先，`.row`)
- **B 矩阵**: `col-major` (列优先，`.col`)
- **C/D 矩阵**: 由线程映射隐式决定 (无布局参数)
- 所有可视化图中 A 的维度为 m×k，B 为 k×n，C/D 为 m×n

---

### 7.2 m16n8k16 — FP16 / BF16 (16-bit 输入, SM 8.0+)

**Ampere 架构主力指令，LLM Kernel 中最常用的 MMA 形状。**

#### 寄存器分配

| 操作数 | 矩阵尺寸 | 元素数/线程 | 寄存器数/线程 | 寄存器类型 |
|--------|---------|-----------|-------------|----------|
| A | 16×16 (m×k) | 8 × f16 | 4 × b32 | 每 reg 含 2 × f16 |
| B | 16×8 (k×n) | 4 × f16 | 2 × b32 | 每 reg 含 2 × f16 |
| C/D (FP16 累加) | 16×8 (m×n) | 4 × f16 | 2 × b32 | 每 reg 含 2 × f16 |
| C/D (FP32 累加) | 16×8 (m×n) | 4 × f32 | 4 × f32 | 每 reg 含 1 × f32 |

#### A Fragment 映射公式 (16×16, row-major)

```
groupID      = lane_id >> 2    (0..7)
t            = lane_id & 3     (0..3)

a[0]:  A[groupID,     t*2],     A[groupID,     t*2+1]      ← m∈[0,7],  k∈[0,7]
a[1]:  A[groupID,     t*2+8],   A[groupID,     t*2+9]      ← m∈[0,7],  k∈[8,15]
a[2]:  A[groupID+8,   t*2],     A[groupID+8,   t*2+1]      ← m∈[8,15], k∈[0,7]
a[3]:  A[groupID+8,   t*2+8],   A[groupID+8,   t*2+9]      ← m∈[8,15], k∈[8,15]
```

**逻辑结构：** 16×16 矩阵分为 4 个 8×8 子块，每个子块对应 1 个寄存器：

```
A[16×16] = ┌─────────┬─────────┐
            │ a[0]    │ a[1]    │    m ∈ [0,7]
            │ k=[0,7] │ k=[8,15]│
            ├─────────┼─────────┤
            │ a[2]    │ a[3]    │    m ∈ [8,15]
            │ k=[0,7] │ k=[8,15]│
            └─────────┴─────────┘
```

#### A Fragment 可视化 (每格 = 1 个 b32 寄存器 = 2 个 f16)

```
A[16×16] row-major, 标记: T{lane_id}.a[reg_idx]

             k=0,1    k=2,3    k=4,5    k=6,7   ║ k=8,9    k=10,11  k=12,13  k=14,15
           ┌────────┬────────┬────────┬────────╫────────┬────────┬────────┬────────┐
  m=0      │T0 .a[0]│T1 .a[0]│T2 .a[0]│T3 .a[0]║T0 .a[1]│T1 .a[1]│T2 .a[1]│T3 .a[1]│
  m=1      │T4 .a[0]│T5 .a[0]│T6 .a[0]│T7 .a[0]║T4 .a[1]│T5 .a[1]│T6 .a[1]│T7 .a[1]│
  m=2      │T8 .a[0]│T9 .a[0]│T10.a[0]│T11.a[0]║T8 .a[1]│T9 .a[1]│T10.a[1]│T11.a[1]│
  m=3      │T12.a[0]│T13.a[0]│T14.a[0]│T15.a[0]║T12.a[1]│T13.a[1]│T14.a[1]│T15.a[1]│
  m=4      │T16.a[0]│T17.a[0]│T18.a[0]│T19.a[0]║T16.a[1]│T17.a[1]│T18.a[1]│T19.a[1]│
  m=5      │T20.a[0]│T21.a[0]│T22.a[0]│T23.a[0]║T20.a[1]│T21.a[1]│T22.a[1]│T23.a[1]│
  m=6      │T24.a[0]│T25.a[0]│T26.a[0]│T27.a[0]║T24.a[1]│T25.a[1]│T26.a[1]│T27.a[1]│
  m=7      │T28.a[0]│T29.a[0]│T30.a[0]│T31.a[0]║T28.a[1]│T29.a[1]│T30.a[1]│T31.a[1]│
           ╠════════╪════════╪════════╪════════╬════════╪════════╪════════╪════════╣
  m=8      │T0 .a[2]│T1 .a[2]│T2 .a[2]│T3 .a[2]║T0 .a[3]│T1 .a[3]│T2 .a[3]│T3 .a[3]│
  m=9      │T4 .a[2]│T5 .a[2]│T6 .a[2]│T7 .a[2]║T4 .a[3]│T5 .a[3]│T6 .a[3]│T7 .a[3]│
  m=10     │T8 .a[2]│T9 .a[2]│T10.a[2]│T11.a[2]║T8 .a[3]│T9 .a[3]│T10.a[3]│T11.a[3]│
  m=11     │T12.a[2]│T13.a[2]│T14.a[2]│T15.a[2]║T12.a[3]│T13.a[3]│T14.a[3]│T15.a[3]│
  m=12     │T16.a[2]│T17.a[2]│T18.a[2]│T19.a[2]║T16.a[3]│T17.a[3]│T18.a[3]│T19.a[3]│
  m=13     │T20.a[2]│T21.a[2]│T22.a[2]│T23.a[2]║T20.a[3]│T21.a[3]│T22.a[3]│T23.a[3]│
  m=14     │T24.a[2]│T25.a[2]│T26.a[2]│T27.a[2]║T24.a[3]│T25.a[3]│T26.a[3]│T27.a[3]│
  m=15     │T28.a[2]│T29.a[2]│T30.a[2]│T31.a[2]║T28.a[3]│T29.a[3]│T30.a[3]│T31.a[3]│
           └────────┴────────┴────────┴────────╨────────┴────────┴────────┴────────┘
```

#### B Fragment 映射公式 (16×8, col-major)

```
b[0]:  B[t*2,   groupID],  B[t*2+1, groupID]       ← k∈[0,7],  n=groupID
b[1]:  B[t*2+8, groupID],  B[t*2+9, groupID]       ← k∈[8,15], n=groupID
```

#### B Fragment 可视化 (每格 = 1 个 b32 寄存器 = 2 个 f16)

```
B[16×8] col-major, 标记: T{lane_id}.b[reg_idx]
(每格覆盖相邻 2 行, groupID 决定列)

            n=0      n=1      n=2      n=3      n=4      n=5      n=6      n=7
          ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
 k=0,1    │T0 .b[0]│T4 .b[0]│T8 .b[0]│T12.b[0]│T16.b[0]│T20.b[0]│T24.b[0]│T28.b[0]│
 k=2,3    │T1 .b[0]│T5 .b[0]│T9 .b[0]│T13.b[0]│T17.b[0]│T21.b[0]│T25.b[0]│T29.b[0]│
 k=4,5    │T2 .b[0]│T6 .b[0]│T10.b[0]│T14.b[0]│T18.b[0]│T22.b[0]│T26.b[0]│T30.b[0]│
 k=6,7    │T3 .b[0]│T7 .b[0]│T11.b[0]│T15.b[0]│T19.b[0]│T23.b[0]│T27.b[0]│T31.b[0]│
          ╠════════╪════════╪════════╪════════╪════════╪════════╪════════╪════════╣
 k=8,9    │T0 .b[1]│T4 .b[1]│T8 .b[1]│T12.b[1]│T16.b[1]│T20.b[1]│T24.b[1]│T28.b[1]│
 k=10,11  │T1 .b[1]│T5 .b[1]│T9 .b[1]│T13.b[1]│T17.b[1]│T21.b[1]│T25.b[1]│T29.b[1]│
 k=12,13  │T2 .b[1]│T6 .b[1]│T10.b[1]│T14.b[1]│T18.b[1]│T22.b[1]│T26.b[1]│T30.b[1]│
 k=14,15  │T3 .b[1]│T7 .b[1]│T11.b[1]│T15.b[1]│T19.b[1]│T23.b[1]│T27.b[1]│T31.b[1]│
          └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

#### C/D Fragment 映射公式 (16×8, FP32 累加器)

```
c[0]:  D[groupID,   t*2]           ← m∈[0,7], 偶数列
c[1]:  D[groupID,   t*2+1]         ← m∈[0,7], 奇数列
c[2]:  D[groupID+8, t*2]           ← m∈[8,15], 偶数列
c[3]:  D[groupID+8, t*2+1]         ← m∈[8,15], 奇数列
```

#### C/D Fragment 可视化 (每格 = 1 个 f32 寄存器 = 1 个元素)

```
D[16×8] 标记: T{lane_id}.c[reg_idx]

          n=0      n=1      n=2      n=3      n=4      n=5      n=6      n=7
        ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
 m=0    │T0 .c[0]│T0 .c[1]│T1 .c[0]│T1 .c[1]│T2 .c[0]│T2 .c[1]│T3 .c[0]│T3 .c[1]│
 m=1    │T4 .c[0]│T4 .c[1]│T5 .c[0]│T5 .c[1]│T6 .c[0]│T6 .c[1]│T7 .c[0]│T7 .c[1]│
 m=2    │T8 .c[0]│T8 .c[1]│T9 .c[0]│T9 .c[1]│T10.c[0]│T10.c[1]│T11.c[0]│T11.c[1]│
 m=3    │T12.c[0]│T12.c[1]│T13.c[0]│T13.c[1]│T14.c[0]│T14.c[1]│T15.c[0]│T15.c[1]│
 m=4    │T16.c[0]│T16.c[1]│T17.c[0]│T17.c[1]│T18.c[0]│T18.c[1]│T19.c[0]│T19.c[1]│
 m=5    │T20.c[0]│T20.c[1]│T21.c[0]│T21.c[1]│T22.c[0]│T22.c[1]│T23.c[0]│T23.c[1]│
 m=6    │T24.c[0]│T24.c[1]│T25.c[0]│T25.c[1]│T26.c[0]│T26.c[1]│T27.c[0]│T27.c[1]│
 m=7    │T28.c[0]│T28.c[1]│T29.c[0]│T29.c[1]│T30.c[0]│T30.c[1]│T31.c[0]│T31.c[1]│
        ╠════════╪════════╪════════╪════════╪════════╪════════╪════════╪════════╣
 m=8    │T0 .c[2]│T0 .c[3]│T1 .c[2]│T1 .c[3]│T2 .c[2]│T2 .c[3]│T3 .c[2]│T3 .c[3]│
 m=9    │T4 .c[2]│T4 .c[3]│T5 .c[2]│T5 .c[3]│T6 .c[2]│T6 .c[3]│T7 .c[2]│T7 .c[3]│
 m=10   │T8 .c[2]│T8 .c[3]│T9 .c[2]│T9 .c[3]│T10.c[2]│T10.c[3]│T11.c[2]│T11.c[3]│
 m=11   │T12.c[2]│T12.c[3]│T13.c[2]│T13.c[3]│T14.c[2]│T14.c[3]│T15.c[2]│T15.c[3]│
 m=12   │T16.c[2]│T16.c[3]│T17.c[2]│T17.c[3]│T18.c[2]│T18.c[3]│T19.c[2]│T19.c[3]│
 m=13   │T20.c[2]│T20.c[3]│T21.c[2]│T21.c[3]│T22.c[2]│T22.c[3]│T23.c[2]│T23.c[3]│
 m=14   │T24.c[2]│T24.c[3]│T25.c[2]│T25.c[3]│T26.c[2]│T26.c[3]│T27.c[2]│T27.c[3]│
 m=15   │T28.c[2]│T28.c[3]│T29.c[2]│T29.c[3]│T30.c[2]│T30.c[3]│T31.c[2]│T31.c[3]│
        └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘

每线程持有 2×2 的子块：c[0],c[1] 在 Row 0-7, c[2],c[3] 在 Row 8-15
```

#### Branchless 索引公式 (C/D → 全局坐标)

```cpp
// 从 mma.sync 输出写回 Global Memory
for (int i = 0; i < 4; i++) {
    int row = (lane_id >> 2) + 8 * (i / 2);       // m 坐标
    int col = 2 * (lane_id & 3) + (i % 2);        // n 坐标
    output[(tile_m + row) * N + (tile_n + col)] = d[i];
}
```

---

### 7.3 m16n8k16 — INT8 / FP8 (8-bit 输入, SM 8.0+ / SM 8.9+)

**8-bit 类型每个 32-bit 寄存器打包 4 个元素，寄存器数量减半。**

#### 寄存器分配

| 操作数 | 矩阵尺寸 | 元素数/线程 | 寄存器数/线程 | 打包 |
|--------|---------|-----------|-------------|------|
| A | 16×16 (m×k) | 8 × int8 | **2** × b32 | 4 × int8/reg |
| B | 16×8 (k×n) | 4 × int8 | **1** × b32 | 4 × int8/reg |
| C/D (INT8→INT32) | 16×8 | 4 × s32 | 4 × s32 | 1/reg |
| C/D (FP8→FP32) | 16×8 | 4 × f32 | 4 × f32 | 1/reg |

#### A Fragment 映射 (16×16, INT8/FP8, row-major)

```
a[0]:  A[groupID,   t*4+0], A[groupID,   t*4+1], A[groupID,   t*4+2], A[groupID,   t*4+3]
       ← m∈[0,7], k=[t*4 .. t*4+3], 4 threads × 4 elements = 16 columns ✓

a[1]:  A[groupID+8, t*4+0], A[groupID+8, t*4+1], A[groupID+8, t*4+2], A[groupID+8, t*4+3]
       ← m∈[8,15], k=[t*4 .. t*4+3]
```

**与 FP16 的关键差异：** K=16 的 16 列不再分为两半 (a[0]/a[1])，而是由 4 个线程各持 4 列完整覆盖。M 维度的上下两半由 a[0]/a[1] 区分。

#### B Fragment 映射 (16×8, INT8/FP8, col-major)

```
b[0]:  B[t*4+0, groupID], B[t*4+1, groupID], B[t*4+2, groupID], B[t*4+3, groupID]
       ← k=[t*4 .. t*4+3], n=groupID
       4 threads × 4 = 16 rows, 8 groups = 8 cols → 16×8 ✓
```

#### C/D Fragment — 与 FP16 完全相同

累加器映射不受输入类型影响，始终遵循 7.2 节的 C/D 映射公式。

#### A Fragment 可视化 (INT8, 每格 = 1 个 b32 = 4 个 int8)

```
A[16×16] INT8, 标记: T{lane_id}.a[reg_idx]
(每格覆盖连续 4 列)

             k=0..3     k=4..7     k=8..11    k=12..15
           ┌──────────┬──────────┬──────────┬──────────┐
  m=0      │ T0 .a[0] │ T1 .a[0] │ T2 .a[0] │ T3 .a[0] │
  m=1      │ T4 .a[0] │ T5 .a[0] │ T6 .a[0] │ T7 .a[0] │
  m=2      │ T8 .a[0] │ T9 .a[0] │ T10.a[0] │ T11.a[0] │
  m=3      │ T12.a[0] │ T13.a[0] │ T14.a[0] │ T15.a[0] │
  m=4      │ T16.a[0] │ T17.a[0] │ T18.a[0] │ T19.a[0] │
  m=5      │ T20.a[0] │ T21.a[0] │ T22.a[0] │ T23.a[0] │
  m=6      │ T24.a[0] │ T25.a[0] │ T26.a[0] │ T27.a[0] │
  m=7      │ T28.a[0] │ T29.a[0] │ T30.a[0] │ T31.a[0] │
           ╠══════════╪══════════╪══════════╪══════════╣
  m=8      │ T0 .a[1] │ T1 .a[1] │ T2 .a[1] │ T3 .a[1] │
  m=9      │ T4 .a[1] │ T5 .a[1] │ T6 .a[1] │ T7 .a[1] │
  ...      │  ...     │  ...     │  ...     │  ...     │
  m=15     │ T28.a[1] │ T29.a[1] │ T30.a[1] │ T31.a[1] │
           └──────────┴──────────┴──────────┴──────────┘
```

---

### 7.4 m16n8k32 — INT8 / FP8 加倍 K (SM 8.0+ / SM 8.9+)

**K 从 16 扩展到 32，寄存器数量翻倍。**

#### 寄存器分配

| 操作数 | 矩阵尺寸 | 寄存器数/线程 | 打包 |
|--------|---------|-------------|------|
| A | 16×32 (m×k) | **4** × b32 | 4 × int8/reg |
| B | 32×8 (k×n) | **2** × b32 | 4 × int8/reg |
| C/D | 16×8 | 4 × s32/f32 | 1/reg |

#### A Fragment 映射 (16×32, row-major)

```
a[0]:  A[groupID,   t*4+0..3]       ← m∈[0,7],  k∈[0,15]
a[1]:  A[groupID,   t*4+16..19]     ← m∈[0,7],  k∈[16,31]
a[2]:  A[groupID+8, t*4+0..3]       ← m∈[8,15], k∈[0,15]
a[3]:  A[groupID+8, t*4+16..19]     ← m∈[8,15], k∈[16,31]
```

**结构：** 与 FP16 m16n8k16 的 4 寄存器布局**同构** — 上/下半 M × 前/后半 K。

```
A[16×32] = ┌────────────┬────────────┐
            │ a[0]       │ a[1]       │   m∈[0,7]
            │ k∈[0,15]   │ k∈[16,31]  │
            ├────────────┼────────────┤
            │ a[2]       │ a[3]       │   m∈[8,15]
            │ k∈[0,15]   │ k∈[16,31]  │
            └────────────┴────────────┘
```

#### B Fragment 映射 (32×8, col-major)

```
b[0]:  B[t*4+0..3,  groupID]        ← k∈[0,15],  n=groupID
b[1]:  B[t*4+16..19, groupID]       ← k∈[16,31], n=groupID
```

---

### 7.5 m16n8k64 — INT4 / FP4 (4-bit 输入, SM 8.0+ / SM 12.0+)

**4-bit 类型每个 32-bit 寄存器打包 8 个元素，K=64 是 4-bit MMA 的最大 K 值。**

> **这是 RTX 5090 (SM 12.0) 上 NVFP4 推理的核心指令形状。**

#### 寄存器分配

| 操作数 | 矩阵尺寸 | 元素数/线程 | 寄存器数/线程 | 打包 |
|--------|---------|-----------|-------------|------|
| A | 16×64 (m×k) | 32 × 4-bit | **4** × b32 | **8** × 4-bit/reg |
| B | 64×8 (k×n) | 16 × 4-bit | **2** × b32 | **8** × 4-bit/reg |
| C/D (INT4→INT32) | 16×8 | 4 × s32 | 4 × s32 | 1/reg |
| C/D (FP4→FP32, block_scale) | 16×8 | 4 × f32 | 4 × f32 | 1/reg |
| SF_A (block_scale) | — | 1 × ue4m3/ue8m0 | 1 × u8 | 缩放因子 |
| SF_B (block_scale) | — | 1 × ue4m3/ue8m0 | 1 × u8 | 缩放因子 |

#### A Fragment 映射 (16×64, row-major, 4-bit)

```
每个寄存器打包 8 个 4-bit 元素 (8 × 4 = 32 bits)
4 threads × 8 elements = 32 columns per register set

a[0]:  A[groupID,   t*8+0..7]       ← m∈[0,7],  k∈[0,31]
a[1]:  A[groupID,   t*8+32..39]     ← m∈[0,7],  k∈[32,63]
a[2]:  A[groupID+8, t*8+0..7]       ← m∈[8,15], k∈[0,31]
a[3]:  A[groupID+8, t*8+32..39]     ← m∈[8,15], k∈[32,63]
```

**与其他形状同构：** 依然是 上/下半 M × 前/后半 K 的 4 寄存器结构。

```
A[16×64] = ┌──────────────┬──────────────┐
            │ a[0]         │ a[1]         │   m∈[0,7]
            │ k∈[0,31]     │ k∈[32,63]    │
            ├──────────────┼──────────────┤
            │ a[2]         │ a[3]         │   m∈[8,15]
            │ k∈[0,31]     │ k∈[32,63]    │
            └──────────────┴──────────────┘
```

#### B Fragment 映射 (64×8, col-major, 4-bit)

```
b[0]:  B[t*8+0..7,  groupID]        ← k∈[0,31],  n=groupID
b[1]:  B[t*8+32..39, groupID]       ← k∈[32,63], n=groupID
```

#### C/D Fragment — 与所有 m16n8k* 形状相同

累加器映射始终遵循 7.2 节的 C/D 公式 (16×8, 4 regs)。

#### A Fragment 可视化 (4-bit, 每格 = 1 个 b32 = 8 个 4-bit 元素)

```
A[16×64] 4-bit, 标记: T{lane_id}.a[reg_idx]
(每格覆盖连续 8 列)

            k=0..7     k=8..15    k=16..23   k=24..31  ║ k=32..39   k=40..47   k=48..55   k=56..63
          ┌──────────┬──────────┬──────────┬──────────╫──────────┬──────────┬──────────┬──────────┐
 m=0      │ T0 .a[0] │ T1 .a[0] │ T2 .a[0] │ T3 .a[0] ║ T0 .a[1] │ T1 .a[1] │ T2 .a[1] │ T3 .a[1] │
 m=1      │ T4 .a[0] │ T5 .a[0] │ T6 .a[0] │ T7 .a[0] ║ T4 .a[1] │ T5 .a[1] │ T6 .a[1] │ T7 .a[1] │
 ...      │  ...     │  ...     │  ...     │  ...     ║  ...     │  ...     │  ...     │  ...     │
 m=7      │T28.a[0]  │T29.a[0]  │T30.a[0]  │T31.a[0]  ║T28.a[1]  │T29.a[1]  │T30.a[1]  │T31.a[1]  │
          ╠══════════╪══════════╪══════════╪══════════╬══════════╪══════════╪══════════╪══════════╣
 m=8      │ T0 .a[2] │ T1 .a[2] │ T2 .a[2] │ T3 .a[2] ║ T0 .a[3] │ T1 .a[3] │ T2 .a[3] │ T3 .a[3] │
 ...      │  ...     │  ...     │  ...     │  ...     ║  ...     │  ...     │  ...     │  ...     │
 m=15     │T28.a[2]  │T29.a[2]  │T30.a[2]  │T31.a[2]  ║T28.a[3]  │T29.a[3]  │T30.a[3]  │T31.a[3]  │
          └──────────┴──────────┴──────────┴──────────╨──────────┴──────────┴──────────┴──────────┘
```

#### 各 4-bit 形状打包规则统一

| 形状 | 每 reg 打包 | A 总元素/线程 | B 总元素/线程 | 校验 |
|------|-----------|-------------|-------------|------|
| m8n8k32 | 8 × 4-bit | 8 (1 reg) | 8 (1 reg) | 8×32/32=8 ✓ |
| m16n8k32 | 8 × 4-bit | 16 (2 reg) | 8 (1 reg) | 16×32/32=16 ✓ |
| m16n8k64 | 8 × 4-bit | 32 (4 reg) | 16 (2 reg) | 16×64/32=32 ✓ |

---

### 7.6 m16n8k8 — FP16 / BF16 (SM 7.5+)

**K 减半的版本，寄存器需求更少，适用于 K 较小的场景。**

#### 寄存器分配

| 操作数 | 矩阵尺寸 | 寄存器数/线程 | 打包 |
|--------|---------|-------------|------|
| A | 16×8 (m×k) | **2** × b32 | 2 × f16/reg |
| B | 8×8 (k×n) | **1** × b32 | 2 × f16/reg |
| C/D | 16×8 | 4 × f32 | 1/reg |

#### Fragment 映射

```
A (16×8, row-major):
  a[0]:  A[groupID,   t*2],   A[groupID,   t*2+1]     ← m∈[0,7],  k∈[0,7]
  a[1]:  A[groupID+8, t*2],   A[groupID+8, t*2+1]     ← m∈[8,15], k∈[0,7]

B (8×8, col-major):
  b[0]:  B[t*2, groupID],  B[t*2+1, groupID]          ← k∈[0,7], n=groupID

C/D: 与 m16n8k16 的 C/D 完全相同 (输出都是 16×8)
```

> 本质上是 m16n8k16 FP16 去掉 k∈[8,15] 的后半部分 (寄存器 a[1]/a[3] → 消失，a[0]/a[2] → a[0]/a[1])。

---

### 7.7 m8n8k4 — FP64 (64-bit, SM 8.0+)

**唯一使用 64-bit 寄存器的 MMA 形状。**

#### 寄存器分配

| 操作数 | 矩阵尺寸 | 元素数/线程 | 寄存器数/线程 | 寄存器类型 |
|--------|---------|-----------|-------------|----------|
| A | 8×4 (m×k) | 1 × f64 | 1 × f64 | 64-bit |
| B | 4×8 (k×n) | 1 × f64 | 1 × f64 | 64-bit |
| C/D | 8×8 | 2 × f64 | 2 × f64 | 64-bit |

#### Fragment 映射

```
A (8×4, row-major, 仅支持 .row):
  a[0]:  A[groupID, t]                 ← m=groupID (0..7), k=t (0..3)

B (4×8, col-major, 仅支持 .col):
  b[0]:  B[t, groupID]                 ← k=t (0..3), n=groupID (0..7)

C/D (8×8):
  c[0]:  D[groupID, t*2]               ← m=groupID, n=偶数列
  c[1]:  D[groupID, t*2+1]             ← m=groupID, n=奇数列
```

> C/D 只有上半 (没有 groupID+8)，因为 M=8 仅覆盖 8 行。

#### 布局约束

- A **必须** row-major，B **必须** col-major (`.row.col`)
- `ldmatrix` 不适用于 FP64 (它只处理 16-bit 粒度)
- FP64 操作数需用标准 `ld.shared.f64` 加载

---

### 7.8 m8n8k4 — FP16 (Volta, SM 7.0+)

**第一代 Tensor Core 指令，使用 Quadpair (8 线程) 而非完整 Warp。**

#### 架构差异

```
Volta m8n8k4:    8 线程 (Quadpair) 执行 1 个 MMA
                 1 个 Warp = 4 个 Quadpair → 并行执行 4 个独立 MMA
Ampere m16n8k16: 32 线程 (Full Warp) 协作执行 1 个 MMA
```

Quadpair 组成：Thread `[T0..T3]` + `[T16..T19]` 构成 QP0，以此类推。

#### 寄存器分配 (每 Quadpair)

| 操作数 | 矩阵尺寸 | 寄存器数/线程 | 打包 |
|--------|---------|-------------|------|
| A | 8×4 | 2 × b32 | 2 × f16/reg |
| B | 4×8 | 2 × b32 | 2 × f16/reg |
| C/D (f16) | 8×8 | 4 × b32 | 2 × f16/reg |
| C/D (f32) | 8×8 | 8 × f32 | 1/reg |

#### 布局灵活性

与 Ampere+ 不同，Volta 的 A 和 B **都支持** `.row` 和 `.col` 布局 (共 4 种组合)。

> 在 Ampere+ 上，推荐使用 m16n8k16 替代 m8n8k4。m8n8k4 仍可用于向后兼容。

---

### 7.9 ldmatrix 线程-地址映射

#### 地址提供者

`ldmatrix` 按 **Phase** 执行，每 Phase 8 个线程。每个线程提供一个 Shared Memory 地址，指向 8 个连续 f16 (16 字节)。

| 变体 | 加载 8×8 矩阵数 | 地址提供线程 | 输出寄存器/线程 |
|------|---------------|------------|---------------|
| `.x1` | 1 | T0–T7 提供 8 行地址 | 1 × b32 |
| `.x2` | 2 | T0–T7 (矩阵1), T8–T15 (矩阵2) | 2 × b32 |
| `.x4` | 4 | T0–T7, T8–T15, T16–T23, T24–T31 | 4 × b32 |

#### 数据分发规则

每行 8 个 f16 = 16 字节 = 128 位。由 4 个线程接收 (每线程 32 位 = 2 个 f16)：

```
Thread T 在 ldmatrix.x1 后持有的数据:

8×8 矩阵 (每行 8 个 f16, 由 T{row*4+col_group} 提供地址):

       col=0,1     col=2,3     col=4,5     col=6,7
     ┌───────────┬───────────┬───────────┬───────────┐
row 0│ T0  .r[0] │ T1  .r[0] │ T2  .r[0] │ T3  .r[0] │  T0 提供地址
row 1│ T4  .r[0] │ T5  .r[0] │ T6  .r[0] │ T7  .r[0] │  T4 提供地址
row 2│ T8  .r[0] │ T9  .r[0] │ T10 .r[0] │ T11 .r[0] │  T8 提供地址 (仅 .x2/.x4)
row 3│ T12 .r[0] │ T13 .r[0] │ T14 .r[0] │ T15 .r[0] │  T12 提供地址
row 4│ T16 .r[0] │ T17 .r[0] │ T18 .r[0] │ T19 .r[0] │  T16 提供地址 (仅 .x4)
row 5│ T20 .r[0] │ T21 .r[0] │ T22 .r[0] │ T23 .r[0] │  T20 提供地址
row 6│ T24 .r[0] │ T25 .r[0] │ T26 .r[0] │ T27 .r[0] │  T24 提供地址
row 7│ T28 .r[0] │ T29 .r[0] │ T30 .r[0] │ T31 .r[0] │  T28 提供地址
     └───────────┴───────────┴───────────┴───────────┘
```

> `.x4` 加载 4 个 8×8 矩阵，结果分别存入 r[0]、r[1]、r[2]、r[3]。

#### ldmatrix → mma.sync 直接对应

`ldmatrix` 的数据分发格式与 `mma.sync` 的 Fragment 布局**完全匹配** (硬件设计如此)：

```
m16n8k16 FP16 的 A 矩阵 (16×16):
  ldmatrix.x4  →  加载 4 个 8×8 子块  →  输出 r[0..3]
  mma.sync     →  需要 a[0..3]
  ∴ ldmatrix.x4 的输出直接作为 mma.sync 的 A 操作数 ✓

m16n8k16 FP16 的 B 矩阵 (16×8):
  ldmatrix.x2  →  加载 2 个 8×8 子块  →  输出 r[0..1]
  mma.sync     →  需要 b[0..1]
  ∴ ldmatrix.x2 的输出直接作为 mma.sync 的 B 操作数 ✓
```

#### `.trans` 变体

`ldmatrix.x2.trans` 在加载时执行转置。B 矩阵存储为行优先但 MMA 需要列优先时使用：

```cpp
// B 在 smem 中是 row-major，但 mma.sync 需要 .col
// → 使用 ldmatrix.trans 在加载时转置
ldmatrix_x2_trans(frag_B, &smem_B[...]);
```

---

### 7.10 WMMA Fragment 布局

#### 不透明性

WMMA 的 Fragment 布局是 **opaque** 的——NVIDIA 不保证线程到矩阵元素的映射在不同架构间一致。

#### 已知属性

| Fragment | m16n16k16 FP16 | `num_elements`/线程 | 32-bit Regs/线程 |
|----------|---------------|-------------------|-----------------|
| `matrix_a` (f16) | 16×16 = 256 元素 | **16** | 8 (2 f16/reg) |
| `matrix_b` (f16) | 16×16 = 256 元素 | **16** | 8 (2 f16/reg) |
| `accumulator` (f16) | 16×16 = 256 元素 | **8** | 4 (2 f16/reg) |
| `accumulator` (f32) | 16×16 = 256 元素 | **8** | 8 (1 f32/reg) |

#### 元素访问

```cpp
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
// 通过 .x[] 数组访问单个元素
for (int i = 0; i < c_frag.num_elements; i++) {
    c_frag.x[i] = some_value;  // 每线程 8 个 f32
}
```

#### 底层实现 (非公开保证)

在 Ampere 上，WMMA `m16n16k16` 实际编译为 **2 条 `HMMA.16816`** (= 2 × mma.sync.m16n8k16)，将 16×16 的 N 拆为两个 N=8 的半块。因此：

```
WMMA 累加器 frag.x[0..7] 实际分布:
  frag.x[0..3] ≈ 第一个 m16n8k16 的 c[0..3]  (n∈[0,7])
  frag.x[4..7] ≈ 第二个 m16n8k16 的 c[0..3]  (n∈[8,15])
```

> **警告：** 以上映射是逆向分析的结果，不保证跨架构/驱动版本一致。生产代码不应依赖 WMMA 的具体布局。

---

### 7.11 WGMMA 累加器布局 (SM 9.0)

#### 线程模型

WGMMA 使用 **128 线程 (= 4 Warps = 1 Warpgroup)** 协作执行。M 固定为 64。

```
warp_idx  = (threadIdx.x % 128) / 32    // 0..3
lane_id   = threadIdx.x % 32            // 0..31
groupID   = lane_id >> 2                // 0..7
t         = lane_id & 3                 // 0..3
```

#### 累加器寄存器数量 (取决于 N)

| WGMMA Shape | 输出尺寸 | f32 寄存器/线程 | f16 寄存器/线程 | 公式 |
|-------------|---------|---------------|---------------|------|
| m64n8k16 | 64×8 | 4 | 2 | N/2 |
| m64n16k16 | 64×16 | 8 | 4 | N/2 |
| m64n32k16 | 64×32 | 16 | 8 | N/2 |
| m64n64k16 | 64×64 | 32 | 16 | N/2 |
| m64n128k16 | 64×128 | 64 | 32 | N/2 |
| m64n256k16 | 64×256 | 128 | 64 | N/2 |

> 通用公式：f32 累加器寄存器数 = **N / 2** per thread。

#### 累加器线程映射

```
D[64×N] 中线程 (warp_idx, lane_id) 持有的矩阵行:

  row = 16 * warp_idx + groupID + {0, 8}

即每个线程覆盖 2 行 (间隔 8), 由 warp_idx 决定 16 行块:
  Warp 0: rows 0–15
  Warp 1: rows 16–31
  Warp 2: rows 32–47
  Warp 3: rows 48–63

列覆盖: 每线程覆盖 N/2 个列 (与 m16n8k16 的 C/D 列映射模式类推)
  col = 2 * t + {0, 1}  (每 8 列一组循环)
```

#### CuTe TV Layout 编码

```
// m64nN 累加器的 CuTe Thread-Value Layout:
TV_Layout = Layout<
  Shape <Shape <_4, _8, _4>,  Shape <_2, _2, Int<N/8>>>,
  Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>
>
// Thread 模式: (4,8,4):(128,1,16)  → 128 threads
//   4 warps stride 128; 8 lanes stride 1; 4-thread groups stride 16
// Value 模式: (2,2,N/8):(64,8,512) → N/2 values per thread
//   2 rows (±8 offset); 2 adjacent cols; N/8 column groups
```

---

### 7.12 全形状寄存器计数速查表

| 指令形状 | 输入类型 | 位宽 | A Regs | B Regs | C/D Regs | C/D 类型 | 最低 SM |
|----------|---------|------|--------|--------|----------|---------|--------|
| **m16n8k16** | FP16 | 16 | 4×b32 | 2×b32 | 4×f32 | f32 | 8.0 |
| **m16n8k16** | BF16 | 16 | 4×b32 | 2×b32 | 4×f32 | f32 | 8.0 |
| **m16n8k16** | INT8 | 8 | 2×b32 | 1×b32 | 4×s32 | s32 | 8.0 |
| **m16n8k16** | FP8 | 8 | 2×b32 | 1×b32 | 4×f32 | f32 | 8.9 |
| **m16n8k32** | INT8 | 8 | 4×b32 | 2×b32 | 4×s32 | s32 | 8.0 |
| **m16n8k32** | FP8 | 8 | 4×b32 | 2×b32 | 4×f32 | f32 | 8.9 |
| **m16n8k8** | FP16 | 16 | 2×b32 | 1×b32 | 4×f32 | f32 | 7.5 |
| **m16n8k8** | BF16 | 16 | 2×b32 | 1×b32 | 4×f32 | f32 | 8.0 |
| **m8n8k4** | FP16 | 16 | 2×b32 | 2×b32 | 8×f32 | f32 | 7.0 |
| **m8n8k4** | FP64 | 64 | 1×f64 | 1×f64 | 2×f64 | f64 | 8.0 |
| **m8n8k16** | INT8 | 8 | 1×b32 | 1×b32 | 2×s32 | s32 | 7.5 |
| **m8n8k32** | INT4 | 4 | 1×b32 | 1×b32 | 2×s32 | s32 | 7.5 |
| **m16n8k32** | INT4 | 4 | 2×b32 | 1×b32 | 4×s32 | s32 | 8.0 |
| **m16n8k64** | INT4 | 4 | 4×b32 | 2×b32 | 4×s32 | s32 | 8.0 |

#### 映射模式总结

```
┌─────────────────────────────────────────────────────────────────┐
│                   通用映射结构 (所有 m16n8k* 形状)              │
│                                                                 │
│  M 维度: 上半 [0,7] = groupID,  下半 [8,15] = groupID + 8     │
│  K 维度: 前半/后半由不同寄存器区分                               │
│  N 维度: t*2 / t*2+1 (16-bit) 或 t*4..t*4+3 (8-bit)            │
│                                                                 │
│  C/D 累加器: 所有形状一致 (m16n8 → 4 regs: 上下 × 奇偶列)     │
│                                                                 │
│  寄存器排列规律:                                                │
│    reg[0] → 上半 M, 前半 K                                      │
│    reg[1] → 上半 M, 后半 K   (若存在)                           │
│    reg[2] → 下半 M, 前半 K                                      │
│    reg[3] → 下半 M, 后半 K   (若存在)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 性能对比与选型

### 8.1 GEMM 性能对比 (相对 cuBLAS)

| 接口 | V100 (SM 7.0) | A100 (SM 8.0) | H100 (SM 9.0) |
|------|:------------:|:------------:|:------------:|
| **WMMA** | 57–79% | 50–70% | ~50% |
| **mma.sync** | 91–98% | 90–98% | 80–90% |
| **wgmma** | — | — | **95–100%** |
| cuBLAS (参考) | 100% | 100% | 100% |

### 8.2 性能差距原因

**WMMA 性能低的原因：**
- Fragment 布局不透明 → 无法配合 `ldmatrix` 消除 Bank Conflict
- 编译器选择的加载路径可能非最优
- m16n16k16 底层实为 2 条 HMMA.16816 → 无法精确调度
- 无法实现精细的 Pipeline (不控制数据何时到达)

**mma.sync 更快的原因：**
- 显式寄存器映射 → 配合 `ldmatrix` 消除 Bank Conflict
- 可精确控制 MMA 与 `cp.async` 的交错 → 更好的流水线
- 灵活选择 m16n8k16/m16n8k8 等形状匹配数据特征

**wgmma 最快的原因：**
- 异步执行 → MMA 与 TMA 数据加载完全重叠
- 128 线程协作 → 更大 tile 更高算术强度
- B 直接从 SMEM 读取 → 不占用寄存器带宽
- 连续 wgmma 可被硬件批处理 → 多个 Tensor Core 并行饱和

### 8.3 寄存器压力对比

| 接口 | m×n Tile | 累加器寄存器/线程 | 操作数寄存器/线程 | 总计 |
|------|---------|----------------|----------------|------|
| mma.sync (m16n8k16) | 16×8 | 4 × f32 | 4+2 (A+B) = 6 × b32 | ~10 |
| mma.sync (m16n8k16 ×4) | 64×32 | 32 × f32 | ~24 × b32 | ~56 |
| wgmma (m64n128k16) | 64×128 | 64 × f32 | B 在 SMEM (0) | ~64 |
| tcgen05 (m64n128k16) | 64×128 | 在 TMEM (0) | 0 | ~0 |

→ 接口演进的一个关键方向是**减少通用寄存器压力**。

---

## 9. 常见问题与调试

### 9.1 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `Illegal instruction` / CUDA error 719 | 使用了目标架构不支持的 MMA 指令 | 检查 `-arch` 编译选项 (如 wgmma 需 `sm_90a`) |
| 结果全零 | wgmma 的 scale_D 设为 0 (清零) 而非 1 (累加) | 首次使用 0，后续累加用 1 |
| 结果数值错误 | Fragment 布局不匹配 (行/列主序搞反) | 检查 `.row.col` 与实际数据布局一致 |
| Bank Conflict 导致性能低 | ldmatrix 访问未 swizzle 的 SMEM | 对 SMEM 应用 XOR/TMA swizzle |
| WMMA 性能远低于预期 | 使用 WMMA 而非 mma.sync | 升级到 mma.sync + ldmatrix |
| wgmma 死锁 | fence/commit/wait 顺序错误 | 确保 fence→mma→commit→wait 顺序 |
| 寄存器溢出 (spill) | 累加器 tile 太大 | 减小 MMA tile 或用 `.maxnreg` 限制 |
| TMEM 分配失败 | 列数不是 2 的幂或 <32 | 检查 tcgen05.alloc 参数 |

### 9.2 性能调优检查清单

- [ ] **选择正确的 MMA 接口** (Ampere: mma.sync, Hopper: wgmma)
- [ ] **累加器 tile 足够大** 以提高算术强度 (至少 m64n64)
- [ ] **SMEM 使用 Swizzle** 消除 ldmatrix/wgmma 的 Bank Conflict
- [ ] **数据加载与 MMA 流水线重叠** (cp.async 双缓冲 / TMA+mbarrier)
- [ ] **寄存器使用在预算内** (检查 `.maxnreg` 和 occupancy)
- [ ] **K 维度 unroll** 适度 (过多 → 寄存器压力，过少 → 指令开销)
- [ ] **Epilogue 与下一轮数据加载重叠** (避免尾部空闲)

### 9.3 调试工具

```bash
# 检查 SASS 中的 MMA 指令
cuobjdump --dump-sass my_kernel.cubin | grep -i "HMMA\|IMMA\|GMMA"

# 检查 PTX 中的 MMA 指令
cuobjdump --dump-ptx my_kernel.cubin | grep -i "mma\|wgmma\|wmma"

# Nsight Compute 的 Tensor Core 利用率
ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
    ./my_kernel

# 检查指令发射效率
ncu --metrics smsp__inst_executed_pipe_tensor.sum ./my_kernel
```

---

## 参考资源

- [NVIDIA PTX ISA — Warp Level Matrix Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA Developer Forum: WMMA vs MMA](https://forums.developer.nvidia.com/t/wmma-vs-mma/318949)
- [NVIDIA Developer Forum: WMMA vs WGMMA on H100](https://forums.developer.nvidia.com/t/wmma-vs-wgmma-on-h100-gpu/354730)
- [Bruce-Lee-LY: WMMA API Programming](https://bruce-lee-ly.medium.com/nvidia-tensor-core-introduction-to-wmma-api-programming-21bcfee4ec45)
- [Bruce-Lee-LY: MMA PTX Programming](https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d)
- [Colfax Research: WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
- [Colfax Research: Tensor Memory on Blackwell](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
- [spatters.ca: Fast Tensor Core Matmul on Ada](https://www.spatters.ca/mma-matmul)
- [A Gentle Introduction to GEMM Using MMA Tensor Cores](https://am17an.bearblog.dev/a-gentle-introduction-to-gemm-using-mma-tensor-cores/)
- [Programming Tensor Cores in CUDA 9 (NVIDIA Blog)](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [0mean1sigma: Introduction to Tensor Cores Programming](https://0mean1sigma.com/tgemm/)
- [Lei Mao: NVIDIA Tensor Core Programming](https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/)
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [DeepWiki: CUTLASS SM90 Hopper Architecture](https://deepwiki.com/NVIDIA/cutlass/7.1-sm90-hopper-architecture)

---

*本文档作为 LLM Kernel Agent 的 Tensor Core 编程接口参考。配合 `tensor-core.md`（硬件架构）和 `official-std/` 目录下的文档共同使用。*
