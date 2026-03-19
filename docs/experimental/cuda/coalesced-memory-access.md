# Global Memory Coalescing 深度指南

> 面向 LLM 高性能 Kernel 开发的全局内存合并访问原理、模式分析与优化实践
> 覆盖事务机制、对齐规则、Stride 分析、向量化访问、AoS/SoA 以及 LLM Kernel 实战

---

## 目录

1. [全局内存访问基础](#1-全局内存访问基础)
2. [合并访问 (Coalescing) 原理](#2-合并访问-coalescing-原理)
3. [事务粒度与缓存路径](#3-事务粒度与缓存路径)
4. [对齐与边界效应](#4-对齐与边界效应)
5. [Stride 访问模式分析](#5-stride-访问模式分析)
6. [AoS vs SoA 数据布局](#6-aos-vs-soa-数据布局)
7. [向量化访问 (Vectorized Access)](#7-向量化访问-vectorized-access)
8. [单指令访问的大小与对齐要求](#8-单指令访问的大小与对齐要求)
9. [利用 Shared Memory 实现合并访问](#9-利用-shared-memory-实现合并访问)
10. [LLM Kernel 实战模式](#10-llm-kernel-实战模式)
11. [性能分析与度量](#11-性能分析与度量)
12. [合并访问检查清单](#12-合并访问检查清单)

---

## 1. 全局内存访问基础

### 1.1 全局内存的物理特性

全局内存 (Global Memory) 位于 GPU 板载的 HBM/GDDR 上，是容量最大但延迟最高的存储层次：

| 特性 | 数值 (典型) |
|------|-----------|
| 容量 | 16–192 GB (HBM2e/HBM3) |
| 带宽 | 900–3,350 GB/s (A100: 2 TB/s, H100: 3.35 TB/s) |
| 延迟 | 400–600 个时钟周期 |
| 访问粒度 | 32、64 或 128 字节事务 |

全局内存带宽是 LLM 推理的核心瓶颈 (memory-bound kernel 占主导)，因此最大化有效带宽利用率是关键。

### 1.2 Warp 级内存访问

GPU 执行的最小调度单位是 **Warp (32 个线程)**。当 Warp 中的线程执行同一条全局内存访问指令时，硬件会尝试将这 32 个线程的独立内存请求**合并 (coalesce)** 为尽可能少的内存事务：

```
32 线程各请求 4B → 理想情况合并为 1 个 128B 事务 (4 个 32B sector)
32 线程各请求 4B → 最差情况产生 32 个独立事务 (每个 32B)
```

**合并的核心条件：** 线程访问的地址在物理内存中**连续且对齐**。

### 1.3 为什么合并如此重要

```
完美合并:   32 线程 × 4B = 128B 数据 / 128B 传输 = 100% 利用率
完全不合并: 32 线程 × 4B = 128B 数据 / 1024B 传输 = 12.5% 利用率
```

不合并意味着传输了大量无用数据，等同于浪费 87.5% 的内存带宽。对于 memory-bound 的 LLM kernel (如 Attention 的 K/V 加载)，这直接导致数倍的性能下降。

---

## 2. 合并访问 (Coalescing) 原理

### 2.1 合并的定义

**Memory Coalescing (内存合并)**：当一个 Warp 内的线程同时访问全局内存时，硬件将这些访问按地址连续性合并为最少数量的内存事务。

合并的本质是利用 DRAM 的突发传输 (burst transfer) 特性——DRAM 一次传输一整段连续数据的代价与传输其中一个字几乎相同。

### 2.2 理想合并模式

```
Thread:  T0    T1    T2    T3   ...  T30   T31
Address: 0x100 0x104 0x108 0x10C ... 0x178 0x17C
         ├─────────────── 128 bytes ──────────────┤
         → 合并为 1 个 128B 事务 (4 个 sector)  ✅
```

每个线程访问相邻的 4 字节 → Warp 共 128 字节 → 恰好是一个 128 字节对齐段 → 1 个事务完成。

### 2.3 合并条件总结

| 条件 | 说明 |
|------|------|
| **地址连续** | 线程 `t` 访问 `base + t * sizeof(T)`，无间隔 |
| **自然对齐** | 访问段的首地址是段大小的倍数 |
| **同一指令** | 同一 Warp 的线程在同一时刻执行相同的 load/store |
| **数据大小** | 每线程访问 1/2/4/8/16 字节 |

### 2.4 合并的粒度

硬件在合并时会将 Warp 的 32 个地址归入一个或多个**对齐的内存段 (segment)**：

- 32B 对齐段 (1 个 sector)
- 64B 对齐段 (2 个 sector)
- 128B 对齐段 (4 个 sector)

落入同一段的访问合并为一个事务，跨越多个段则产生多个事务。

---

## 3. 事务粒度与缓存路径

### 3.1 两种缓存路径

现代 NVIDIA GPU 的全局内存加载有两条路径：

| 路径 | L1 参与 | 事务粒度 | 缓存提示 | PTX 修饰符 |
|------|---------|---------|---------|-----------|
| **L1+L2** | 是 | 128 字节 (cache line) | 默认 | `.ca` |
| **L2 only** | 否 | 32 字节 (sector) | 绕过 L1 | `.cg` |

### 3.2 L1+L2 路径 (128B 事务)

```
Warp 请求 → L1 Cache 查找 → 命中: 直接返回
                            → 未命中: 从 L2/DRAM 加载整个 128B cache line

特点:
- 即使只需 4 字节，也会加载整个 128B cache line
- 适合空间局部性好的访问 (相邻线程/后续访问复用同一 line)
- 不适合稀疏访问 (浪费带宽)
```

### 3.3 L2-Only 路径 (32B 事务)

```
Warp 请求 → 绕过 L1 → L2 Cache 以 32B sector 粒度服务

特点:
- 事务粒度更细 (32B vs 128B)
- 稀疏访问时浪费更少
- 流式大数据场景首选 (.cs / .cg)
```

**关键对比示例 — stride-128B 访问：**

```
场景: 每线程读 4B，stride = 32 个 float (128 字节)

L1+L2 路径: 32 × 128B = 4096B 传输量 (仅用 128B → 3.1% 效率)
L2 路径:    32 × 32B  = 1024B 传输量 (仅用 128B → 12.5% 效率)
```

### 3.4 Store 操作

Store 操作**不经过 L1**，仅在 L2 缓存后写回设备内存。Store 事务粒度为 **32 字节**。

### 3.5 编译时控制缓存路径

```bash
# 默认: 全局加载经过 L1+L2
nvcc -Xptxas -dlcm=ca kernel.cu

# 全局加载仅经过 L2
nvcc -Xptxas -dlcm=cg kernel.cu
```

在 CUDA C++ 中通过内联 PTX 精确控制：

```cpp
// L1+L2 (默认)
ld.global.ca.f32 %f0, [%rd0];

// L2 only (绕过 L1)
ld.global.cg.f32 %f0, [%rd0];

// Streaming (L2 低优先驱逐)
ld.global.cs.f32 %f0, [%rd0];
```

---

## 4. 对齐与边界效应

### 4.1 自然对齐要求

全局内存事务必须**自然对齐**——事务首地址必须是事务大小的整数倍：

```
32B 事务: 首地址必须是 32 的倍数 (0x00, 0x20, 0x40, ...)
64B 事务: 首地址必须是 64 的倍数 (0x00, 0x40, 0x80, ...)
128B 事务: 首地址必须是 128 的倍数 (0x00, 0x80, 0x100, ...)
```

### 4.2 cudaMalloc 的对齐保证

`cudaMalloc` 返回的指针保证**至少 256 字节对齐** (实际通常 512B)，因此：
- 数组首元素的访问天然满足 128B/64B/32B 对齐
- 数组中间元素的访问取决于偏移量

### 4.3 偏移导致的额外事务

```cpp
// base 是 256B 对齐的
float* base = ...; // cudaMalloc 分配
int offset = 1;    // 偏移 1 个 float = 4 字节

// Kernel: 每线程读 base[tid + offset]
// Thread 0 → base + 4  (不再 128B 对齐)
// Thread 31 → base + 128
// 跨越 2 个 128B 段 → 需要 2 个事务而非 1 个
```

**偏移对带宽的影响：**

| 偏移 (float 个数) | 128B 段数 | 事务数 | 利用率 |
|-------------------|----------|--------|--------|
| 0 | 1 | 1 (4 sectors) | 100% |
| 1–31 | 2 | 2 (5 sectors) | 80% |
| 0 (L2-only 路径) | — | 4 sectors | 100% |
| 1–7 (L2-only) | — | 5 sectors | 80% |

> **偏移导致的带宽损失通常在 10–20%**，远小于 stride 访问的损失。对于绝大多数 LLM kernel，cudaMalloc 的自然对齐已足够。

### 4.4 结构体对齐

```cpp
// ❌ 非自然对齐的结构体
struct Bad {
    char a;      // 1B
    float b;     // 4B, 但偏移 1 → 不 4B 对齐
    double c;    // 8B, 但偏移 5 → 不 8B 对齐
};  // sizeof = 13 (无 padding) 或 16 (有 compiler padding)

// ✅ 使用 __align__ 确保对齐
struct __align__(16) Good {
    float x, y, z, w;  // 16B, 16B 对齐
};
```

---

## 5. Stride 访问模式分析

### 5.1 什么是 Stride 访问

当线程 `t` 访问 `data[t * stride]` 而 `stride > 1` 时，称为跨步 (strided) 访问。

```
Stride = 1 (连续):  T0→[0] T1→[1] T2→[2] ... T31→[31]    ← 完美合并
Stride = 2:          T0→[0] T1→[2] T2→[4] ... T31→[62]    ← 部分合并
Stride = 32:         T0→[0] T1→[32] T2→[64] ... T31→[992] ← 完全不合并
```

### 5.2 Stride 对有效带宽的影响

```
有效带宽 = (有用数据) / (实际传输数据)

Stride=1:  128B 有用 / 128B 传输 = 100%
Stride=2:  128B 有用 / 256B 传输 = 50%
Stride=4:  128B 有用 / 512B 传输 = 25%
Stride=8:  128B 有用 / 1024B 传输 = 12.5%
Stride=32: 128B 有用 / 4096B 传输 = 3.1%  (L1 路径)
           128B 有用 / 1024B 传输 = 12.5%  (L2 路径)
```

**实测数据趋势 (以 A100 为例)：**

| Stride | 有效带宽 (相对峰值) | 说明 |
|--------|-------------------|------|
| 1 | ~90–95% | 接近峰值 |
| 2 | ~50% | 半数传输浪费 |
| 4 | ~25% | |
| 8 | ~13% | |
| 16 | ~7% | |
| 32+ | ~3–4% | 极差，需要重构访问模式 |

### 5.3 Stride 访问在 LLM Kernel 中的场景

| 场景 | Stride 原因 | 影响 |
|------|------------|------|
| **多头注意力 K/V 访问** | Head 维度交错存储 (BSNH 布局) | V 的列访问跨 head_dim stride |
| **矩阵列访问** | row-major 矩阵按列读取 | stride = 行宽 |
| **AoS 结构体中取单字段** | 字段间隔为结构体大小 | stride = sizeof(struct)/sizeof(field) |
| **转置操作** | 写入方向与存储方向正交 | stride = 矩阵维度 |
| **通道优先 (NCHW) 的空间访问** | 空间相邻元素间隔 C 个通道 | stride = C |

### 5.4 解决 Stride 访问的策略

**策略 1：调整数据布局**
```cpp
// ❌ BSNH: V 的 head_dim 维度不连续
// V[batch][seq][num_heads][head_dim]
float v = V[b * S * N * D + s * N * D + h * D + d];  // d 连续 ✅
// 但跨 seq 读取: stride = N * D

// ✅ BNSH: 同一 head 内 seq 连续
// V[batch][num_heads][seq][head_dim]
float v = V[b * N * S * D + h * S * D + s * D + d];  // s 和 d 都连续 ✅
```

**策略 2：通过 Shared Memory 中转**
```cpp
// 先合并读入 Shared Memory，再以任意模式读出
__shared__ float smem[TILE_SIZE][TILE_SIZE + 1]; // +1 避免 bank conflict

// 合并读取 Global → Shared (stride=1 in global)
smem[ty][tx] = global[gy * width + gx];
__syncthreads();

// 从 Shared Memory stride 读取 (Shared Memory 无合并问题)
float val = smem[tx][ty];  // 转置读取
```

**策略 3：向量化加载后寄存器 shuffle**
```cpp
// 加载连续数据到向量寄存器，然后在寄存器级别重排
float4 vec = reinterpret_cast<float4*>(data)[tid];  // 合并加载
// 在寄存器中 shuffle 得到所需的 stride 数据
```

---

## 6. AoS vs SoA 数据布局

### 6.1 两种布局的对比

**Array of Structures (AoS) — 交错布局：**

```cpp
struct Particle { float x, y, z, mass; };
Particle particles[N];

// 内存布局:
// [x0 y0 z0 m0] [x1 y1 z1 m1] [x2 y2 z2 m2] ...

// 读取所有 x: stride = 4 (sizeof(Particle)/sizeof(float))
float x = particles[tid].x;  // ❌ 4-way stride
```

**Structure of Arrays (SoA) — 连续布局：**

```cpp
struct ParticleSystem {
    float* x;      // [x0 x1 x2 x3 ... xN]
    float* y;      // [y0 y1 y2 y3 ... yN]
    float* z;      // [z0 z1 z2 z3 ... zN]
    float* mass;   // [m0 m1 m2 m3 ... mN]
};

// 读取所有 x: stride = 1 (连续)
float x = ps.x[tid];  // ✅ 完美合并
```

### 6.2 带宽利用率对比

| 操作 | AoS | SoA | 差异 |
|------|-----|-----|------|
| 读取单字段 (如 x) | 25% 利用率 | 100% 利用率 | **4×** |
| 读取所有字段 | 100% 利用率 | 100% 利用率 | 相同 |
| 更新单字段 | 需 read-modify-write | 直接写入 | SoA 更优 |

### 6.3 LLM 中的等价问题

LLM kernel 中不直接使用 AoS/SoA 术语，但存在等价的布局选择：

| AoS 等价 | SoA 等价 | 场景 |
|----------|----------|------|
| `[seq][num_heads][head_dim]` (交错 head) | `[num_heads][seq][head_dim]` (head 连续) | Multi-Head Attention |
| NHWC (通道交错) | NCHW (通道连续) | 卷积 (少用于 LLM) |
| 交错 Q/K/V 在一个张量 | Q, K, V 独立张量 | Attention 输入 |
| 参数+梯度交错 | 参数和梯度分离 | 训练优化器 |

### 6.4 何时 AoS 可以接受

当 **所有字段** 在同一 kernel 中同时被访问时，AoS 的合并效率与 SoA 相当（因为加载整个结构体本身就是连续的）。特别是当结构体大小为 4/8/16 字节时，整体加载与向量化等价。

```cpp
// 结构体 = 16B = float4，可向量化加载
struct __align__(16) Vec3 {
    float x, y, z, w;
};

Vec3 v = data[tid];  // ✅ 等价于 float4 加载，完美合并
```

---

## 7. 向量化访问 (Vectorized Access)

### 7.1 向量化的意义

向量化访问使用宽数据类型 (如 `float4`, `int4`, `half2`) 一次加载/存储多个元素：

```
标量加载:   每线程 4B → LDG.E.32     → 32 线程 = 128B, 32 条指令
向量化加载: 每线程 16B → LDG.E.128   → 32 线程 = 512B, 8 条指令

收益:
- 指令数减少 4× (LDG.E.128 vs LDG.E.32)
- 每条指令传输更多数据
- 更少的指令发射开销和调度压力
```

### 7.2 向量化类型与加载宽度

| 类型 | 大小 | SASS 指令 | 适用场景 |
|------|------|----------|---------|
| `float` / `int` | 4B | `LDG.E.32` | 基线 |
| `float2` / `int2` / `half4` | 8B | `LDG.E.64` | 中等优化 |
| `float4` / `int4` | 16B | `LDG.E.128` | 最优 |
| `half2` / `__nv_bfloat162` | 4B | `LDG.E.32` | FP16/BF16 标量 |
| `short4` (4×half packed) | 8B | `LDG.E.64` | FP16 中等优化 |
| `uint4` (8×half packed) | 16B | `LDG.E.128` | FP16 最优 |

### 7.3 向量化加载代码模式

```cpp
// ============= float4 向量化加载 =============
__global__ void vectorized_copy(float4* out, const float4* in, int n4) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n4) {
        out[tid] = in[tid];  // 128-bit 合并加载 + 128-bit 合并存储
    }
}

// 调用时确保 N 是 4 的倍数
vectorized_copy<<<(N/4 + 255)/256, 256>>>(
    reinterpret_cast<float4*>(d_out),
    reinterpret_cast<const float4*>(d_in),
    N / 4
);

// ============= 通用的 reinterpret_cast 模式 =============
__global__ void kernel(float* data, int N) {
    // 每线程处理 4 个 float
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4;
    if (idx + 3 < N) {
        float4 val = reinterpret_cast<float4*>(data)[tid];
        // 处理 val.x, val.y, val.z, val.w
        val.x = val.x * 2.0f;
        val.y = val.y * 2.0f;
        val.z = val.z * 2.0f;
        val.w = val.w * 2.0f;
        reinterpret_cast<float4*>(data)[tid] = val;
    }
}
```

### 7.4 FP16/BF16 向量化 (LLM 重点)

```cpp
// ============= half2 SIMD 操作 =============
#include <cuda_fp16.h>

__global__ void half2_kernel(half* out, const half* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 2;
    if (idx + 1 < N) {
        // 一次加载 2 个 half (4B，等效 1 个 float)
        half2 val = reinterpret_cast<const half2*>(in)[tid];
        // half2 SIMD: 一条指令处理 2 个 FP16
        val = __hmul2(val, __float2half2_rn(2.0f));
        reinterpret_cast<half2*>(out)[tid] = val;
    }
}

// ============= 128-bit FP16 向量化 (8 个 half) =============
__global__ void half8_kernel(half* out, const half* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 8;
    if (idx + 7 < N) {
        // uint4 = 16B = 8 × half
        uint4 val = reinterpret_cast<const uint4*>(in)[tid];
        // 逐 half2 处理
        half2* h2 = reinterpret_cast<half2*>(&val);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            h2[i] = __hmul2(h2[i], __float2half2_rn(2.0f));
        }
        reinterpret_cast<uint4*>(out)[tid] = val;
    }
}
```

### 7.5 向量化的对齐要求

| 类型 | 对齐要求 | `cudaMalloc` 保证 | 注意事项 |
|------|---------|------------------|---------|
| `float4` / `uint4` | 16B 对齐 | ✅ (256B 对齐) | 子数组偏移可能破坏对齐 |
| `float2` / `uint2` | 8B 对齐 | ✅ | |
| `half2` | 4B 对齐 | ✅ | |

```cpp
// ⚠️ 偏移可能破坏对齐
float* ptr = cudaMalloc(...);  // 256B 对齐 ✅
float* offset_ptr = ptr + 1;   // 4B 对齐 → 不能 reinterpret_cast<float4*> ❌

// 解决: 确保偏移是向量宽度的倍数
float* safe_ptr = ptr + 4;     // 16B 对齐 → float4 安全 ✅
```

### 7.6 向量化的权衡

| 优点 | 缺点 |
|------|------|
| 指令数减少 2–4× | 寄存器压力增加 |
| 更高的指令吞吐 | 需要数据量是向量宽度的倍数 |
| 降低 L1 tag 查找压力 | 需要满足对齐要求 |
| 编译器可能无法自动向量化 | 边界处理更复杂 |

---

## 8. 单指令访问的大小与对齐要求

### 8.1 单指令访问条件

全局内存指令 (LDG/STG) 能编译为**单条**指令的充要条件：

```
数据大小 ∈ {1, 2, 4, 8, 16} 字节  AND  数据自然对齐
```

**自然对齐：** 地址是数据大小的整数倍。

| 大小 | 对齐要求 | 对应类型 |
|------|---------|---------|
| 1B | 任意地址 | `char`, `int8_t` |
| 2B | 2 的倍数 | `short`, `half`, `__nv_bfloat16` |
| 4B | 4 的倍数 | `float`, `int`, `half2` |
| 8B | 8 的倍数 | `double`, `float2`, `int2` |
| 16B | 16 的倍数 | `float4`, `int4`, `double2` |

### 8.2 非标准大小结构体

```cpp
// ❌ 12B 结构体 → 编译器拆为多条指令
struct Vec3 { float x, y, z; };  // 12 bytes, 不在 {1,2,4,8,16} 中
Vec3 v = data[tid];  // 编译为 3 条 LDG.E.32

// ✅ 16B 对齐结构体 → 单条 128-bit 指令
struct __align__(16) Vec4 { float x, y, z, w; };  // 16 bytes
Vec4 v = data[tid];  // 编译为 1 条 LDG.E.128
```

### 8.3 使用 `__align__` 优化

```cpp
// 将 3 元素结构体 pad 到 16B
struct __align__(16) AlignedVec3 {
    float x, y, z;
    float _pad;  // 填充到 16B
};

// 验证
static_assert(sizeof(AlignedVec3) == 16, "Must be 16 bytes");
static_assert(alignof(AlignedVec3) == 16, "Must be 16-byte aligned");
```

---

## 9. 利用 Shared Memory 实现合并访问

### 9.1 基本模式：合并读 → 重排 → 合并写

当目标访问模式无法直接合并时，使用 Shared Memory 作为中间缓冲：

```
Step 1: 合并读取 Global → Shared (stride=1 in global)
Step 2: __syncthreads()
Step 3: 任意模式读取 Shared → Register
Step 4: 合并写回 Register → Global (stride=1 in global)
```

### 9.2 矩阵转置示例

```cpp
__global__ void transpose(float* out, const float* in, int N) {
    __shared__ float tile[32][33];  // +1 padding 避免 bank conflict

    int xIdx = blockIdx.x * 32 + threadIdx.x;
    int yIdx = blockIdx.y * 32 + threadIdx.y;

    // Step 1: 合并读取 (行优先，stride=1) ✅
    if (xIdx < N && yIdx < N)
        tile[threadIdx.y][threadIdx.x] = in[yIdx * N + xIdx];

    __syncthreads();

    // Step 2: 合并写回 (转置方向也是 stride=1) ✅
    xIdx = blockIdx.y * 32 + threadIdx.x;
    yIdx = blockIdx.x * 32 + threadIdx.y;

    if (xIdx < N && yIdx < N)
        out[yIdx * N + xIdx] = tile[threadIdx.x][threadIdx.y];
    // tile[tx][ty] 是列式读取 shared memory
    // → Bank conflict 由 padding (+1) 解决
    // Global 写入是 stride=1 → 合并 ✅
}
```

### 9.3 GEMM 中的合并加载

```cpp
// GEMM: C = A × B
// A: M×K (row-major), B: K×N (row-major)
// B 的列访问 stride = N → 不合并

// 解决: 先合并加载 B 的 tile 到 shared memory
__shared__ float Bs[TILE_K][TILE_N + 1]; // pad for bank conflict

// 合并加载 B tile (每线程读取 B 的一行元素，stride=1)
for (int i = 0; i < TILE_K; i += blockDim.y) {
    int row = k_start + i + threadIdx.y;
    int col = n_start + threadIdx.x;
    if (row < K && col < N)
        Bs[i + threadIdx.y][threadIdx.x] = B[row * N + col];  // 合并 ✅
}
__syncthreads();

// 从 shared memory 列式读取 (无合并问题，仅需关注 bank conflict)
for (int k = 0; k < TILE_K; k++) {
    float b_val = Bs[k][threadIdx.x];  // 行式读取 shared → 无 bank conflict
    // ...
}
```

---

## 10. LLM Kernel 实战模式

### 10.1 Attention K/V 加载

**问题：** Attention 中 K 和 V 张量的访问模式取决于存储布局。

```
K 形状: [batch, num_heads, seq_len, head_dim]
V 形状: [batch, num_heads, seq_len, head_dim]

Attention 计算: S = Q × K^T, O = softmax(S) × V
```

**K^T 的列访问：**

```
K[b][h][s][d]: head_dim 维连续 ✅ (d 是最内维)
K^T 需要沿 seq_len 维读取: stride = head_dim (通常 64/128)

方案 1: 先将 K tile 加载到 shared memory (合并), 在 smem 中转置
方案 2: 使用 TMA (Hopper) 直接以 2D tile 加载，硬件处理
```

### 10.2 QKV 打包与拆分

```cpp
// ❌ 打包 QKV: [batch, seq, 3, num_heads, head_dim]
// 读 Q 时 stride = 3 (跳过 K 和 V)
float q = qkv[b * S * 3 * N * D + s * 3 * N * D + 0 * N * D + h * D + d];

// ✅ 分离 QKV: Q[batch, seq, num_heads, head_dim], K[...], V[...]
// 读 Q 时 stride = 1 (head_dim 连续)
float q = Q[b * S * N * D + s * N * D + h * D + d];
```

在实践中，许多框架 (如 PyTorch) 的 attention 实现会在计算前将打包 QKV 拆分为独立张量，正是为了合并访问。

### 10.3 Embedding / Gather 操作

```cpp
// Embedding lookup: 本质是 gather (间接索引)
// token_ids: [batch, seq_len], embed_table: [vocab_size, embed_dim]

// ❌ 不合并: 不同线程的 token_id 不同 → 访问 embed_table 的行不连续
float val = embed_table[token_ids[tid] * embed_dim + d];  // 随机行 ❌

// ✅ 合并: 让相邻线程处理同一 token 的不同 embed 维度
// Thread block 处理 1 个 token, 线程沿 embed_dim 展开
int token = token_ids[blockIdx.x];
float val = embed_table[token * embed_dim + threadIdx.x];  // embed_dim 连续 ✅
```

### 10.4 LayerNorm / RMSNorm

```cpp
// LayerNorm: 对每个 token 的 hidden_dim 维度归一化
// data: [batch * seq_len, hidden_dim]

// ✅ 天然合并: 相邻线程访问同一行的连续元素
float val = data[row * hidden_dim + threadIdx.x];  // stride=1 ✅

// ✅ 向量化加载提升效率
float4 val4 = reinterpret_cast<float4*>(&data[row * hidden_dim])[threadIdx.x];
// 每线程加载 4 个 float, 128-bit 向量化 ✅
```

### 10.5 Softmax

```cpp
// Online Softmax: 对 [seq_len] 维度做 softmax
// logits: [batch * num_heads, seq_len]

// ✅ 合并: 同一行的连续元素
float val = logits[row * seq_len + threadIdx.x + offset];  // stride=1 ✅

// ⚠️ 当 seq_len 很大时，需要循环多次覆盖
// 使用向量化加载进一步提升
for (int i = threadIdx.x * 4; i < seq_len; i += blockDim.x * 4) {
    float4 v = reinterpret_cast<float4*>(&logits[row * seq_len])[i / 4];
    // 处理 v.x, v.y, v.z, v.w
}
```

### 10.6 量化 / 反量化

```cpp
// INT8 反量化: output[i] = (int8_data[i] - zero_point) * scale
// int8_data: [N], output: [N]

// ✅ 合并 + 向量化: 每线程加载 16 个 int8 (= 16B = 1 个 uint4)
__global__ void dequantize(half* out, const int8_t* in,
                           float scale, float zp, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 16;  // 每线程处理 16 个元素
    if (idx + 15 < N) {
        // 128-bit 合并加载 16 个 int8
        uint4 packed = reinterpret_cast<const uint4*>(in)[tid];

        // 解包并转换
        int8_t* bytes = reinterpret_cast<int8_t*>(&packed);
        half results[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            results[i] = __float2half((float(bytes[i]) - zp) * scale);
        }

        // 128-bit 合并存储 8 个 half (16B)
        reinterpret_cast<uint4*>(out)[tid * 2]     = reinterpret_cast<uint4*>(results)[0];
        reinterpret_cast<uint4*>(out)[tid * 2 + 1] = reinterpret_cast<uint4*>(results)[1];
    }
}
```

### 10.7 MoE (Mixture of Experts) 的 Gather/Scatter

```cpp
// MoE: 每个 token 路由到不同 expert
// 输入: tokens[N, D], routing_indices[N] → expert_id
// 问题: 不同 token 路由到不同 expert → gather 后 token 不连续

// ❌ 不合并的写入
out[routing_indices[tid] * D + d] = computed_val;  // 随机写入 ❌

// ✅ 优化: 先按 expert 排序 token → 同一 expert 的 token 连续
// Step 1: sort tokens by expert_id (使用 CUB RadixSort)
// Step 2: 批量处理每个 expert 的连续 token → 合并 ✅
```

---

## 11. 性能分析与度量

### 11.1 关键指标

| 指标 | 含义 | 理想值 |
|------|------|--------|
| `sectors_per_request` (加载) | 每个 Warp 请求触发的 32B sector 数 | 4 (= 128B / 32B) |
| `sectors_per_request` (存储) | 每个存储请求的 sector 数 | 4 |
| `gld_efficiency` | 全局加载效率 | >90% |
| `gst_efficiency` | 全局存储效率 | >90% |
| `dram_read_throughput` | DRAM 实际读带宽 | 接近峰值 |
| `l2_read_throughput` | L2 读带宽 | — |

### 11.2 Nsight Compute 度量命令

```bash
# 全局内存效率
ncu --metrics \
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
    l1tex__t_requests_pipe_lsu_mem_global_op_st.sum \
    ./my_kernel

# 计算 sectors_per_request:
# ratio = sectors.sum / requests.sum
# 完美合并 → ratio = 4
# 完全不合并 → ratio = 32
```

### 11.3 判断合并质量

```
sectors_per_request = 4   → 完美合并 ✅
sectors_per_request = 5   → 轻微不对齐 (偏移) → 可接受
sectors_per_request = 8   → 2× stride → 需要优化
sectors_per_request = 16  → 4× stride → 必须优化
sectors_per_request = 32  → 完全不合并 → 性能灾难
```

### 11.4 效率计算

```
Global Load Efficiency = (请求字节数) / (实际传输字节数) × 100%

例:
  请求: 32 线程 × 4B = 128B
  实际: 5 sectors × 32B = 160B
  效率: 128/160 = 80%

  请求: 32 线程 × 4B = 128B
  实际: 32 sectors × 32B = 1024B
  效率: 128/1024 = 12.5%
```

---

## 12. 合并访问检查清单

### 开发时检查

- [ ] **最内层循环的 threadIdx.x 对应最快变化的内存维度** (连续地址)
- [ ] **数组索引中 threadIdx.x 乘以的系数为 1** (或为 sizeof(element) 的 stride)
- [ ] **避免 stride > 1 的全局内存访问**；若不可避免，通过 Shared Memory 中转
- [ ] **结构体大小 ∈ {1, 2, 4, 8, 16} 字节** 且使用 `__align__` 对齐
- [ ] **使用向量化加载** (`float4`, `uint4`, `half2`) 减少指令数
- [ ] **数据布局选择 SoA** 而非 AoS（除非整个结构体同时访问）
- [ ] **偏移量是向量宽度的倍数** (向量化 `reinterpret_cast` 安全)

### 代码 Review 红旗

```cpp
// 🚩 stride 访问 — 严重不合并
data[threadIdx.x * stride]       // stride > 1
matrix[threadIdx.x][col]         // 列式访问 row-major 矩阵
struct_array[threadIdx.x].field  // AoS 单字段

// 🚩 间接索引 — 随机访问
data[index_table[threadIdx.x]]   // gather

// 🚩 条件加载 — 部分线程不参与
if (threadIdx.x % 2 == 0)
    val = data[threadIdx.x / 2]; // 只有偶数线程加载

// ✅ 合并模式
data[threadIdx.x]                     // stride=1
data[blockIdx.x * blockDim.x + threadIdx.x]  // grid-stride
reinterpret_cast<float4*>(data)[threadIdx.x]  // 向量化
```

### 性能验证

- [ ] `ncu` 报告 `sectors_per_request` ≤ 5 (加载)
- [ ] `ncu` 报告 `gld_efficiency` ≥ 80%
- [ ] 实测带宽达到设备峰值的 70%+

---

## 参考资源

- [NVIDIA Blog: Unlock GPU Performance — Global Memory Access in CUDA](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/)
- [NVIDIA Blog: How to Access Global Memory Efficiently in CUDA C/C++ Kernels](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [NVIDIA Blog: CUDA Pro Tip — Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
- [Modal GPU Glossary: Memory Coalescing](https://modal.com/gpu-glossary/perf/memory-coalescing)
- [Aussie AI: CUDA C++ Memory Coalescing Optimizations](https://www.aussieai.com/blog/cuda-memory-coalescing)
- [UIC MCS572: Memory Coalescing Techniques](https://homepages.math.uic.edu/~jan/mcs572/memory_coalescing.pdf)
- [Lei Mao: CUDA Data Alignment](https://leimao.github.io/blog/CUDA-Data-Alignment/)
- [NVIDIA Nsight Docs: Memory Transactions](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/sourcelevel/memorytransactions.htm)
- [siboehm: How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)

---

*本文档作为 LLM Kernel Agent 的全局内存合并访问技能参考。配合 `conflict-free-accesses.md`（Shared Memory Bank Conflict）和 `official-std/` 目录下的 CUDA 编程指南共同使用。*
