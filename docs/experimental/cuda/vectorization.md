# CUDA 向量化深度指南

> 面向 LLM 高性能 Kernel 开发的向量化访存与向量化计算全面解析
> 覆盖向量化加载/存储、Packed Arithmetic、Shared Memory 向量化、编译器行为、FP8/FP4 打包、LLM Kernel 实战模式

---

## 目录

1. [向量化概述与分类](#1-向量化概述与分类)
2. [向量化内存访问 (Vectorized Memory Access)](#2-向量化内存访问-vectorized-memory-access)
3. [Shared Memory 向量化访问](#3-shared-memory-向量化访问)
4. [Packed Arithmetic (向量化计算)](#4-packed-arithmetic-向量化计算)
5. [编译器自动向量化与控制](#5-编译器自动向量化与控制)
6. [FP8 打包与向量化](#6-fp8-打包与向量化)
7. [FP4 (NVFP4) 打包与向量化](#7-fp4-nvfp4-打包与向量化)
8. [LLM Kernel 向量化实战模式](#8-llm-kernel-向量化实战模式)
9. [寄存器压力与向量化权衡](#9-寄存器压力与向量化权衡)
10. [向量化检查清单与最佳实践](#10-向量化检查清单与最佳实践)

---

## 1. 向量化概述与分类

### 1.1 什么是 CUDA 向量化

CUDA 向量化是指利用宽数据类型和专用指令，在**单条指令中处理多个数据元素**的技术。它包含两个互补的维度：

```
CUDA 向量化
├── 向量化访存 (Vectorized Memory Access)
│   ├── Global Memory: LDG.E.128 / STG.E.128 (128-bit)
│   ├── Shared Memory: LDS.128 / STS.128
│   └── 效果: 减少指令数, 提升带宽利用率
│
└── 向量化计算 (Packed Arithmetic / SIMD)
    ├── half2 / __nv_bfloat162: 一条指令处理 2 个 FP16/BF16
    ├── __nv_fp8x4_e4m3: 4 个 FP8 打包存储
    └── 效果: 提升计算吞吐, 减少指令发射开销
```

### 1.2 为什么 LLM Kernel 需要向量化

| 场景 | 瓶颈 | 向量化收益 |
|------|------|-----------|
| Memory-bound Kernel (RMSNorm, Residual Add, Softmax) | 指令发射 / 内存带宽 | 128-bit 加载减少 4× 指令数 |
| Compute-bound Kernel (GEMM epilogue) | 计算吞吐 | half2 packed FMA 翻倍 FP16 吞吐 |
| 量化推理 (FP8/FP4 dequant) | 数据搬运 | 单次加载 16–32 个低精度值 |
| Fused Kernel (SwiGLU + Residual + Norm) | 两者兼有 | 向量化同时优化访存和计算 |

### 1.3 向量化与合并访问的关系

向量化和合并 (coalescing) 是**正交但互补**的优化：

```
合并 (Coalescing):  32 线程的 32 个独立请求 → 合并为 1 个 128B 事务
向量化:             1 个线程的 1 次加载 → 加载 16B (而非 4B)

两者结合:
  标量合并:  32 线程 × 4B = 128B / 事务, 32 条 LDG.E.32 指令
  向量化合并: 32 线程 × 16B = 512B / 事务, 8 条 LDG.E.128 指令
              (同样数据量, 指令数减少 4×)
```

> **关键区别：** 合并减少事务数 (提升有效带宽利用率)，向量化减少指令数 (降低指令发射开销)。对于 memory-bound kernel，两者需同时优化。
>
> 详细的合并访问分析见 `coalesced-memory-access.md`。

---

## 2. 向量化内存访问 (Vectorized Memory Access)

### 2.1 向量化类型与 SASS 指令映射

| 向量类型 | 字节数 | 等价标量 | SASS 加载指令 | SASS 存储指令 |
|---------|--------|---------|-------------|-------------|
| `float` / `int32_t` | 4B | 1 × FP32 | `LDG.E.32` | `STG.E.32` |
| `float2` / `int2` | 8B | 2 × FP32 | `LDG.E.64` | `STG.E.64` |
| `float4` / `int4` / `uint4` | 16B | 4 × FP32 | `LDG.E.128` | `STG.E.128` |
| `half2` / `__nv_bfloat162` | 4B | 2 × FP16 | `LDG.E.32` | `STG.E.32` |
| 4 × `half` (packed in `uint2`) | 8B | 4 × FP16 | `LDG.E.64` | `STG.E.64` |
| 8 × `half` (packed in `uint4`) | 16B | 8 × FP16 | `LDG.E.128` | `STG.E.128` |
| 16 × `int8_t` (packed in `uint4`) | 16B | 16 × INT8 | `LDG.E.128` | `STG.E.128` |
| 16 × `__nv_fp8_e4m3` (packed in `uint4`) | 16B | 16 × FP8 | `LDG.E.128` | `STG.E.128` |
| 32 × FP4 (packed in `uint4`) | 16B | 32 × FP4 | `LDG.E.128` | `STG.E.128` |

**核心原则：** 每线程单指令访问的最大宽度为 **128 bit (16 字节)**，对应 `LDG.E.128`。

### 2.2 reinterpret_cast 模式

向量化加载的标准 C++ 模式是通过 `reinterpret_cast` 将标量指针转为向量指针：

```cpp
// ============= FP32 向量化 (每线程加载 4 个 float) =============
__global__ void scale_fp32(float* data, float factor, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4;
    if (idx + 3 < N) {
        // 128-bit 加载: 1 条 LDG.E.128 替代 4 条 LDG.E.32
        float4 v = reinterpret_cast<float4*>(data)[tid];
        v.x *= factor;
        v.y *= factor;
        v.z *= factor;
        v.w *= factor;
        // 128-bit 存储: 1 条 STG.E.128
        reinterpret_cast<float4*>(data)[tid] = v;
    }
}

// ============= FP16 向量化 (每线程加载 8 个 half) =============
__global__ void scale_fp16(half* data, half factor, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 8;
    if (idx + 7 < N) {
        // 128-bit 加载: 8 × half = 16 bytes
        uint4 packed = reinterpret_cast<const uint4*>(data)[tid];
        half2* h2 = reinterpret_cast<half2*>(&packed);
        half2 factor2 = __half2half2(factor);  // broadcast

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            h2[i] = __hmul2(h2[i], factor2);
        }

        reinterpret_cast<uint4*>(data)[tid] = packed;
    }
}
```

### 2.3 对齐要求

向量化加载/存储要求地址满足**自然对齐** (地址是数据大小的整数倍)：

| 类型 | 对齐要求 | `cudaMalloc` 保证 | 常见问题 |
|------|---------|------------------|---------|
| `float4` / `uint4` | 16B 对齐 | ✅ (256B 对齐) | 子数组偏移可能破坏对齐 |
| `float2` / `uint2` | 8B 对齐 | ✅ | |
| `half2` | 4B 对齐 | ✅ | |

**未对齐访问的后果：**

```cpp
float* ptr = (float*)cudaMalloc(...);  // 256B 对齐

// ✅ 安全: ptr + 0 是 256B 对齐 → 满足 16B 要求
float4 v0 = reinterpret_cast<float4*>(ptr)[0];

// ❌ 未定义行为: ptr + 1 仅 4B 对齐 → 不满足 float4 的 16B 要求
float4 v1 = reinterpret_cast<float4*>(ptr + 1)[0];

// ✅ 安全: ptr + 4 是 16B 对齐
float4 v2 = reinterpret_cast<float4*>(ptr + 4)[0];
```

**处理非对齐尾部的常见模式：**

```cpp
__global__ void safe_vectorized(float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 主循环: 向量化处理
    int vec_idx = tid * 4;
    if (vec_idx + 3 < N) {
        float4 v = reinterpret_cast<float4*>(data)[tid];
        // ... 处理 ...
        reinterpret_cast<float4*>(data)[tid] = v;
    }

    // 尾部: 标量回退处理剩余元素
    int remaining_start = (N / 4) * 4;
    int scalar_tid = remaining_start + (tid % (N - remaining_start));
    if (tid < (N - remaining_start) && scalar_tid < N) {
        data[scalar_tid] *= 2.0f;
    }
}
```

### 2.4 向量化加载的指令级收益

以 32 个线程 (1 个 Warp) 处理 128 个 float 为例：

```
方案 A: 标量加载
  每线程加载 4 个 float → 4 条 LDG.E.32/线程 → Warp 共 128 条指令
  内存事务: 4 个 128B 事务 (已合并)

方案 B: 向量化加载
  每线程加载 1 个 float4 → 1 条 LDG.E.128/线程 → Warp 共 32 条指令
  内存事务: 4 个 128B 事务 (已合并, 每事务 512B → 需多个事务)

指令数: 128 → 32 (减少 4×)
内存带宽: 相同 (合并后相同数据量)
额外收益:
  - L1 tag 查找次数减少 (每次 LDG 需 1 次 tag 查找)
  - Warp Scheduler 发射压力降低
  - 指令缓存 (I-Cache) 压力减少
```

> **实测经验：** 在 memory-bound kernel 上，向量化通常带来 **1.5–2.5× 性能提升**。在极度 memory-bound 的 elementwise kernel (如 residual add) 上，提升可达 **3–4×**。

---

## 3. Shared Memory 向量化访问

### 3.1 Shared Memory 加载指令

Shared Memory 的向量化加载使用独立的 SASS 指令系列：

| 指令 | 宽度 | 每线程加载字节 | 涉及 Bank 数 |
|------|------|-------------|-------------|
| `LDS.32` | 32-bit | 4B | 1 bank |
| `LDS.64` | 64-bit | 8B | 2 banks |
| `LDS.128` | 128-bit | 16B | 4 banks |
| `STS.32` | 32-bit | 4B | 1 bank |
| `STS.64` | 64-bit | 8B | 2 banks |
| `STS.128` | 128-bit | 16B | 4 banks |

### 3.2 LDS.128 与 Bank Conflict：硬件分相 (Phasing) 机制

Shared Memory 有 **32 个 bank**，每 bank 每周期提供 **4 字节**带宽，总计 128 字节/周期。

当一个 Warp (32 线程) 执行 `LDS.128` 时，每线程需从 4 个 bank 读取 → 总共需 32×4 = 128 次 bank 访问。这远超每周期 32 次的 bank 访问容量。

**硬件解决方案 — 4 阶段分相 (4-Phase Servicing)：**

```
LDS.128: 32 线程 × 16B = 512B 总数据
硬件拆分为 4 个阶段执行 (每阶段 128B):

阶段 1: 线程 0–7   → 8 线程 × 4 banks = 32 bank 访问 → 128B ✅ 无冲突
阶段 2: 线程 8–15  → 8 线程 × 4 banks = 32 bank 访问 → 128B ✅ 无冲突
阶段 3: 线程 16–23 → 8 线程 × 4 banks = 32 bank 访问 → 128B ✅ 无冲突
阶段 4: 线程 24–31 → 8 线程 × 4 banks = 32 bank 访问 → 128B ✅ 无冲突
```

**为什么每阶段 8 线程？**

```
每周期 Shared Memory 吞吐 = 128 字节
每线程 LDS.128 = 16 字节
128 / 16 = 8 线程/阶段

→ LDS.128 需要 4 个周期完成, LDS.32 需要 1 个周期
→ 每线程每周期吞吐相同: 4 字节/线程/周期 (达到 speed of light)
```

**阶段内无冲突的保证：** 硬件对每阶段 8 线程的 4 个子元素 (各 32-bit) 交错调度 bank 访问顺序。例如 Lane 0 按 bank 0,1,2,3 顺序访问，Lane 1 按 bank 5,4,7,6 交错访问，确保同一周期内无两个线程访问同一 bank。

**关键结论：**

```
LDS.128 总吞吐 = LDS.32 总吞吐 = 4 字节/线程/周期 (达到 speed of light)

区别:
  LDS.32:  1 条指令, 1 周期, 4B/线程
  LDS.128: 1 条指令, 4 周期, 16B/线程

指令数: LDS.128 是 LDS.32 的 1/4  ← 核心收益
吞吐:   相同 (4B/线程/周期)
```

### 3.3 Nsight Compute 中的 Bank Conflict 误报

使用 `LDS.128` 时，Nsight Compute 的 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` 可能报告非零冲突。这是因为多阶段执行的内部调度被计为 "冲突"。实际性能无影响。

**正确判断方法：** 将 bank conflict 数除以 wavefront 数 (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`)。如果比值 < 1%，则为硬件调度产物，非真正冲突。

### 3.4 Shared Memory 向量化代码模式

```cpp
// ============= 向量化 Shared Memory 加载/存储 =============
__shared__ float smem[256];  // 1024 字节, 256B 对齐

// 向量化加载 shared → register
float4 v = reinterpret_cast<float4*>(smem)[threadIdx.x];
// → 编译为 LDS.128

// 向量化存储 register → shared
reinterpret_cast<float4*>(smem)[threadIdx.x] = v;
// → 编译为 STS.128

// ============= Global → Shared 的向量化搬运 =============
// 用于 GEMM tile loading, 先向量化读 global, 再向量化写 shared
float4 tmp = reinterpret_cast<const float4*>(global_ptr)[tid];
reinterpret_cast<float4*>(shared_ptr)[threadIdx.x] = tmp;
__syncthreads();
```

### 3.5 ptxas 对 Shared Memory 的自动向量化

与 Global Memory 不同，`ptxas` **能够自动向量化 Shared Memory** 访问。因为 Shared Memory 的物理对齐在编译时已知：

```cpp
// 源代码: 4 个标量加载
__shared__ float smem[256];
float a = smem[threadIdx.x * 4 + 0];
float b = smem[threadIdx.x * 4 + 1];
float c = smem[threadIdx.x * 4 + 2];
float d = smem[threadIdx.x * 4 + 3];

// ptxas 可能自动合并为:
// LDS.128  →  一次加载 {a, b, c, d}
```

> **注意：** 这种自动向量化不保证发生。显式使用 `float4` 可确保向量化。

---

## 4. Packed Arithmetic (向量化计算)

### 4.1 概念：2-Way SIMD

NVIDIA GPU 的 CUDA Core 支持对 **半精度打包对 (half2 / __nv_bfloat162)** 执行 2-way SIMD 操作——单条指令同时处理 2 个 FP16/BF16 值：

```
标量 FP16:   HFMA  r0, r1, r2    → 1 个 FP16 FMA / 指令
打包 FP16:   HFMA2 r0, r1, r2    → 2 个 FP16 FMA / 指令  (2× 吞吐)

标量 FP32:   FFMA  r0, r1, r2    → 1 个 FP32 FMA / 指令
```

### 4.2 各架构 CUDA Core 吞吐对比

下表为每 SM 每周期的操作数 (CUDA Core, 非 Tensor Core)：

| 架构 | CC | FP16 (`half2`) | BF16 (`bfloat162`) | FP32 | 备注 |
|------|:--:|:--------------:|:------------------:|:----:|------|
| Pascal (GP100) | 6.0 | 128 | — | 64 | FP16 = 2× FP32 |
| Pascal (GP104) | 6.1 | 2 | — | 128 | 消费级无 FP16 快速路径 |
| Volta | 7.0 | 128 | — | 64 | |
| Turing | 7.5 | 128 | — | 64 | |
| Ampere (GA100) | 8.0 | 256 | 128 | 64 | FP16 = 4× FP32, BF16 = 2× FP32 |
| Ampere (GA10x) | 8.6 | 256 | 128 | 128 | 消费级 FP32 翻倍 |
| Hopper (GH100) | 9.0 | 256 | 128 | 64 | BF16 = FP16 的一半 |

> **关键发现：** 在 Ampere/Hopper CUDA Core 上，**FP16 的 packed 吞吐是 BF16 的 2 倍**。`HFMA2` (FP16) 每周期 256 ops vs `HFMA2` (BF16) 每周期 128 ops。这意味着对于 compute-bound 的非 Tensor Core 操作，FP16 可能优于 BF16。

### 4.3 half2 算术指令完整列表

```
头文件: #include <cuda_fp16.h>     (half2)
        #include <cuda_bf16.h>     (__nv_bfloat162)
```

#### 基本算术

| 指令 | 功能 | SASS 指令 |
|------|------|----------|
| `__hadd2(a, b)` | a + b (逐元素) | HADD2 |
| `__hsub2(a, b)` | a - b | HADD2 (取反后加) |
| `__hmul2(a, b)` | a × b | HMUL2 |
| `__hfma2(a, b, c)` | a × b + c | HFMA2 |
| `__hneg2(a)` | -a | HMUL2 (乘 -1) |
| `__h2div(a, b)` | a / b | HMUL2 + HRCP |
| `__hadd2_sat(a, b)` | clamp(a+b, 0, 1) | HADD2.SAT |
| `__hfma2_sat(a, b, c)` | clamp(a×b+c, 0, 1) | HFMA2.SAT |
| `__hfma2_relu(a, b, c)` | max(0, a×b+c) | HFMA2.RELU |

> `__nv_bfloat162` 有完全对应的同名重载函数。

#### 比较

| 指令 | 返回类型 | 说明 |
|------|---------|------|
| `__heq2(a, b)` | `half2` | 逐元素 ==, 返回 1.0/0.0 |
| `__hgt2(a, b)` | `half2` | 逐元素 > |
| `__hlt2(a, b)` | `half2` | 逐元素 < |
| `__hge2(a, b)` | `half2` | 逐元素 >= |
| `__hle2(a, b)` | `half2` | 逐元素 <= |
| `__hne2(a, b)` | `half2` | 逐元素 != |
| `__hmax2(a, b)` | `half2` | 逐元素 max |
| `__hmin2(a, b)` | `half2` | 逐元素 min |
| `__heq2_mask(a, b)` | `unsigned` | 按半字: 0xFFFF / 0x0000 |
| `__hbeq2(a, b)` | `bool` | 两个元素都相等时返回 true |

#### 数学函数 (Transcendental)

| 标量 (`__half`) | 打包 (`__half2`) | 功能 |
|----------------|-----------------|------|
| `hexp(a)` | `h2exp(a)` | e^x |
| `hexp2(a)` | `h2exp2(a)` | 2^x |
| `hexp10(a)` | `h2exp10(a)` | 10^x |
| `hlog(a)` | `h2log(a)` | ln(x) |
| `hlog2(a)` | `h2log2(a)` | log2(x) |
| `hlog10(a)` | `h2log10(a)` | log10(x) |
| `hsqrt(a)` | `h2sqrt(a)` | √x |
| `hrsqrt(a)` | `h2rsqrt(a)` | 1/√x |
| `hrcp(a)` | `h2rcp(a)` | 1/x |
| `hsin(a)` | `h2sin(a)` | sin(x) |
| `hcos(a)` | `h2cos(a)` | cos(x) |
| `hrint(a)` | `h2rint(a)` | 四舍五入 |
| `hceil(a)` | `h2ceil(a)` | 向上取整 |
| `hfloor(a)` | `h2floor(a)` | 向下取整 |

> **注意：** Transcendental 函数在部分架构上会被提升到 FP32 经由 SFU 计算，不一定获得 2× 加速。加减乘 FMA 是稳定获得 2× 吞吐的操作。

### 4.4 转换与构造

```cpp
// ============= 构造 half2 =============
half2 h2 = make_half2(1.0f, 2.0f);            // 从两个 float
half2 h2 = __half2half2(h);                     // broadcast: (h, h)
half2 h2 = __float2half2_rn(f);                // broadcast: (f→h, f→h)
half2 h2 = __floats2half2_rn(f1, f2);          // 两个 float → half2
half2 h2 = __halves2half2(h_lo, h_hi);         // 两个 half → half2

// ============= 提取 =============
half lo = __low2half(h2);                       // 提取低半
half hi = __high2half(h2);                      // 提取高半
float2 f2 = __half22float2(h2);                 // 转 float2

// ============= half2 ↔ 原始位操作 =============
unsigned int bits = __half2_as_uint(h2);        // reinterpret as uint
half2 h2 = __uint_as_half2(bits);               // uint → half2
```

### 4.5 标量 half vs. 打包 half2 性能对比

```
场景: 对 N 个 half 执行 FMA (a*x+b)

标量 half:  每指令 1 个 FMA  → N 条 HFMA 指令
打包 half2: 每指令 2 个 FMA  → N/2 条 HFMA2 指令

实测加速比 (相对 FP32):
  - 标量 half:  ~1.0× (与 FP32 相同, 因为使用相同 ALU 通路)
  - 打包 half2: ~1.3–1.8× (理论 2×, 受内存和其他开销限制)
```

> **规则：永远使用 packed half2, 避免标量 half。** 标量 half 在 CUDA Core 上无吞吐优势。

---

## 5. 编译器自动向量化与控制

### 5.1 Global Memory: 不自动向量化

`nvcc` / `ptxas` **不会**自动将 Global Memory 的标量加载合并为向量化加载。原因：编译器无法在编译时确定 Global Memory 指针的对齐和别名情况。

```cpp
// ❌ 编译器不会自动向量化 (4 条 LDG.E.32)
__global__ void kernel(float* data) {
    float a = data[threadIdx.x * 4 + 0];
    float b = data[threadIdx.x * 4 + 1];
    float c = data[threadIdx.x * 4 + 2];
    float d = data[threadIdx.x * 4 + 3];
    // → 4 × LDG.E.32
}

// ✅ 手动向量化 (1 条 LDG.E.128)
__global__ void kernel_vec(float* data) {
    float4 v = reinterpret_cast<float4*>(data)[threadIdx.x];
    // → 1 × LDG.E.128
}
```

### 5.2 `__restrict__` 和 `--restrict`

`__restrict__` 告知编译器指针不与其他指针别名，使编译器能够重排和合并内存访问：

```cpp
// ❌ 可能别名: 编译器不敢重排 load/store
__global__ void add(float* out, float* a, float* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] + b[i];
}

// ✅ __restrict__: 保证无别名 → 编译器可优化
__global__ void add(float* __restrict__ out,
                    const float* __restrict__ a,
                    const float* __restrict__ b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] + b[i];
}
```

全局开关: `nvcc --restrict` 将所有 kernel 指针参数自动标记为 `__restrict__`。

> **注意：** `__restrict__` 帮助编译器优化指令调度和消除冗余加载，但**不会**自动将标量加载升级为向量化加载。向量化仍需手动。

### 5.3 `__builtin_assume_aligned`

告知编译器指针满足特定对齐，可能触发自动向量化：

```cpp
__global__ void kernel(float* data) {
    // 告诉编译器 data 是 16 字节对齐的
    float* aligned_data = (float*)__builtin_assume_aligned(data, 16);

    // 编译器 *可能* 将连续加载合并为 LDG.E.128
    float a = aligned_data[threadIdx.x * 4 + 0];
    float b = aligned_data[threadIdx.x * 4 + 1];
    float c = aligned_data[threadIdx.x * 4 + 2];
    float d = aligned_data[threadIdx.x * 4 + 3];
}
```

CUDA 11.2+ 支持此内建函数。效果不保证，建议与 `--extra-device-vectorization` 配合使用。

### 5.4 `--extra-device-vectorization`

`nvcc` 编译选项，启用更激进的向量化优化：

```bash
nvcc --extra-device-vectorization -o kernel kernel.cu
```

此选项使编译器在检测到连续内存访问模式且满足对齐条件时，尝试生成向量化指令。通常需配合 `__restrict__` 和 `__builtin_assume_aligned` 使用。

### 5.5 `__builtin_assume`

通用条件提示，帮助编译器做循环展开和向量化决策：

```cpp
__global__ void kernel(float* data, int N) {
    __builtin_assume(N % 4 == 0);  // N 是 4 的倍数
    // 编译器知道不需要处理尾部
    for (int i = threadIdx.x * 4; i < N; i += blockDim.x * 4) {
        // ...
    }
}
```

### 5.6 Shared Memory: 可能自动向量化

与 Global Memory 不同，`ptxas` **能**自动向量化 Shared Memory 访问，因为编译器知道 `__shared__` 数组的物理对齐：

```cpp
__shared__ float smem[1024];

// ptxas 可能自动合并为 LDS.128:
float a = smem[threadIdx.x * 4 + 0];
float b = smem[threadIdx.x * 4 + 1];
float c = smem[threadIdx.x * 4 + 2];
float d = smem[threadIdx.x * 4 + 3];
```

但这种优化不保证发生。显式使用 `float4*` 是最可靠的方式。

### 5.7 检查向量化结果

```bash
# 生成 PTX 检查中间表示
nvcc -ptx kernel.cu -o kernel.ptx
# 查看: ld.global.v4.f32 (向量化) vs ld.global.f32 (标量)

# 生成 SASS 检查最终机器码
nvcc -cubin kernel.cu && cuobjdump -sass kernel.cubin
# 查看: LDG.E.128 (向量化) vs LDG.E.32 (标量)

# 或用 nvdisasm 获取更详细信息
nvdisasm kernel.cubin
```

**PTX 向量化标志：**

```
ld.global.v4.f32  → 128-bit 向量化加载 (4 × f32)
ld.global.v2.f32  → 64-bit 向量化加载 (2 × f32)
ld.global.f32     → 标量加载

st.shared.v4.f32  → 128-bit 向量化 shared 存储
```

> **注意：** PTX 层的向量化不保证 SASS 层也是向量化的 (ptxas 可能拆分)，反之亦然 (ptxas 可能合并标量 PTX 为向量化 SASS)。以 SASS 为准。

---

## 6. FP8 打包与向量化

### 6.1 FP8 数据类型

CUDA 提供原生 FP8 类型 (CUDA 11.8+, `cuda_fp8.h`)：

| 类型 | 格式 | 范围 | 精度 | 用途 |
|------|------|------|------|------|
| `__nv_fp8_e4m3` | 1+4+3 | ±240 | 较高 | 推理 / 前向 |
| `__nv_fp8_e5m2` | 1+5+2 | ±57344 | 较低 | 训练 / 反向梯度 |

### 6.2 FP8 打包类型

| 打包类型 | 大小 | 含元素数 | 等效标量 |
|---------|------|---------|---------|
| `__nv_fp8x2_e4m3` | 16 bit (2B) | 2 个 FP8 | 2 × `__nv_fp8_e4m3` |
| `__nv_fp8x4_e4m3` | 32 bit (4B) | 4 个 FP8 | 4 × `__nv_fp8_e4m3` |
| `__nv_fp8x2_e5m2` | 16 bit (2B) | 2 个 FP8 | 2 × `__nv_fp8_e5m2` |
| `__nv_fp8x4_e5m2` | 32 bit (4B) | 4 个 FP8 | 4 × `__nv_fp8_e5m2` |

> 没有官方 `__nv_fp8x8` 或 `__nv_fp8x16` 类型。更宽的打包使用通用类型 (`uint2`, `uint4`) 实现。

### 6.3 FP8 向量化加载模式 (128-bit = 16 个 FP8)

```cpp
#include <cuda_fp8.h>

__global__ void fp8_dequant_kernel(
    half* __restrict__ output,
    const __nv_fp8_e4m3* __restrict__ input,
    float scale,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 16;  // 每线程处理 16 个 FP8
    if (idx + 15 < N) {
        // Step 1: 128-bit 向量化加载 → 16 个 FP8 值
        uint4 packed = reinterpret_cast<const uint4*>(input)[tid];

        // Step 2: 将 uint4 (4 × uint32) 解释为 4 组 __nv_fp8x4_e4m3
        __nv_fp8x4_e4m3* fp8_groups = reinterpret_cast<__nv_fp8x4_e4m3*>(&packed);

        // Step 3: 逐组转换为 float 并缩放
        half results[16];
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            // __nv_fp8x4_e4m3 内置 operator float4() 转换
            float4 f4 = static_cast<float4>(fp8_groups[g]);
            results[g * 4 + 0] = __float2half(f4.x * scale);
            results[g * 4 + 1] = __float2half(f4.y * scale);
            results[g * 4 + 2] = __float2half(f4.z * scale);
            results[g * 4 + 3] = __float2half(f4.w * scale);
        }

        // Step 4: 128-bit 向量化存储 (16 × half = 32B → 两次 uint4 存储)
        reinterpret_cast<uint4*>(output)[tid * 2]     = reinterpret_cast<uint4*>(results)[0];
        reinterpret_cast<uint4*>(output)[tid * 2 + 1] = reinterpret_cast<uint4*>(results)[1];
    }
}
```

### 6.4 FP8 转换指令

| 操作 | C++ API | PTX 指令 |
|------|---------|----------|
| FP8 → half (单个) | `__nv_cvt_fp8_to_halfraw(storage, __NV_E4M3)` | `cvt.rn.f16.e4m3` |
| 2×FP8 → half2 | 隐式转换 `__nv_fp8x2_e4m3` → `half2` | `cvt.rn.f16x2.e4m3x2` |
| 4×FP8 → float4 | 隐式转换 `__nv_fp8x4_e4m3` → `float4` | 多条指令 |
| half → FP8 | 构造函数 `__nv_fp8_e4m3(half_val)` | `cvt.rn.satfinite.e4m3.f16` |
| half2 → 2×FP8 | 构造函数 `__nv_fp8x2_e4m3(half2_val)` | `cvt.rn.satfinite.e4m3x2.f16x2` |

### 6.5 FP8 向量化中的带宽优势

```
FP32 场景: 每线程 float4 → 4 个 FP32 / 16 字节
FP8 场景:  每线程 uint4  → 16 个 FP8 / 16 字节

同样的内存带宽, FP8 传输 4× 数据量
→ 计算绑定时: FP8 Tensor Core 吞吐也是 FP16 的 2× → 整体 ~2× 加速
→ 内存绑定时: 数据搬运减少 4× → 更显著的加速
```

---

## 7. FP4 (NVFP4) 打包与向量化

### 7.1 NVFP4 数据格式

```
FP4 (E2M1): 1 sign + 2 exponent + 1 mantissa = 4 bits/值
→ 2 个 FP4 打包在 1 个字节中

双级缩放 (Two-Level Scaling):
  Level 1: FP8 E4M3 微块缩放因子 (每 16 个 FP4 值共享 1 个 FP8 scale)
  Level 2: FP32 张量级缩放因子 (所有值共享)

等效 bits/值: 4 + 8/16 = 4.5 bits
```

### 7.2 NVFP4 内存布局

```
16 个 FP4 值 = 1 个微块 (micro-block):

字节排布 (小端):
[byte0]  [byte1]  [byte2]  [byte3]  [byte4]  [byte5]  [byte6]  [byte7]  [scale]
 v0|v1    v2|v3    v4|v5    v6|v7    v8|v9   v10|v11  v12|v13  v14|v15  FP8 E4M3
 ←─────────── 8 字节 (16 个 FP4 值) ──────────→        ←1B→

→ 每个微块 9 字节 (8 字节数据 + 1 字节 FP8 scale)
```

### 7.3 FP4 向量化加载

```cpp
// 128-bit 加载: 16 字节 = 32 个 FP4 值 (不含 scale)
// 实际内存布局需要交错读取数据和 scale

__global__ void fp4_dequant(
    half* __restrict__ output,
    const uint8_t* __restrict__ fp4_data,      // 打包的 FP4 数据
    const __nv_fp8_e4m3* __restrict__ scales,  // 微块 scale
    float tensor_scale,                         // 张量级 scale
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int micro_block = tid;  // 每线程处理 1 个微块 (16 个 FP4 值)
    int idx = micro_block * 16;

    if (idx + 15 < N) {
        // 加载 8 字节 FP4 数据 (16 个 FP4 值, 2 个一字节)
        uint2 packed = reinterpret_cast<const uint2*>(fp4_data)[micro_block];

        // 加载 1 字节 scale
        __nv_fp8_e4m3 block_scale = scales[micro_block];
        float scale = float(block_scale) * tensor_scale;

        // 解包: 每字节 2 个 FP4
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&packed);
        half results[16];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint8_t byte = bytes[i];
            // 低 4 位 = 第一个 FP4, 高 4 位 = 第二个 FP4
            float v_lo = fp4_to_float(byte & 0x0F) * scale;
            float v_hi = fp4_to_float(byte >> 4) * scale;
            results[i * 2 + 0] = __float2half(v_lo);
            results[i * 2 + 1] = __float2half(v_hi);
        }

        // 向量化存储
        reinterpret_cast<uint4*>(output)[micro_block * 2]     =
            reinterpret_cast<uint4*>(results)[0];
        reinterpret_cast<uint4*>(output)[micro_block * 2 + 1] =
            reinterpret_cast<uint4*>(results)[1];
    }
}
```

### 7.4 FP4 转换指令

| 操作 | PTX 指令 | 说明 |
|------|----------|------|
| 2×FP4 → half2 | `cvt.rn.f16x2.e2m1x2` | 1 字节 → 2 个 half |
| half2 → 2×FP4 | `cvt.rn.satfinite.e2m1x2.f16x2` | 2 个 half → 1 字节 |

C++ API: `__nv_cvt_fp4x2_to_halfraw2` 将 1 字节 (2 个 FP4) 转换为 `half2`。

### 7.5 FP4 的带宽优势

```
FP32: 128-bit 加载 → 4 个值
FP16: 128-bit 加载 → 8 个值
FP8:  128-bit 加载 → 16 个值
FP4:  128-bit 加载 → 32 个值

→ 同等内存带宽下, FP4 传输 8× 于 FP32, 4× 于 FP16
→ 对于 weight-only quantization 推理, 减少 weight 加载的内存流量是核心收益
```

---

## 8. LLM Kernel 向量化实战模式

### 8.1 RMSNorm 向量化 (FP16/BF16)

RMSNorm 是 LLaMA 系列模型的基础组件。它是典型的 memory-bound kernel——计算量 O(D) 但需要 2 次遍历 (计算方差 + 归一化)。

```cpp
// ============= vLLM 风格: width-8 向量化 RMSNorm =============
template <typename scalar_t>  // half 或 __nv_bfloat16
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    float epsilon,
    int hidden_size
) {
    // 每个 block 处理一行 (一个 token)
    int row = blockIdx.x;
    const scalar_t* row_in = input + row * hidden_size;
    scalar_t* row_out = output + row * hidden_size;

    using Vec = uint4;  // 128-bit = 8 × half
    constexpr int VEC_WIDTH = 16 / sizeof(scalar_t);  // 8 for half/bf16
    int vec_count = hidden_size / VEC_WIDTH;

    // Pass 1: 计算 sum of squares (FP32 累加保证精度)
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
        Vec v = reinterpret_cast<const Vec*>(row_in)[i];
        // 将 128-bit 数据解释为 4 个 half2 进行 FP32 累加
        half2* h2 = reinterpret_cast<half2*>(&v);
        #pragma unroll
        for (int j = 0; j < VEC_WIDTH / 2; j++) {
            float2 f2 = __half22float2(h2[j]);
            sum_sq += f2.x * f2.x + f2.y * f2.y;
        }
    }

    // Warp-level reduce
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Block-level reduce (via shared memory)
    __shared__ float warp_results[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_results[warp_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x < blockDim.x / warpSize) {
        sum_sq = warp_results[threadIdx.x];
        for (int offset = blockDim.x / warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        }
    }

    __shared__ float rsqrt_val;
    if (threadIdx.x == 0) {
        rsqrt_val = rsqrtf(sum_sq / hidden_size + epsilon);
    }
    __syncthreads();

    float scale = rsqrt_val;

    // Pass 2: 归一化 + 乘 weight (向量化)
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
        Vec v_in = reinterpret_cast<const Vec*>(row_in)[i];
        Vec v_w  = reinterpret_cast<const Vec*>(weight)[i * VEC_WIDTH % hidden_size == 0
                    ? i : i];  // weight 沿 hidden_dim 对齐
        v_w = reinterpret_cast<const Vec*>(weight)[i];

        Vec v_out;
        half2* h2_in  = reinterpret_cast<half2*>(&v_in);
        half2* h2_w   = reinterpret_cast<half2*>(&v_w);
        half2* h2_out = reinterpret_cast<half2*>(&v_out);
        half2 scale2 = __float2half2_rn(scale);

        #pragma unroll
        for (int j = 0; j < VEC_WIDTH / 2; j++) {
            h2_out[j] = __hmul2(__hmul2(h2_in[j], scale2), h2_w[j]);
        }

        reinterpret_cast<Vec*>(row_out)[i] = v_out;
    }
}
```

**实测收益：** 向量化 (width=8) 比标量版本提升 **2–2.5×**，主要来自指令数减少和 L1 tag 查找减少。

### 8.2 Fused Residual Add + RMSNorm

将 residual add 和 RMSNorm 融合为一个 kernel，消除中间结果的全局内存写回：

```cpp
// ============= Fused: residual_add + rms_norm =============
// 计算: residual = residual + input
//        output = rms_norm(residual, weight)
// 节省: 1 次 residual 全局写 + 1 次全局读 (vs 两个独立 kernel)

template <typename scalar_t>
__global__ void fused_add_rms_norm_kernel(
    scalar_t* __restrict__ output,       // 归一化结果
    scalar_t* __restrict__ residual,     // 就地更新的 residual
    const scalar_t* __restrict__ input,  // 当前层输入
    const scalar_t* __restrict__ weight, // RMSNorm 权重
    float epsilon,
    int hidden_size
) {
    int row = blockIdx.x;
    using Vec = uint4;
    constexpr int VEC_W = 16 / sizeof(scalar_t);
    int vec_count = hidden_size / VEC_W;

    // Pass 1: residual add + 计算 sum_sq
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
        Vec v_res = reinterpret_cast<const Vec*>(residual + row * hidden_size)[i];
        Vec v_inp = reinterpret_cast<const Vec*>(input + row * hidden_size)[i];

        Vec v_sum;
        half2* h2_res = reinterpret_cast<half2*>(&v_res);
        half2* h2_inp = reinterpret_cast<half2*>(&v_inp);
        half2* h2_sum = reinterpret_cast<half2*>(&v_sum);

        #pragma unroll
        for (int j = 0; j < VEC_W / 2; j++) {
            h2_sum[j] = __hadd2(h2_res[j], h2_inp[j]);  // packed add
            float2 f2 = __half22float2(h2_sum[j]);
            sum_sq += f2.x * f2.x + f2.y * f2.y;
        }

        // 写回更新的 residual
        reinterpret_cast<Vec*>(residual + row * hidden_size)[i] = v_sum;
    }

    // ... (warp/block reduce, 计算 rsqrt, 同上) ...
    // Pass 2: 归一化 (从 residual 读取更新后的值)
}
```

**内存流量对比：**

```
独立 kernel:
  Kernel 1 (residual add):  读 residual + input → 写 residual  = 3D 字节
  Kernel 2 (rms_norm):      读 residual + weight → 写 output   = 3D 字节
  总计: 6D 字节

Fused kernel:
  读 residual + input + weight → 写 residual + output = 5D 字节
  节省: ~17% 内存流量 + 1 次 kernel launch 开销
```

### 8.3 SiLU (Swish) 与 SwiGLU 向量化

**SiLU: `silu(x) = x × sigmoid(x) = x × 1/(1 + e^(-x))`**

```cpp
// ============= Packed half2 SiLU =============
__device__ __forceinline__ half2 silu_half2(half2 x) {
    half2 neg_x = __hneg2(x);
    half2 exp_neg = h2exp(neg_x);           // e^(-x)
    half2 one = __float2half2_rn(1.0f);
    half2 denom = __hadd2(one, exp_neg);     // 1 + e^(-x)
    half2 sigmoid = h2rcp(denom);            // 1 / (1 + e^(-x))
    return __hmul2(x, sigmoid);              // x × sigmoid(x)
}

// ============= SwiGLU: silu(gate) × up =============
// LLaMA: FFN 输出 = SwiGLU(x) = silu(W_gate × x) ⊙ (W_up × x)
__global__ void swiglu_kernel(
    half* __restrict__ output,
    const half* __restrict__ gate,  // W_gate × x, shape [N, D]
    const half* __restrict__ up,    // W_up × x,   shape [N, D]
    int D
) {
    int row = blockIdx.x;
    constexpr int VEC_W = 8;  // 8 × half = 128 bit
    int vec_count = D / VEC_W;

    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
        uint4 v_gate = reinterpret_cast<const uint4*>(gate + row * D)[i];
        uint4 v_up   = reinterpret_cast<const uint4*>(up + row * D)[i];

        half2* h2_gate = reinterpret_cast<half2*>(&v_gate);
        half2* h2_up   = reinterpret_cast<half2*>(&v_up);
        uint4 v_out;
        half2* h2_out = reinterpret_cast<half2*>(&v_out);

        #pragma unroll
        for (int j = 0; j < VEC_W / 2; j++) {
            half2 activated = silu_half2(h2_gate[j]);
            h2_out[j] = __hmul2(activated, h2_up[j]);  // silu(gate) × up
        }

        reinterpret_cast<uint4*>(output + row * D)[i] = v_out;
    }
}
```

### 8.4 GELU 向量化

**GELU (tanh 近似): `gelu(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))`**

```cpp
__device__ __forceinline__ half2 gelu_half2(half2 x) {
    // 常量 (broadcast)
    half2 half_val = __float2half2_rn(0.5f);
    half2 one      = __float2half2_rn(1.0f);
    half2 coeff    = __float2half2_rn(0.044715f);
    half2 sqrt_2pi = __float2half2_rn(0.7978845608f);  // sqrt(2/pi)

    // x^3
    half2 x2 = __hmul2(x, x);
    half2 x3 = __hmul2(x2, x);

    // sqrt(2/pi) * (x + 0.044715 * x^3)
    half2 inner = __hfma2(coeff, x3, x);       // x + 0.044715 * x^3
    inner = __hmul2(sqrt_2pi, inner);           // √(2/π) × (...)

    // 注意: 没有原生 half2 tanh, 需转 FP32 或近似
    float2 f_inner = __half22float2(inner);
    float2 f_tanh = make_float2(tanhf(f_inner.x), tanhf(f_inner.y));
    half2 tanh_val = __float22half2_rn(f_tanh);

    // 0.5 * x * (1 + tanh(...))
    half2 result = __hmul2(half_val, __hmul2(x, __hadd2(one, tanh_val)));
    return result;
}
```

> **注意：** `tanh` 没有原生 `half2` 版本, 通常需提升到 FP32 使用 SFU 计算。对于高精度需求, 这部分不会获得 packed 加速。FastGELU (ReLU 近似) 可完全在 half2 中完成。

### 8.5 FlashAttention 中的向量化

FlashAttention 在不同层次使用不同的向量化策略：

```
Global → Shared Memory:
  Ampere: cp.async (异步拷贝, 128-bit, 绕过寄存器)
  Hopper: TMA (硬件张量搬运, 更高效)

Shared Memory → Register (用于 MMA):
  ldmatrix (PTX): Warp 协作加载 128 字节到 MMA fragment
  → 每线程 16B, 32 线程 = 512B / 指令
  → 自动 Swizzle 对齐 Tensor Core 需要的布局

Register-Level 计算:
  Softmax 的 online reduction: 使用 half2 packed 比较/max/add
  Rescaling: half2 packed multiply 批量缩放 attention weights

Shared Memory → Register (非 MMA):
  FlashAttention 的 O 累加器更新使用 float4 向量化读写 shared memory
```

### 8.6 Quantized GEMV (FP8/FP4 Weight-Only)

推理中常见的 weight-only quantization GEMV (batch_size=1):

```cpp
// ============= FP8 Weight-Only GEMV =============
// y = dequant(W_fp8) × x_fp16
// W: [N, K] in FP8, x: [K] in FP16, y: [N] in FP16
//
// 每 Block 计算 y 的一个元素, 线程沿 K 维展开

__global__ void fp8_gemv(
    half* __restrict__ y,
    const __nv_fp8_e4m3* __restrict__ W,  // [N, K]
    const half* __restrict__ x,           // [K]
    float w_scale,
    int N, int K
) {
    int row = blockIdx.x;  // 输出维度
    float acc = 0.0f;

    // 每线程处理 K 维的一段, 每次加载 16 个 FP8 权重
    for (int k = threadIdx.x * 16; k < K; k += blockDim.x * 16) {
        // 向量化加载 16 个 FP8 权重
        uint4 w_packed = reinterpret_cast<const uint4*>(W + row * K)[k / 16];
        __nv_fp8x4_e4m3* fp8_groups = reinterpret_cast<__nv_fp8x4_e4m3*>(&w_packed);

        // 向量化加载 16 个 FP16 输入 (两次 uint4)
        uint4 x_packed0 = reinterpret_cast<const uint4*>(x)[k / 8];
        uint4 x_packed1 = reinterpret_cast<const uint4*>(x)[k / 8 + 1];
        half2* h2_x0 = reinterpret_cast<half2*>(&x_packed0);
        half2* h2_x1 = reinterpret_cast<half2*>(&x_packed1);

        #pragma unroll
        for (int g = 0; g < 4; g++) {
            float4 w_f4 = static_cast<float4>(fp8_groups[g]);
            half2* h2_x = (g < 2) ? &h2_x0[g * 2] : &h2_x1[(g - 2) * 2];
            float2 x_f2_0 = __half22float2(h2_x[0]);
            float2 x_f2_1 = __half22float2(h2_x[1]);

            acc += w_f4.x * w_scale * x_f2_0.x;
            acc += w_f4.y * w_scale * x_f2_0.y;
            acc += w_f4.z * w_scale * x_f2_1.x;
            acc += w_f4.w * w_scale * x_f2_1.y;
        }
    }

    // Warp reduce
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (threadIdx.x % warpSize == 0) {
        atomicAdd(reinterpret_cast<half*>(&y[row]),
                  __float2half(acc));
    }
}
```

### 8.7 Softmax 向量化

```cpp
// ============= Online Softmax with Vectorized half2 =============
__global__ void softmax_half2(
    half* __restrict__ output,
    const half* __restrict__ input,
    int seq_len  // 必须是偶数
) {
    int row = blockIdx.x;
    const half2* row_in = reinterpret_cast<const half2*>(input + row * seq_len);
    half2* row_out = reinterpret_cast<half2*>(output + row * seq_len);
    int half2_count = seq_len / 2;

    // Pass 1: 找 max (使用 packed max)
    half2 max_val = __float2half2_rn(-65504.0f);  // half 最小值
    for (int i = threadIdx.x; i < half2_count; i += blockDim.x) {
        half2 v = row_in[i];
        max_val = __hmax2(max_val, v);
    }
    // Warp reduce max (需要标量化)
    float max_f = fmaxf(__half2float(__low2half(max_val)),
                        __half2float(__high2half(max_val)));
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        max_f = fmaxf(max_f, __shfl_down_sync(0xFFFFFFFF, max_f, offset));
    }
    // ... block reduce ...

    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = max_f;
    __syncthreads();
    half2 max2 = __float2half2_rn(shared_max);

    // Pass 2: 计算 exp(x - max) 并求和
    float sum = 0.0f;
    for (int i = threadIdx.x; i < half2_count; i += blockDim.x) {
        half2 v = __hsub2(row_in[i], max2);
        half2 e = h2exp(v);  // packed exp
        float2 ef = __half22float2(e);
        sum += ef.x + ef.y;
        // 可将 exp 结果缓存到 shared memory 避免重算
    }
    // ... reduce sum ...

    // Pass 3: 归一化
    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = sum;
    __syncthreads();
    half2 inv_sum2 = __float2half2_rn(1.0f / shared_sum);

    for (int i = threadIdx.x; i < half2_count; i += blockDim.x) {
        half2 v = __hsub2(row_in[i], max2);
        half2 e = h2exp(v);
        row_out[i] = __hmul2(e, inv_sum2);
    }
}
```

---

## 9. 寄存器压力与向量化权衡

### 9.1 寄存器代价

向量化增加每线程使用的寄存器数量：

```
标量:   每个 float 占 1 个 32-bit 寄存器
float4: 4 个 float 占 4 个 32-bit 寄存器

示例: GEMM tile 加载
  标量: 加载 1 个 A 元素 → 1 reg
  float4: 加载 4 个 A 元素 → 4 regs

如果同时缓存多个向量化加载:
  4 × float4 = 16 个寄存器 → 对 occupancy 的影响不可忽略
```

### 9.2 Occupancy 影响

| SM 版本 | 寄存器文件/SM | 最大 Reg/线程 | 最大线程/SM |
|---------|-------------|-------------|-----------|
| SM 8.0 (A100) | 256 KB | 255 | 2048 |
| SM 9.0 (H100) | 256 KB | 255 | 2048 |

```
Occupancy 计算:
  寄存器/线程 = 64 → 256KB / (64×4B) = 1024 线程/SM → 50% occupancy
  寄存器/线程 = 80 → 256KB / (80×4B) = 819 线程/SM → 40% occupancy
  寄存器/线程 = 128 → 256KB / (128×4B) = 512 线程/SM → 25% occupancy

向量化增加 20-40 个寄存器 → occupancy 下降 5-15%
```

### 9.3 何时向量化值得

| 场景 | 向量化收益 | 寄存器代价 | 推荐 |
|------|-----------|-----------|------|
| **Memory-bound elementwise** (RMSNorm, Residual, Scale) | 高 (减少指令数 4×) | 低 (仅增加少量临时 reg) | ✅ 强烈推荐 |
| **GEMM tile 加载** | 中 (减少加载指令) | 中 (需要临时 buffer) | ✅ 通常值得 |
| **Compute-bound + 大寄存器需求** (大 tile GEMM 内循环) | 低 (瓶颈在计算) | 高 (可能导致 spill) | ⚠️ 需要 profiling |
| **高 occupancy 需求** (latency-hiding 场景) | 中 | 高 (降低 occupancy) | ⚠️ 权衡 |

### 9.4 避免寄存器溢出 (Register Spill)

```cpp
// ❌ 同时缓存过多向量化数据 → 寄存器溢出到 local memory
float4 a0 = load_a[0], a1 = load_a[1], a2 = load_a[2], a3 = load_a[3];
float4 b0 = load_b[0], b1 = load_b[1], b2 = load_b[2], b3 = load_b[3];
// 32 个寄存器仅用于临时数据

// ✅ 加载后立即消费, 减少活跃寄存器
for (int i = 0; i < 4; i++) {
    float4 a = load_a[i];
    process(a);  // 立即使用后 a 的寄存器可复用
}
```

监控寄存器使用：

```bash
# 编译时显示寄存器使用
nvcc --ptxas-options=-v kernel.cu
# 输出: Used 48 registers, ...

# 限制最大寄存器数 (以提升 occupancy)
nvcc --maxrregcount=64 kernel.cu
# 注意: 可能导致 spill → 适得其反
```

---

## 10. 向量化检查清单与最佳实践

### 10.1 开发检查清单

**内存访问向量化：**

- [ ] 所有 elementwise kernel 的 Global Memory 加载使用 `float4` / `uint4` (128-bit)
- [ ] FP16/BF16 kernel 使用 `uint4` 加载 8 个 half 值
- [ ] 数据量 (hidden_dim, seq_len 等) 是向量宽度的倍数 (或有尾部回退处理)
- [ ] `cudaMalloc` 偏移后的子数组仍满足 16B 对齐
- [ ] SASS 输出确认为 `LDG.E.128` (而非多个 `LDG.E.32`)

**向量化计算：**

- [ ] FP16/BF16 计算使用 `half2` / `__nv_bfloat162` packed 指令
- [ ] 避免标量 `half` 运算 (无吞吐优势)
- [ ] 减少 half2 ↔ float2 不必要的来回转换
- [ ] 精度敏感的累加 (sum, variance) 在 FP32 中进行

**编译器配合：**

- [ ] 所有指针参数标记 `__restrict__` 或使用 `--restrict`
- [ ] 考虑使用 `__builtin_assume_aligned(ptr, 16)` 辅助编译器
- [ ] 内循环使用 `#pragma unroll` 展开
- [ ] 检查 `nvcc --ptxas-options=-v` 的寄存器使用和 spill 报告

### 10.2 常见陷阱

```cpp
// 🚩 陷阱 1: 使用标量 half (无吞吐优势)
half a = input[i];
half b = a * weight[i];  // HMUL, 吞吐 = FP32 的 1×
// ✅ 使用 half2
half2 a = reinterpret_cast<const half2*>(input)[i];

// 🚩 陷阱 2: 未对齐的 reinterpret_cast
float4 v = reinterpret_cast<float4*>(ptr + 3)[0];  // ptr+3 不 16B 对齐 → UB
// ✅ 确保偏移是 4 的倍数 (对于 float4)

// 🚩 陷阱 3: 依赖编译器自动向量化 Global Memory
float a = data[tid * 4 + 0];  // 期望合并为 float4 → 不会发生
float b = data[tid * 4 + 1];
float c = data[tid * 4 + 2];
float d = data[tid * 4 + 3];
// ✅ 显式使用 float4
float4 v = reinterpret_cast<float4*>(data)[tid];

// 🚩 陷阱 4: 过度向量化导致寄存器溢出
// 同时加载 8 个 float4 (32 regs) + 大量计算中间变量 → spill
// ✅ 限制同时活跃的向量化数据量

// 🚩 陷阱 5: transcendental 函数期望 packed 加速
half2 result = h2exp(x);  // 可能被提升到 FP32 → SFU → 无 2× 加速
// ✅ 加减乘 FMA 才有稳定 2× 加速; transcendental 加速不保证
```

### 10.3 向量化宽度选择决策树

```
选择向量化宽度:

数据类型是 FP32?
├── 是 → 使用 float4 (128-bit, 4 个 float)
│        数据量不是 4 的倍数? → 尾部标量回退
└── 否 → 数据类型是 FP16/BF16?
          ├── 是 → 使用 uint4 加载 8 个 half (128-bit)
          │        逐对用 half2 packed 计算
          │        数据量不是 8 的倍数? → 尾部回退
          └── 否 → 数据类型是 FP8?
                    ├── 是 → 使用 uint4 加载 16 个 FP8 (128-bit)
                    │        转换为 float4 / half2 后计算
                    └── 否 → 数据类型是 FP4?
                              └── 是 → 使用 uint4 加载 32 个 FP4 (128-bit)
                                       需同时加载 scale 数据
```

### 10.4 性能验证

```bash
# 确认向量化 SASS 指令
cuobjdump -sass kernel.cubin | grep -E "LDG|STG|LDS|STS"
# 期望: LDG.E.128, STG.E.128, LDS.128
# 问题: 大量 LDG.E.32 表示未向量化

# 确认无寄存器溢出
nvcc --ptxas-options=-v kernel.cu 2>&1 | grep -E "registers|spill"
# 期望: 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads

# Nsight Compute 带宽利用率
ncu --metrics \
    dram__bytes_read.sum.per_second,\
    dram__bytes_write.sum.per_second,\
    sm__inst_executed.sum \
    ./my_kernel
# 向量化后: inst_executed 应显著减少, 带宽利用率应接近峰值
```

---

## 参考资源

- [NVIDIA Blog: CUDA Pro Tip — Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
- [NVIDIA Blog: CUDA Pro Tip — Optimize Pointer Aliasing](https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing/)
- [NVIDIA Blog: Boosting Productivity and Performance with the NVIDIA CUDA 11.2 C++ Compiler](https://developer.nvidia.com/blog/boosting-productivity-and-performance-with-the-nvidia-cuda-11-2-c-compiler/)
- [NVIDIA Blog: Mixed-Precision Programming with CUDA 8](https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/)
- [NVIDIA Blog: Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [CUDA Math API: Half2 Arithmetic Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__ARITHMETIC.html)
- [CUDA Math API: Half2 Math Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__FUNCTIONS.html)
- [CUDA Math API: Half2 Comparison Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__COMPARISON.html)
- [CUDA Math API: FP8 Types and Conversions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html)
- [Lei Mao: CUDA Shared Memory Bank Conflict-Free Vectorized Access](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank-Conflict-Free-Vectorized-Access/)
- [vLLM: Vectorize RMSNorm CUDA Kernel (PR #22602)](https://github.com/vllm-project/vllm/pull/22602)
- [Twelve Attempts at an NVFP4 Kernel](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)
- [Scaleway: Understanding NVIDIA FP8 Format](https://www.scaleway.com/en/docs/gpu/reference-content/understanding-nvidia-fp8/)
- [CUDA Binary Utilities Documentation](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)

---

*本文档作为 LLM Kernel Agent 的向量化技能参考。与 `coalesced-memory-access.md`（合并访问基础与向量化加载概述）、`conflict-free-accesses.md`（Shared Memory Bank Conflict）配合使用。本文档聚焦向量化的深度技术细节：Packed Arithmetic、编译器行为、FP8/FP4 打包、LLM Kernel 实战模式。*
