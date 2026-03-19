# CUDA C++ Best Practices Guide — Kernel Agent Skill Reference

> 基于 [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) 编写。
> 面向 LLM 高性能 Kernel 自动生成场景，聚焦**可操作的性能优化策略**。
> 与 `cuda-c-programming-guide.md` 互补：前者侧重"是什么"，本文档侧重"怎么做到最快"。

---

## 目录

1. [APOD 优化方法论](#1-apod-优化方法论)
2. [性能度量与分析](#2-性能度量与分析)
3. [数据传输优化](#3-数据传输优化)
4. [内存优化](#4-内存优化)
5. [执行配置优化](#5-执行配置优化)
6. [指令优化](#6-指令优化)
7. [控制流优化](#7-控制流优化)
8. [浮点精度与数学运算](#8-浮点精度与数学运算)
9. [Profile 驱动优化](#9-profile-驱动优化)
10. [多 GPU 编程](#10-多-gpu-编程)
11. [部署与兼容性](#11-部署与兼容性)
12. [面向 LLM Kernel 的 Best Practices 检查清单](#12-面向-llm-kernel-的-best-practices-检查清单)

---

## 1. APOD 优化方法论

NVIDIA 推荐的 **APOD（Assess → Parallelize → Optimize → Deploy）** 循环迭代流程：

```
┌──────────┐     ┌──────────────┐     ┌──────────┐     ┌──────────┐
│ Assess   │ ──→ │ Parallelize  │ ──→ │ Optimize │ ──→ │ Deploy   │
│ 评估热点  │     │ 并行化实现    │     │ 逐层优化  │     │ 部署验证  │
└──────────┘     └──────────────┘     └──────────┘     └──────────┘
      ↑                                                       │
      └───────────────── 迭代循环 ←───────────────────────────┘
```

### 1.1 Assess（评估）

- 使用 Profiler（Nsight Systems/Compute）定位**热点函数**。
- 计算 Amdahl's Law 加速比上限：`Speedup_max = 1 / (1 - P + P/S)`
  - `P` = 可并行化部分占比，`S` = 并行加速倍数。
- **优先优化耗时占比最大的 Kernel**。

### 1.2 Parallelize（并行化）

三种策略（由易到难）：

| 策略 | 说明 | 示例 |
|------|------|------|
| 调用 GPU 优化库 | 零代码修改，直接替换 CPU 库 | cuBLAS, cuDNN, cuFFT, Thrust, CUB |
| 编译器指令 | 最小代码修改 | OpenACC `#pragma acc` |
| 手写 CUDA Kernel | 最大灵活性与性能 | 本文档核心场景 |

### 1.3 Optimize（优化）

按优先级排序的优化层次：

| 优先级 | 优化类别 | 预期收益 |
|--------|----------|----------|
| **最高** | 内存优化（合并访问、Shared Memory、向量化） | 数量级提升 |
| **高** | 执行配置（Block 大小、Occupancy） | 2–5× |
| **中** | 指令优化（fast math、循环展开、Tensor Core） | 1.5–3× |
| **低** | 控制流优化（减少 divergence） | 1.1–1.5× |

### 1.4 Deploy（部署）

- 验证数值精度（对比 CPU 参考实现）。
- 确保目标架构兼容（编译 flag）。
- 回归测试、CI 集成。

---

## 2. 性能度量与分析

### 2.1 计时方法

#### CUDA Event 计时（推荐）

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
myKernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);

float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel time: %.3f ms\n", ms);
```

#### CPU Timer 注意事项

```cuda
// WRONG: Kernel launch 是异步的，CPU timer 只测到 launch 开销
auto t0 = high_resolution_clock::now();
myKernel<<<grid, block>>>(args);
auto t1 = high_resolution_clock::now();  // ← 不包含 GPU 执行时间

// RIGHT: 必须同步
auto t0 = high_resolution_clock::now();
myKernel<<<grid, block>>>(args);
cudaDeviceSynchronize();  // ← 等待 GPU 完成
auto t1 = high_resolution_clock::now();
```

### 2.2 带宽计算

#### 理论峰值带宽

```
Theoretical BW (GB/s) = 内存时钟频率(GHz) × 内存位宽(bit) × 2(DDR) / 8(bits/byte)
```

| GPU | 内存类型 | 理论带宽 |
|-----|----------|----------|
| A100 (80GB) | HBM2e | 2039 GB/s |
| H100 (80GB) | HBM3 | 3350 GB/s |
| RTX 4090 | GDDR6X | 1008 GB/s |

#### 有效带宽

```
Effective BW (GB/s) = (Bytes_Read + Bytes_Written) / (Kernel_Time_s × 10⁹)
```

**最佳实践**：对于 memory-bound Kernel，有效带宽应达到理论峰值的 **60–80%** 以上。达不到说明存在内存访问低效（非合并、Bank Conflict 等）。

### 2.3 算术强度与 Roofline 模型

```
算术强度 (AI) = FLOPs / Bytes_Transferred

性能上界 = min(峰值算力, AI × 峰值带宽)
```

```
        Performance (FLOP/s)
         │
 Peak    │.......................─────────────── Compute Roof
 Compute │                    ╱
         │                  ╱
         │                ╱
         │              ╱    ← Memory Roof (slope = bandwidth)
         │            ╱
         │          ╱
         │        ╱   Ridge Point
         │      ╱     (AI = Peak_FLOPS / Peak_BW)
         │    ╱
         │  ╱
         │╱
         └──────────────────────────────── Arithmetic Intensity (FLOP/Byte)
```

| Kernel 类型 | 典型 AI | 瓶颈类型 |
|-------------|---------|----------|
| Element-wise (ReLU, GELU) | < 1 | Memory-bound |
| LayerNorm / Softmax | 1–10 | Memory-bound |
| GEMM (大矩阵) | 50–200 | Compute-bound |
| FlashAttention (fused) | 20–100 | Compute-bound |

**关键判断**：
- AI < Ridge Point → **Memory-bound**：优先优化内存访问模式。
- AI > Ridge Point → **Compute-bound**：优先提高算力利用率（Tensor Core、ILP）。

### 2.4 Nsight Compute 关键指标

| 指标 | 含义 | 理想值 |
|------|------|--------|
| `SM Throughput (%)` | SM 计算单元利用率 | > 60% |
| `Memory Throughput (%)` | 内存带宽利用率 | > 60% |
| `Achieved Occupancy` | 实际活跃 Warp 比例 | > 50% |
| `Warp Stall Reasons` | Warp 停顿原因分布 | 无主导单一原因 |
| `L1 Hit Rate` | L1 缓存命中率 | 尽可能高 |
| `L2 Hit Rate` | L2 缓存命中率 | 尽可能高 |
| `Register Spills` | 寄存器溢出到 Local Memory | = 0（理想） |
| `Bank Conflicts` | Shared Memory Bank 冲突次数 | = 0（理想） |

---

## 3. 数据传输优化

### 3.1 核心原则

**高优先级规则**：
1. **最小化 Host↔Device 传输量**——即使 GPU 没有加速效果，也应避免不必要的传输。
2. **中间数据留在 Device**——多个 Kernel 间的中间结果不要拷回 Host。
3. **批量传输优于多次小传输**——每次传输有固定开销。

### 3.2 Pinned Memory（页锁定内存）

```cuda
// 普通内存分配 → 内部会额外拷贝到临时 pinned buffer
float *h_data = (float*)malloc(size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);  // 慢

// Pinned 内存 → 直接 DMA 传输，带宽提高 2–3×
float *h_pinned;
cudaMallocHost(&h_pinned, size);  // 或 cudaHostAlloc
cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);  // 快
cudaFreeHost(h_pinned);
```

**注意**：Pinned Memory 是稀缺资源，过度使用会降低系统整体性能。

### 3.3 异步传输与 Compute 重叠

```cuda
// 双 Stream 流水线：传输和计算重叠
cudaStream_t stream[2];
for (int i = 0; i < 2; i++) cudaStreamCreate(&stream[i]);

float *h_pinned;
cudaMallocHost(&h_pinned, 2 * chunkSize);

for (int i = 0; i < numChunks; i++) {
    int s = i % 2;
    // Stream s: 异步传输第 i 块
    cudaMemcpyAsync(d_buf[s], h_pinned + i * chunkN, chunkSize,
                    cudaMemcpyHostToDevice, stream[s]);
    // Stream s: 处理第 i 块
    processKernel<<<grid, block, 0, stream[s]>>>(d_buf[s], chunkN);
    // Stream s: 结果拷回
    cudaMemcpyAsync(h_pinned + i * chunkN, d_buf[s], chunkSize,
                    cudaMemcpyDeviceToHost, stream[s]);
}
```

重叠要求：
- 使用**非默认 Stream**
- Host 端使用 **Pinned Memory**
- 使用 **`cudaMemcpyAsync`**
- GPU 支持 `asyncEngineCount > 0`

### 3.4 Write-Combining Memory

```cuda
float *h_wc;
cudaHostAlloc(&h_wc, size,
              cudaHostAllocWriteCombined | cudaHostAllocMapped);
// Host 写入 → Device 读取：传输带宽提高最多 40%
// 注意：Host 端读取 WC 内存极慢，仅适用于 Host 只写的场景
```

### 3.5 Zero-Copy / Mapped Memory

```cuda
float *h_mapped, *d_mapped;
cudaHostAlloc(&h_mapped, size, cudaHostAllocMapped);
cudaHostGetDevicePointer(&d_mapped, h_mapped, 0);
// Kernel 可直接通过 d_mapped 访问 Host 内存，无需显式拷贝
// 适用场景：数据只读一次、集成 GPU、Jetson 平台
```

**性能特点**：
- 每次访问都走 PCIe/NVLink，延迟高。
- 适合**稀疏访问**或**单次读写**的场景。
- **不适合**反复读取的计算密集型 Kernel。

### 3.6 PCIe vs NVLink 带宽对比

| 互联类型 | 单向带宽 | 延迟 | 典型场景 |
|----------|----------|------|----------|
| PCIe 4.0 x16 | ~25 GB/s | ~10–15 μs | CPU↔GPU |
| PCIe 5.0 x16 | ~50 GB/s | ~8–12 μs | CPU↔GPU (新平台) |
| NVLink 3.0 (A100) | ~300 GB/s (双向 600) | ~1.3 μs | GPU↔GPU |
| NVLink 4.0 (H100) | ~450 GB/s (双向 900) | < 1 μs | GPU↔GPU |

---

## 4. 内存优化

> **内存优化是最重要的性能优化领域。** 目标：最大化快速存储使用，最小化慢速存储访问。

### 4.1 Global Memory 合并访问

**高优先级：确保全局内存访问始终合并。**

#### 单指令访问的前提

Global Memory 指令支持读写 **1、2、4、8 或 16 字节** 的 word。对数据的一次访问**当且仅当**同时满足以下条件才编译为单条全局内存指令：

1. 数据类型大小 ∈ {1, 2, 4, 8, 16} 字节。
2. 数据**自然对齐**（地址是数据类型大小的整数倍）。

否则编译器拆分为多条指令，吞吐量大幅下降。使用 `__align__(N)` 确保自定义结构体对齐。

#### 事务合并条件

同一 Warp 内的 32 个线程访问的地址落在尽可能少的 **32、64 或 128 字节** 自然对齐内存段内（段的首地址必须是其大小的整数倍）。

```cuda
// GOOD: stride-1 合并访问 — 32 线程 × 4B = 128B → 1 个 128B 事务
float val = data[blockIdx.x * blockDim.x + threadIdx.x];

// BAD: stride-2 — 地址跨越更多内存段，需要更多事务
float val = data[(blockIdx.x * blockDim.x + threadIdx.x) * 2];

// WORST: stride-32 — 每个线程触发独立事务
float val = data[threadIdx.x * 32];
```

#### 矩阵转置的合并技巧

```cuda
// 朴素转置：读合并但写不合并（或反过来）
// 解决：用 Shared Memory 做中转

__global__ void transpose(float *out, const float *in, int W, int H) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 padding 消除 bank conflict

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 合并读取 → 行主序
    if (x < W && y < H)
        tile[threadIdx.y][threadIdx.x] = in[y * W + x];
    __syncthreads();

    // 交换 x/y 后合并写入
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    if (x < H && y < W)
        out[y * H + x] = tile[threadIdx.x][threadIdx.y];
}
```

### 4.2 数据布局：SoA vs AoS

```
AoS (Array of Structures):          SoA (Structure of Arrays):
┌──────────────────────┐            ┌──────────────────────┐
│ x0 y0 z0 │ x1 y1 z1 │ ...        │ x0 x1 x2 x3 ... xN  │
└──────────────────────┘            │ y0 y1 y2 y3 ... yN  │
                                    │ z0 z1 z2 z3 ... zN  │
                                    └──────────────────────┘
```

| 布局 | GPU 性能 | 原因 |
|------|----------|------|
| **SoA** | **快 2–4×** | 同一字段连续存储 → 合并访问 |
| AoS | 慢 | 访问单一字段时 stride = sizeof(struct) |
| AoSoA (混合) | 最快 | tile 大小 = warp size，兼顾缓存局部性和合并 |

**LLM 场景**：权重矩阵天然是 SoA（连续 float/half 数组），通常无需转换。注意 **多头注意力的 Q/K/V 拼接** 后的内存布局。

### 4.3 向量化内存访问

```cuda
// 使用 float4 加载 16 字节 → 减少内存事务数量 4×
__global__ void vectorized_copy(float4 *dst, const float4 *src, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        dst[tid] = src[tid];
    }
}

// 对齐要求：指针必须对齐到 sizeof(vector_type)
// 确保分配时 cudaMalloc 返回的指针是 256 字节对齐的（默认保证）
```

向量类型对照表：

| 标量类型 | 2-wide | 4-wide | 每次访问字节 |
|----------|--------|--------|-------------|
| `float` | `float2` | `float4` | 8 / 16 |
| `half` | `half2` | — | 4 |
| `int` | `int2` | `int4` | 8 / 16 |
| `char` | `char2` | `char4` | 2 / 4 |

### 4.4 Shared Memory 最佳实践

#### 作为软件管理缓存

```
Global Memory → Shared Memory → 计算 → Shared Memory → Global Memory
      (合并读)      (多次复用)                          (合并写)
```

#### Bank Conflict 消除

Shared Memory 由 **32 个 Bank** 组成，每 Bank 宽 4 字节（CC 3.x+ 可配为 8 字节）。

| 场景 | Bank Conflict | 解决方案 |
|------|---------------|----------|
| stride-1 访问 `smem[tid]` | 无 | — |
| stride-2 访问 `smem[tid*2]` | 2-way | padding 或 swizzle |
| 列访问 `smem[tid][col]` (N=32) | 32-way (最严重) | `smem[32][33]` padding |
| 广播（所有线程读同一地址） | 无（硬件广播） | — |

```cuda
// 矩阵 tile 的 padding 消除 bank conflict
__shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // 关键：+1

// Swizzle 方法（更高级，不浪费空间）
// 用 XOR 重映射列索引：col' = col ^ (row % 32)
int swizzled_col = col ^ (row & 31);
float val = smem[row][swizzled_col];
```

#### 动态 Shared Memory 大于 48 KB

```cuda
// 1. Kernel 中声明
extern __shared__ char smem[];

// 2. 在 Host 端设置上限
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

// 3. Launch 时指定大小
myKernel<<<grid, block, 98304, stream>>>(args);
```

### 4.5 L1 / L2 Cache 优化

#### L1 Cache 与 Shared Memory 配比

```cuda
// 设置 Shared Memory 优先（适合手动管理缓存的 Kernel）
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared);

// 设置 L1 优先（适合无 Shared Memory 使用、随机访问模式的 Kernel）
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxL1);
```

#### L1 缓存 vs 非缓存加载

```
编译选项              L1 行为              事务粒度
-Xptxas -dlcm=ca    缓存到 L1 + L2       128 字节（默认）
-Xptxas -dlcm=cg    仅缓存到 L2（跳过L1） 32 字节
```

**当访问模式 stride 很大时**，跳过 L1 可减少无效缓存行加载：
- stride-128B 合并 → L1 事务 32×128B = 4096B（浪费 75%）
- stride-128B 跳过L1 → L2 事务 32×32B = 1024B（无浪费）

#### L2 Cache Persistence（CC 8.0+）

```cuda
// 设置 L2 持久化缓存窗口（保持热数据在 L2）
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
size_t persistSize = min(size, (size_t)(prop.persistingL2CacheMaxSize * 0.75));

cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persistSize);

cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = d_hotData;
attr.accessPolicyWindow.num_bytes = persistSize;
attr.accessPolicyWindow.hitRatio = 1.0f;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

### 4.6 Texture Memory / `__ldg` 只读缓存

```cuda
// 方法1: 使用 __ldg 内置函数（CC 3.5+）
float val = __ldg(&input[tid]);  // 走只读数据缓存，不污染 L1

// 方法2: 使用 const __restrict__ 让编译器自动走只读缓存
__global__ void kernel(const float* __restrict__ input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        output[tid] = input[tid] * 2.0f;  // 编译器自动使用 ldg
}
```

### 4.7 寄存器管理

| 策略 | 方法 | 权衡 |
|------|------|------|
| 限制寄存器数 | `--maxrregcount=N` | 更高 Occupancy，可能 spill |
| `__launch_bounds__` | `__launch_bounds__(256, 4)` | 编译器优化提示 |
| 减少变量 | 手动合并临时变量 | 代码可读性下降 |
| 循环展开 | `#pragma unroll` | 更多寄存器需求 |

```cuda
// 查看寄存器使用和 spill 情况
// nvcc -Xptxas -v myKernel.cu
// 输出: ptxas info: Used 42 registers, 0 bytes lmem, ...
// lmem > 0 表示有 register spill → 性能可能受影响
```

---

## 5. 执行配置优化

### 5.1 Block 大小选择

**核心规则**：线程数必须是 **32 的倍数**（Warp 大小）。

| Block 大小 | 适用场景 | 说明 |
|-----------|----------|------|
| **128** | 通用、寄存器密集型 | 平衡 Occupancy 与寄存器/Shared Memory |
| **256** | 大多数场景的推荐值 | 良好的 Occupancy + 足够线程级并行 |
| **512** | 需要大 Block 共享数据 | Block 级归约更高效 |
| **1024** | 特殊场景 | 最大 Block 大小，Occupancy 可能受限 |

**实验验证**：不同 Kernel 的最优 Block 大小不同，需要 benchmark。

### 5.2 Occupancy 优化

```cuda
// 运行时查询最佳配置
int minGridSize, bestBlockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize,
                                    myKernel, 0, 0);
// 0 = 动态 Shared Memory 大小
// 0 = Block 大小上限（0 表示无限制）

myKernel<<<minGridSize, bestBlockSize>>>(args);
```

```cuda
// 查询给定 Block 大小的 Occupancy
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
    myKernel, blockSize, dynamicSmemSize);
float occupancy = (float)(numBlocks * blockSize) /
                  prop.maxThreadsPerMultiProcessor;
```

**Occupancy 经验法则**：

| Occupancy | 状态 | 行动 |
|-----------|------|------|
| < 25% | 过低 | 严重延迟隐藏不足，需调整 |
| 25–50% | 可接受 | 适合寄存器/Shared 密集型 Kernel |
| 50–75% | 良好 | 大多数 Kernel 的最优区间 |
| > 75% | 很高 | 不一定更快；验证是否因寄存器不足反而变慢 |

### 5.3 Grid 大小选择

```cuda
// 确保 Grid 有足够 Block 填满所有 SM
// 最少 = SM 数量 × 每 SM 活跃 Block 数
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

// 最少 Block 数 = numSMs × blocksPerSM
// 推荐 Block 数 = 上述值的数倍（GPU 需要足够的波次来隐藏延迟）

// Grid-stride loop 模式 → Grid 大小与数据大小解耦
int gridSize = (N + blockSize - 1) / blockSize;
gridSize = min(gridSize, numSMs * maxBlocksPerSM);
```

### 5.4 Wave Quantization

Block 以"波（wave）"为单位在 SM 上调度执行。当 Block 总数不是 wave 大小的整数倍时，最后一波的 SM 利用率下降（**tail effect**）。

```
Wave size = numSMs × maxActiveBlocksPerSM

示例: 108 SM × 4 blocks/SM = 432 blocks/wave
- 432 blocks → 1 波，100% 利用
- 433 blocks → 2 波，第二波仅 1 block（0.2% 利用率）
- 864 blocks → 2 波，100% 利用

应避免的情况: Block 总数刚好超过 wave 边界一点点。
```

---

## 6. 指令优化

### 6.1 算术指令吞吐

| 操作 | 吞吐 (ops/SM/cycle) CC 8.0 | 说明 |
|------|----------------------------|------|
| FP32 add/mul/fma | 64 | 全速 |
| FP64 add/mul/fma | 32 | A100 有完整 FP64 |
| FP16 fma (CUDA Core) | 128 | 2× FP32 |
| INT32 add | 64 | 全速 |
| Integer division | ~20 | **很慢**，用位运算替代 |
| `__sinf, __cosf, __expf` | 16 (SFU) | SFU 吞吐有限 |

### 6.2 整数除法与取模优化

```cuda
// SLOW: 整数除法约 20 个时钟周期
int q = n / divisor;
int r = n % divisor;

// FAST: 2 的幂除法用位运算替代
int q = n >> log2_divisor;       // 除以 2^k
int r = n & (divisor - 1);       // 模 2^k

// FAST: 编译器常量除法优化
// 如果 divisor 是编译时常量，编译器会自动用乘法+移位替代
```

### 6.3 循环展开

```cuda
// 编译器自动展开（提示编译器完全展开）
#pragma unroll
for (int i = 0; i < 8; i++) {
    result += data[i] * weights[i];
}

// 指定展开因子
#pragma unroll 4
for (int i = 0; i < N; i++) {
    // 展开 4 次
}

// 禁止展开（减少寄存器压力时使用）
#pragma unroll 1
for (int i = 0; i < N; i++) { ... }
```

### 6.4 FMA（Fused Multiply-Add）

```cuda
// 编译器默认启用 FMA (--fmad=true)
// a * b + c → 单条 FMA 指令，比分开的 MUL + ADD 更快且更精确
float result = a * b + c;  // 编译为 FMA

// 如需禁用（精度对比时）
// nvcc --fmad=false
```

### 6.5 Reciprocal Square Root

```cuda
// SLOW
float val = 1.0f / sqrtf(x);

// FAST: 使用 rsqrtf 内置函数
float val = rsqrtf(x);  // 硬件单指令实现

// EVEN FASTER (低精度)
float val = __frsqrt_rn(x);  // intrinsic 版本
```

### 6.6 `__ldg` 只读加载

```cuda
// 通过只读缓存路径加载，不与 L1/Shared 竞争
float val = __ldg(&input[tid]);
```

---

## 7. 控制流优化

### 7.1 Warp Divergence

```cuda
// BAD: 50% 的线程闲置于每个分支
if (threadIdx.x % 2 == 0) {
    // path A
} else {
    // path B
}

// GOOD: 整个 Warp 走同一分支
if ((threadIdx.x / warpSize) % 2 == 0) {
    // path A — 偶数 Warp
} else {
    // path B — 奇数 Warp
}
```

### 7.2 Branch Predication

当分支体很短（< 7 条指令）时，编译器可能使用 **predication** 替代真正的分支跳转：所有线程都执行两条路径的指令，但只有符合条件的线程写入结果。这消除了 Warp 序列化开销。

```cuda
// 编译器可能自动 predicate 这种短分支
float result = (x > 0.0f) ? x : 0.0f;
// → 不产生 divergence，所有线程都计算 x 和 0，条件选择结果
```

### 7.3 减少条件判断

```cuda
// 边界检查优化：主体不检查边界，仅最后一个 Block 检查
if (blockIdx.x < gridDim.x - 1) {
    // 无边界检查的主体（大多数 Block）
    float val = input[tid];
    output[tid] = process(val);
} else {
    // 仅最后一个 Block 做边界检查
    if (tid < N) {
        float val = input[tid];
        output[tid] = process(val);
    }
}
```

### 7.4 Early Exit 的正确使用

```cuda
// 允许整个 Warp 提前退出（无 divergence）
if (blockIdx.x * blockDim.x >= N) return;  // Block 级 early exit

// 但 Thread 级 early exit 可能导致 __syncthreads 死锁
// BAD:
if (tid >= N) return;  // ← 部分线程退出
__syncthreads();        // ← 死锁：剩余线程永远等不到退出的线程

// SAFE:
if (tid < N) {
    // do work
}
__syncthreads();  // 所有线程都到达
```

---

## 8. 浮点精度与数学运算

### 8.1 编译器浮点选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--ftz={true\|false}` | `false` | Flush subnormal to zero（性能提升，精度微降） |
| `--prec-div={true\|false}` | `true` | IEEE 精确除法（false → 快速近似） |
| `--prec-sqrt={true\|false}` | `true` | IEEE 精确开方（false → 快速近似） |
| `--fmad={true\|false}` | `true` | 启用 FMA 指令融合 |
| `--use_fast_math` | off | **等同于** `--ftz=true --prec-div=false --prec-sqrt=false --fmad=true` |

```bash
# 最大性能（牺牲精度）
nvcc --use_fast_math kernel.cu

# 保守设置（最大精度可移植性）
nvcc --ftz=false --prec-div=true --prec-sqrt=true kernel.cu
```

### 8.2 标准函数 vs 内置 Intrinsic

| 标准函数 | Intrinsic（快速近似） | 精度 | 速度提升 |
|----------|----------------------|------|----------|
| `expf(x)` | `__expf(x)` | ~2 ULP | ~3–5× |
| `logf(x)` | `__logf(x)` | ~1 ULP | ~3–5× |
| `sinf(x)` | `__sinf(x)` | ~2 ULP | ~5–8× |
| `cosf(x)` | `__cosf(x)` | ~2 ULP | ~5–8× |
| `x / y` | `__fdividef(x, y)` | ~2 ULP | ~2–3× |
| `powf(x, y)` | `__powf(x, y)` | 较大 | ~3–5× |
| `sqrtf(x)` | `__fsqrt_rn(x)` | 0 ULP | 同速但指定舍入 |
| `1/sqrtf(x)` | `rsqrtf(x)` | ~1 ULP | 单指令 |

```cuda
// 同时计算 sin 和 cos（避免重复计算）
float s, c;
sincosf(x, &s, &c);         // 标准版
__sincosf(x, &s, &c);       // intrinsic 版
```

### 8.3 `--use_fast_math` 的使用建议

**适合使用的场景**：
- LLM inference：Softmax、GELU 等激活函数对精度不敏感。
- 对精度要求在 1% 以内的应用。
- 性能关键路径，且已验证数值误差可接受。

**不适合使用的场景**：
- 数值敏感的科学计算。
- 训练过程中的梯度计算（累积误差可能导致不收敛）。
- 金融计算等需要 IEEE 精确结果的场景。

**折中方案**：对个别热点函数手动使用 intrinsic，其余保持精确。

```cuda
// 整体使用精确数学，仅 softmax 中的 exp 使用快速近似
float score = __expf(val - max_val);  // 快速 exp
float norm = score / sum;              // 精确除法
```

### 8.4 FP16 / BF16 / FP8 精度考量

| 类型 | 位宽 | 指数位 | 尾数位 | 精度范围 | 适用场景 |
|------|------|--------|--------|----------|----------|
| FP32 | 32 | 8 | 23 | ~7 位十进制 | 训练累积、精确计算 |
| FP16 | 16 | 5 | 10 | ~3.3 位十进制 | 推理、混合精度训练 |
| BF16 | 16 | 8 | 7 | ~2.4 位十进制 | 训练（更大动态范围） |
| TF32 | 19 | 8 | 10 | ~3.3 位十进制 | Tensor Core 内部格式 |
| FP8 E4M3 | 8 | 4 | 3 | ~1 位十进制 | Hopper+ 推理 |
| FP8 E5M2 | 8 | 5 | 2 | ~0.6 位十进制 | 梯度（更大范围） |

**Mixed Precision 最佳实践**：
1. 权重和激活用 FP16/BF16 存储和计算。
2. 累加器用 FP32（避免 FP16 溢出/精度损失）。
3. Loss scaling（训练时）防止 FP16 梯度下溢。

```cuda
// Tensor Core 自动 mixed precision 模式
// A/B 用 FP16/BF16，C/D 用 FP32 累加
fragment<accumulator, 16, 16, 16, float> c_frag;  // FP32 累加
```

### 8.5 浮点非结合性

```
// 数学上: (A + B) + C == A + (B + C)
// 浮点上: 不保证相等！
// 并行归约会改变运算顺序 → 结果可能与串行不同
// 这是正常的，不是 bug

// 验证策略：比较并行 vs 串行结果，容忍合理的相对误差
// 典型阈值: FP32 相对误差 < 1e-5, FP16 < 1e-2
```

---

## 9. Profile 驱动优化

### 9.1 工具链

| 工具 | 用途 | 粒度 |
|------|------|------|
| **Nsight Systems** | 系统级时间线分析 | Kernel launch、数据传输、CPU/GPU 重叠 |
| **Nsight Compute** | Kernel 级深度分析 | 指令吞吐、内存效率、Occupancy、Roofline |
| **cuda-memcheck / compute-sanitizer** | 正确性检查 | 越界访问、Race condition |
| **nvprof**（已废弃） | 旧版 profiler | 兼容旧代码 |

### 9.2 Nsight Systems 使用流程

```bash
# 收集系统级 trace
nsys profile --stats=true ./my_cuda_app

# 输出 .nsys-rep 文件，在 Nsight Systems GUI 中查看时间线
# 关注：
# - Kernel 之间是否有空隙（launch 开销 / 同步等待）
# - 数据传输是否与 Kernel 执行重叠
# - CPU 是否成为瓶颈
```

### 9.3 Nsight Compute 使用流程

```bash
# 对指定 Kernel 做深度 profiling
ncu --set full -o profile_report ./my_cuda_app

# 常用分析选项
ncu --section SpeedOfLight         # 计算/内存利用率摘要
ncu --section MemoryWorkloadAnalysis  # 内存层次详细分析
ncu --section Occupancy             # Occupancy 分析
ncu --section WarpStateStatistics   # Warp 停顿原因
ncu --section SourceCounters        # 源码级热点

# Roofline 分析
ncu --section SpeedOfLight_RooflineChart ./my_cuda_app
```

### 9.4 Profile 驱动优化工作流

```
1. Nsight Systems 全局概览
   └─ 确定瓶颈是 Kernel 执行 / 数据传输 / CPU 端

2. Nsight Compute 深入热点 Kernel
   ├─ SpeedOfLight → 确定 Compute-bound 还是 Memory-bound
   ├─ MemoryWorkloadAnalysis → 检查合并率、L1/L2 命中率
   ├─ Occupancy → 检查资源限制因素
   ├─ WarpStateStatistics → 检查停顿原因
   └─ SourceCounters → 定位源码级热点

3. 根据分析结果选择优化策略
   ├─ Memory-bound → 合并访问、向量化、Shared Memory 缓存
   ├─ Compute-bound → Tensor Core、ILP、fast math
   ├─ Latency-bound → 提高 Occupancy、增加并行度
   └─ Launch-bound → CUDA Graph、Kernel 融合

4. 优化后重新 Profile，验证效果
```

---

## 10. 多 GPU 编程

### 10.1 基础 Multi-GPU 模式

```cuda
int numDevices;
cudaGetDeviceCount(&numDevices);

// 每个 GPU 分配并处理数据的一部分
for (int dev = 0; dev < numDevices; dev++) {
    cudaSetDevice(dev);
    cudaMalloc(&d_data[dev], chunkSize);
    cudaMemcpyAsync(d_data[dev], h_data + dev * chunkN,
                    chunkSize, cudaMemcpyHostToDevice, stream[dev]);
    processKernel<<<grid, block, 0, stream[dev]>>>(d_data[dev], chunkN);
}
```

### 10.2 Peer-to-Peer (P2P) 内存访问

```cuda
// 启用 GPU 0 和 GPU 1 之间的 P2P 访问
int canAccess;
cudaDeviceCanAccessPeer(&canAccess, 0, 1);
if (canAccess) {
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);

    // GPU 0 的 Kernel 可以直接读写 GPU 1 的内存
    // 或使用 cudaMemcpyPeer 进行显式拷贝
    cudaMemcpyPeerAsync(d_dst, 0, d_src, 1, size, stream);
}
```

### 10.3 通信库

| 库 | 用途 | 最佳场景 |
|-----|------|----------|
| **NCCL** | GPU 集合通信 (AllReduce, AllGather...) | 数据并行训练 |
| **NVSHMEM** | GPU 发起的 Put/Get 通信 | 细粒度通信 |
| **MPI + CUDA-aware** | 跨节点通信 | 传统 HPC 模式 |

### 10.4 LLM 多 GPU 策略

| 策略 | 描述 | 通信模式 |
|------|------|----------|
| 数据并行 (DP) | 每 GPU 完整模型副本，数据分片 | AllReduce 梯度 |
| 张量并行 (TP) | 矩阵按列/行切分到多 GPU | AllReduce / AllGather 每层 |
| 流水线并行 (PP) | 不同层分配到不同 GPU | 点对点发送激活 |
| 序列并行 (SP) | 序列维度分片 | AllGather / ReduceScatter |

---

## 11. 部署与兼容性

### 11.1 编译架构选择

```bash
# 为多种架构编译（fat binary）
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 \  # PTX 前向兼容
     -o my_app kernel.cu

# 最小化二进制大小：仅目标架构
nvcc -arch=sm_80 -o my_app kernel.cu
```

**PTX 前向兼容**：包含 PTX 代码（`code=compute_XX`）可让驱动在新架构上 JIT 编译，但首次启动会较慢。

### 11.2 CUDA 版本兼容性

| CUDA Toolkit | 最低驱动版本 | 新增支持架构 |
|-------------|-------------|-------------|
| CUDA 11.0 | 450.x | Ampere (CC 8.0) |
| CUDA 11.1+ | 455.x+ | CC 8.6 |
| CUDA 11.8 | 520.x | CC 8.9 (Ada) |
| CUDA 12.0 | 525.x | CC 9.0 (Hopper) |
| CUDA 12.8+ | 570.x+ | CC 10.0 (Blackwell) |

### 11.3 错误处理清单

```cuda
// 1. 每个 Runtime API 调用都检查返回值
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

// 2. Kernel launch 后检查异步错误
myKernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());        // 检查 launch 配置错误
CUDA_CHECK(cudaDeviceSynchronize());   // 检查执行错误（仅调试时使用）

// 3. 生产环境中使用 compute-sanitizer 检测内存错误
// compute-sanitizer --tool memcheck ./my_app
```

### 11.4 性能移植性注意事项

- 不同架构最优参数不同（Block 大小、tile 大小、Shared Memory 用量）。
- 使用 **auto-tuning** 或 **编译时模板参数** 适配不同架构。
- Hopper 特有功能（TMA、WGMMA、Cluster）需要 `sm_90a`，不能回退到旧架构。
- 注意 CC 8.6/8.9 的每 SM Block 限制（16/24）与 CC 8.0/9.0（32）不同。

---

## 12. 面向 LLM Kernel 的 Best Practices 检查清单

### 12.1 Memory-bound Kernel 优化清单（LayerNorm, Softmax, Activation）

- [ ] 使用 `float4` / `half2` 向量化加载和存储
- [ ] 确保 stride-1 合并访问
- [ ] 使用 Warp Shuffle 做归约（避免 Shared Memory 开销）
- [ ] 融合相邻 element-wise 操作（Bias + Activation + Residual Add）
- [ ] Kernel 参数使用 `const __restrict__`
- [ ] 考虑 `__ldg` 或只读缓存
- [ ] 对 FP16 输入使用 `half2` 打包操作（`__hadd2`, `__hmul2`）
- [ ] 考虑 `--use_fast_math` 或选择性使用 `__expf`

### 12.2 Compute-bound Kernel 优化清单（GEMM, Attention）

- [ ] 使用 Tensor Core（WMMA / WGMMA / CUTLASS / CuTe）
- [ ] Tiling: Block tile (Shared) + Thread tile (Register)
- [ ] Double buffering / Software pipelining 隐藏 Shared Memory 加载延迟
- [ ] 寄存器中的外积累加 (Outer Product Accumulate)
- [ ] 检查寄存器压力：`nvcc -Xptxas -v`，无 spill
- [ ] Shared Memory bank conflict 消除（padding / swizzle）
- [ ] `__launch_bounds__` 提示编译器
- [ ] 混合精度：A/B 用 FP16/BF16，累加器用 FP32
- [ ] 对 Hopper: 考虑 TMA + WGMMA + Warp Specialization

### 12.3 端到端 Inference Pipeline 优化清单

- [ ] CUDA Graph 减少 Kernel launch 开销
- [ ] Stream 并发：数据加载与计算重叠
- [ ] 中间张量全部留在 GPU（避免 D2H 拷回）
- [ ] KV-Cache 内存管理（PagedAttention / 预分配池）
- [ ] 权重预加载 + Pinned Memory
- [ ] Multi-GPU: 张量并行用 NCCL AllReduce
- [ ] 量化推理：INT8/FP8 GEMM + 融合反量化

### 12.4 调试与验证清单

- [ ] `compute-sanitizer --tool memcheck` 检查越界访问
- [ ] `compute-sanitizer --tool racecheck` 检查 Shared Memory 竞争
- [ ] 与 CPU 参考实现对比精度（FP32 相对误差 < 1e-5）
- [ ] 边界条件测试：N 不整除 Block 大小、Tile 大小
- [ ] 极端输入：NaN、Inf、subnormal、零长度
- [ ] 多次运行验证确定性（注意浮点非结合性）

---

## 参考文献

1. [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
2. [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
3. [NVIDIA CUDA Programming Guide (New)](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
4. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
5. [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
6. [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
7. [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)
8. [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
9. [nvcc Compiler Switches — Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/nvcc-compiler-switches.html)
10. [Floating-Point Computation — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
11. [CUDA Math API — Single Precision Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html)
12. [How to Optimize Data Transfers in CUDA C/C++ (NVIDIA Blog)](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
13. [Using Shared Memory in CUDA C/C++ (NVIDIA Blog)](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
14. [Accelerating HPC with Nsight Compute Roofline (NVIDIA Blog)](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/)
15. [Multi-GPU Programming — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)
16. [GPU Gems 3: Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
17. [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
