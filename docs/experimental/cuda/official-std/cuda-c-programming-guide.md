# CUDA C++ Programming Guide — Kernel Agent Skill Reference

> 基于 [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 及
> [CUDA Programming Guide (新版)](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) 编写。
> 面向 LLM 高性能 Kernel 自动生成场景，覆盖 CUDA 编程模型、语言扩展、Runtime API、
> 硬件执行模型、内存层次、性能优化与高级特性。

---

## 目录

1. [编程模型概述](#1-编程模型概述)
2. [CUDA C++ 语言扩展](#2-cuda-c-语言扩展)
3. [线程层次结构](#3-线程层次结构)
4. [内存层次结构](#4-内存层次结构)
5. [CUDA Runtime API](#5-cuda-runtime-api)
6. [异步并发执行与 Stream](#6-异步并发执行与-stream)
7. [硬件执行模型](#7-硬件执行模型)
8. [性能优化指南](#8-性能优化指南)
9. [高级编程特性](#9-高级编程特性)
10. [Compute Capability 规格表](#10-compute-capability-规格表)
11. [面向 LLM 的 Kernel 设计模式](#11-面向-llm-的-kernel-设计模式)

---

## 1. 编程模型概述

### 1.1 异构计算模型

CUDA 采用 **Host + Device** 异构编程模型：

- **Host**：CPU 及其 DRAM（主机内存）。负责串行逻辑、内存管理、Kernel 调度。
- **Device**：GPU 及其 DRAM（设备内存 / HBM）。负责大规模并行计算。

程序运行流程：

```
Host 代码执行 → 分配 Device 内存 → Host→Device 数据拷贝
→ 启动 Kernel（GPU 并行执行）→ Device→Host 结果拷回 → 释放资源
```

### 1.2 Kernel 概念

**Kernel** 是在 GPU 上执行的函数。一次 Kernel Launch 会产生数百万个线程并行执行同一段代码。每个线程通过内置变量（`threadIdx`, `blockIdx` 等）确定自己的唯一标识，从而操作不同数据。

### 1.3 自动可扩展性

线程块（Thread Block）可被调度到任意可用的 SM 上执行，无需显式指定。因此，拥有更多 SM 的 GPU 会自动获得更短的执行时间——这就是 CUDA 的**自动可扩展性（Automatic Scalability）**。

线程块之间**不保证调度顺序**，也**不能依赖跨块执行结果**（除非使用 Cooperative Launch 或显式同步原语）。

---

## 2. CUDA C++ 语言扩展

### 2.1 函数执行空间限定符

| 限定符 | 调用方 | 执行位置 | 说明 |
|--------|--------|----------|------|
| `__global__` | Host（或 Device via CDP） | Device | 定义 Kernel 函数；返回类型必须为 `void`；调用是**异步**的 |
| `__device__` | Device | Device | 只能从 Device 代码调用的辅助函数 |
| `__host__` | Host | Host | 等同于普通 C++ 函数（可省略） |
| `__host__ __device__` | Host 或 Device | 对应调用方 | 为 Host 和 Device 各生成一份代码 |
| `__noinline__` | — | Device | 建议编译器不内联 |
| `__forceinline__` | — | Device | 强制内联 |

### 2.2 变量类型限定符

| 限定符 | 存储位置 | 作用域 | 生命周期 | 典型用途 |
|--------|----------|--------|----------|----------|
| `int var` | 寄存器 | 线程 | 线程 | 线程私有计算变量 |
| `int arr[N]` | Local Memory（寄存器溢出时） | 线程 | 线程 | 线程私有数组 |
| `__shared__` | Shared Memory（片上 SRAM） | 线程块 | 线程块 | 块内线程协作、数据复用 |
| `__device__` | Global Memory（HBM/DRAM） | 全局 | 应用程序 | 全局可读写数据 |
| `__constant__` | Constant Memory（缓存优化） | 全局 | 应用程序 | 只读广播数据（如超参数） |
| `__restrict__` | — | — | — | 指针无别名提示，帮助编译器优化 |

**`__restrict__` 最佳实践**：对 Kernel 参数指针同时使用 `const` 和 `__restrict__` 可让编译器使用只读数据缓存（`__ldg`），实测可提升 2× 以上性能。

### 2.3 执行配置语法 `<<<>>>`

```cuda
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `gridDim` | `dim3` / `int` | Grid 中 Block 的数量（1D/2D/3D） |
| `blockDim` | `dim3` / `int` | 每个 Block 中 Thread 的数量（1D/2D/3D） |
| `sharedMemBytes` | `size_t` | 动态 Shared Memory 大小（字节），默认 0 |
| `stream` | `cudaStream_t` | 所属 Stream，默认 0（Default Stream） |

```cuda
dim3 grid(N / 256);
dim3 block(256);
myKernel<<<grid, block, 0, stream>>>(d_input, d_output, N);
```

### 2.4 内置变量

| 变量 | 类型 | 说明 |
|------|------|------|
| `threadIdx` | `uint3` | 当前线程在 Block 内的索引 (`.x`, `.y`, `.z`) |
| `blockIdx` | `uint3` | 当前 Block 在 Grid 内的索引 |
| `blockDim` | `dim3` | Block 的维度大小 |
| `gridDim` | `dim3` | Grid 的维度大小 |
| `warpSize` | `int` | Warp 大小，当前所有架构均为 32 |

**标准全局线程索引计算**：

```cuda
// 1D
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// Grid-stride loop（处理数据量 > 总线程数）
for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
    output[i] = func(input[i]);
}
```

**Grid 尺寸上限**：`gridDim.x ≤ 2^31 - 1`，`gridDim.y ≤ 65535`，`gridDim.z ≤ 65535`。

---

## 3. 线程层次结构

### 3.1 Thread → Warp → Block → Grid

```
Grid
├── Block (0,0)          Block (1,0)          Block (2,0) ...
│   ├── Warp 0 (threads  0–31)
│   ├── Warp 1 (threads 32–63)
│   └── ...
└── ...
```

- **Thread**：最小执行单元，拥有私有寄存器和 Local Memory。
- **Warp**：32 个连续线程组成的 SIMT 执行单元。同一 Warp 中的线程执行同一指令（分支时可能 diverge）。
- **Thread Block**：可包含 1–1024 个线程。块内线程可通过 Shared Memory 协作，通过 `__syncthreads()` 同步。
- **Grid**：所有 Block 的集合。Block 间相互独立，不保证执行顺序。

### 3.2 Warp 执行模型 (SIMT)

- GPU 以 **Warp** 为粒度调度执行。
- 同一 Warp 内的 32 个线程同时执行同一条指令（SIMT：Single Instruction, Multiple Threads）。
- **Warp Divergence**：当 Warp 内的线程遇到分支时，走不同分支的线程会**串行化**执行，导致吞吐下降。
  - 优化策略：尽量让同一 Warp 内的线程走相同分支。

```cuda
// BAD: 奇偶线程走不同分支 → Warp Divergence
if (threadIdx.x % 2 == 0) { ... } else { ... }

// BETTER: 连续 32 个线程走相同分支
if (threadIdx.x / 32 % 2 == 0) { ... } else { ... }
```

### 3.3 Thread Block Cluster (CC 9.0+, Hopper)

Hopper 架构引入可选的 **Thread Block Cluster** 层次：

- 由多个 Thread Block 组成（最多 8 个）。
- Cluster 内的 Block 保证被共调度到同一 GPC 上。
- Cluster 内的 Block 可访问彼此的 Shared Memory → **Distributed Shared Memory**。

```cuda
// 使用 Cluster 的 Kernel 声明
__cluster_dims__(2, 1, 1)
__global__ void myKernel(...) { ... }
```

---

## 4. 内存层次结构

### 4.1 完整内存层次

```
┌─────────────────────────────────────────────────────────────┐
│                        GPU Device                          │
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                    │
│  │   SM 0  │  │   SM 1  │  │   SM N  │                    │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │                    │
│  │ │Regs │ │  │ │Regs │ │  │ │Regs │ │  ← 寄存器（最快）  │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │                    │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │                    │
│  │ │Shared│ │  │ │Shared│ │  │ │Shared│ │  ← Shared Memory │
│  │ │ Mem  │ │  │ │ Mem  │ │  │ │ Mem  │ │    (片上 SRAM)    │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │                    │
│  │  L1 $   │  │  L1 $   │  │  L1 $   │                    │
│  └─────────┘  └─────────┘  └─────────┘                    │
│              ┌──────────────────┐                           │
│              │    L2 Cache      │   ← 所有 SM 共享         │
│              └──────────────────┘                           │
│  ┌──────────────────────────────────────────────┐          │
│  │           Global Memory (HBM / DRAM)         │          │
│  │  ┌──────────┐  ┌────────────┐  ┌──────────┐ │          │
│  │  │ Constant │  │  Texture   │  │  Global  │ │          │
│  │  │  Memory  │  │  Memory    │  │  Data    │ │          │
│  │  └──────────┘  └────────────┘  └──────────┘ │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 各级内存特性

| 内存类型 | 位置 | 延迟（cycles） | 带宽 | 作用域 | 读写 | 大小 |
|----------|------|----------------|------|--------|------|------|
| 寄存器 | 片上 | 1 | 最高 | 线程 | R/W | 255 个/线程，64K/SM |
| Shared Memory | 片上 SRAM | ~10–50 | 很高 | 线程块 | R/W | 48–228 KB/SM（架构相关） |
| L1 Cache | 片上 | ~30 | 高 | SM | — | 与 Shared 共享物理资源 |
| L2 Cache | 片上 | ~200 | 中高 | 全局 | R/W | 数 MB（A100: 40MB） |
| Global Memory | HBM/DRAM | ~400–800 | 中 | 全局 | R/W | 数 GB–数十 GB |
| Constant Memory | DRAM + Cache | ~1（命中）~400（未命中） | 广播高效 | 全局 | RO | 64 KB |
| Texture Memory | DRAM + Cache | ~400（未命中） | 空间局部性优化 | 全局 | RO | — |
| Local Memory | DRAM | ~400–800 | 中 | 线程 | R/W | 寄存器溢出时使用 |

### 4.3 Shared Memory 详解

#### 静态 vs 动态分配

```cuda
// 静态 Shared Memory — 编译时确定大小
__shared__ float smem[256];

// 动态 Shared Memory — 运行时通过执行配置指定大小
extern __shared__ float smem_dyn[];

// Launch 时指定动态 Shared Memory 大小
kernel<<<grid, block, dynamicSmemSize>>>(args);
```

#### 超过 48 KB 的 Shared Memory

静态声明限制为 48 KB。使用更大 Shared Memory 需要：

```cuda
// 1. 使用动态 Shared Memory
extern __shared__ char smem[];

// 2. 启动前设置属性
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);  // 96 KB

// 3. 在执行配置中指定
myKernel<<<grid, block, 98304, stream>>>(args);
```

#### Bank Conflict

- Shared Memory 被分为 **32 个 Bank**，每个 Bank 宽度为 4 字节。
- 当同一 Warp 中的多个线程访问同一 Bank 的不同地址时，访问会**串行化**。
- **避免策略**：
  - 连续线程访问连续地址（stride-1）。
  - 使用 **padding** 消除冲突：`__shared__ float smem[32][33];`（多加一列）。
  - 使用 **swizzle** 模式重映射地址。

```cuda
// BAD: stride-2 访问，偶数线程冲突
float val = smem[threadIdx.x * 2];

// GOOD: stride-1 访问，无冲突
float val = smem[threadIdx.x];

// GOOD: 用 padding 消除冲突
__shared__ float smem[32][33];  // 32x32 矩阵加 1 列 padding
float val = smem[threadIdx.y][threadIdx.x];
```

### 4.4 Unified Memory

Unified Memory 提供统一地址空间，Host 和 Device 通过同一指针访问：

```cuda
float *data;
cudaMallocManaged(&data, N * sizeof(float));
// Host 和 Device 均可通过 data 指针访问
// 运行时自动在 Host/Device 间迁移页面
```

适合原型开发，但高性能 Kernel 通常使用显式内存管理。

---

## 5. CUDA Runtime API

### 5.1 设备管理

```cuda
int deviceCount;
cudaGetDeviceCount(&deviceCount);      // 获取 GPU 数量
cudaSetDevice(0);                       // 选择 GPU
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);      // 查询设备属性
// prop.sharedMemPerBlock, prop.maxThreadsPerBlock,
// prop.warpSize, prop.multiProcessorCount, ...
```

### 5.2 内存管理

```cuda
float *d_data;
size_t size = N * sizeof(float);

// 分配 Device 内存
cudaMalloc(&d_data, size);

// Host → Device 拷贝
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Device → Host 拷贝
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

// 释放
cudaFree(d_data);

// 异步拷贝（需要 Pinned Memory + 非默认 Stream）
float *h_pinned;
cudaMallocHost(&h_pinned, size);  // 分配页锁定内存
cudaMemcpyAsync(d_data, h_pinned, size,
                cudaMemcpyHostToDevice, stream);
```

### 5.3 错误处理

```cuda
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

CUDA_CHECK(cudaMalloc(&d_data, size));
```

### 5.4 Event 与计时

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
myKernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

---

## 6. 异步并发执行与 Stream

### 6.1 Stream 概念

- Stream 是 GPU 上的**有序任务队列**。
- 同一 Stream 内的操作**按顺序执行**（FIFO）。
- 不同 Stream 间的操作**可以并发**。
- 所有未指定 Stream 的操作在 **Default Stream** 中执行。

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 不同 Stream 中的操作可以重叠
cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2);
kernelA<<<grid, block, 0, stream1>>>(d_A);
kernelB<<<grid, block, 0, stream2>>>(d_B);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

### 6.2 并发要求

实现 Kernel 执行与数据传输重叠需要：
1. 使用**非默认 Stream**。
2. Host 端使用 **页锁定内存**（`cudaMallocHost` / `cudaHostAlloc`）。
3. 使用 **`cudaMemcpyAsync`**。

### 6.3 CUDA Graph

对于需要反复执行的 Kernel 序列，CUDA Graph 可减少启动开销：

```cuda
cudaGraph_t graph;
cudaGraphExec_t graphExec;

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
// ... 一系列 kernel launch 和 memcpy ...
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// 多次重放
for (int i = 0; i < iterations; i++) {
    cudaGraphLaunch(graphExec, stream);
}
```

---

## 7. 硬件执行模型

### 7.1 GPU 硬件层次

```
GPU
├── GPC (Graphics Processing Cluster) × N
│   ├── SM (Streaming Multiprocessor) × M
│   │   ├── Warp Scheduler × 4（典型）
│   │   ├── 寄存器文件 (64K × 32-bit)
│   │   ├── Shared Memory / L1 Cache（统一，可配置比例）
│   │   ├── FP32 / INT32 / FP64 计算单元
│   │   ├── Tensor Core
│   │   └── SFU（特殊函数单元：sin, cos, exp, ...）
│   └── ...
├── L2 Cache（全局共享）
└── HBM / DRAM（Global Memory）
```

### 7.2 Block 到 SM 的映射

- 每个 Block **只能运行在一个 SM** 上（不可迁移）。
- 一个 SM 可同时运行**多个 Block**（受寄存器/Shared Memory/线程数限制）。
- Block 的资源需求决定了 SM 上可驻留的 Block 数量 → **Occupancy**。

### 7.3 Warp 调度

- 每个 SM 有多个 **Warp Scheduler**（通常 4 个）。
- 每个时钟周期，Scheduler 从就绪的 Warp 中选择一个发射指令。
- **延迟隐藏**：当一个 Warp 因访存等待时，Scheduler 切换到另一个就绪 Warp → 关键性能机制。

---

## 8. 性能优化指南

### 8.1 Occupancy（占用率）

**Occupancy = 活跃 Warp 数 / SM 最大 Warp 数**

影响因素：
- 每线程寄存器使用量（寄存器压力）
- 每 Block 的 Shared Memory 使用量
- 每 Block 的线程数

| 方法 | 说明 |
|------|------|
| 限制寄存器 | `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` |
| 编译选项 | `--maxrregcount=N` |
| 调整 Block 大小 | 通常 128 或 256，必须为 32 的倍数 |
| 减少 Shared Memory | 减少每 Block 使用量以容纳更多 Block |

**注意**：高 Occupancy 不一定等于高性能。60–80% 通常是甜蜜点。对于 compute-bound Kernel，适当降低 Occupancy 换取更多寄存器/Shared Memory 可能更优。

```cuda
// 使用 launch_bounds 提示编译器
__global__ void __launch_bounds__(256, 4) myKernel(...) {
    // 限制每 Block 最多 256 线程，每 SM 至少 4 个 Block
}
```

### 8.2 Memory Coalescing（内存合并访问）

#### Size 与对齐要求

Global Memory 指令支持读写大小为 **1、2、4、8 或 16 字节**的 word。对数据的一次访问（通过变量或指针）**当且仅当**满足以下两个条件时，才会编译为**单条**全局内存指令：

1. 数据类型大小为 1、2、4、8 或 16 字节。
2. 数据是**自然对齐**的（地址是数据类型大小的整数倍）。

不满足条件时，编译器会将访问拆分为多条指令，**吞吐量显著下降**。

```cuda
// 自然对齐的内置类型 — 编译为单条指令
float  val4  = data[tid];           // 4 字节，对齐 4
float4 val16 = vec_data[tid];       // 16 字节，对齐 16

// 结构体需显式对齐
struct __align__(16) AlignedStruct {
    float x, y, z, w;               // 16 字节，对齐 16 → 单条指令
};

struct MisalignedStruct {
    float x, y, z;                  // 12 字节 → 不在 {1,2,4,8,16} 中
};                                  // 编译为多条指令，效率低
```

#### 事务合并

Global Memory 以 **32、64 或 128 字节**的自然对齐事务为单位访问（事务的首地址必须是其大小的整数倍）。当同一 Warp 的 32 个线程访问连续内存地址时，硬件可将多个访问合并为尽可能少的事务。

```cuda
// GOOD: 合并访问 — 连续线程访问连续地址
// 32 个线程 × 4B = 128B → 1 个 128B 事务
float val = data[blockIdx.x * blockDim.x + threadIdx.x];

// BAD: 非合并访问 — stride 访问
// 地址分散 → 需要更多事务
float val = data[threadIdx.x * stride];  // stride > 1 时效率下降

// BAD: 随机访问
// 最坏情况：32 个线程触发 32 个独立事务
float val = data[random_index[threadIdx.x]];
```

**合并 vs 非合并访问可以产生一个数量级的性能差异。**

### 8.3 Vectorized Memory Access（向量化访问）

使用向量类型一次加载 4/8/16 字节，减少内存事务数：

```cuda
// 标量访问 — 每次 4 字节
float val = data[tid];

// 向量化访问 — 每次 16 字节 (4×float)
float4 val4 = reinterpret_cast<float4*>(data)[tid];

// 对 half 类型使用 half2 向量化
half2 val2 = reinterpret_cast<half2*>(data)[tid];
```

### 8.4 Shared Memory 优化

- 作为 **软件管理的缓存**，减少全局内存访问。
- **Tiling 模式**：将数据分块加载到 Shared Memory，块内复用。

```cuda
__global__ void matmul_tiled(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++) {
        // 协作加载 tile 到 Shared Memory
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();

        // 在 Shared Memory 中计算
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

### 8.5 寄存器优化

- 寄存器是最快的存储，但数量有限（每线程 255 个，每 SM 64K 个）。
- **寄存器溢出（Register Spill）** → 数据被存到 Local Memory（实际在全局内存）→ 性能暴跌。
- **Register Blocking**：在寄存器中累积更多计算结果再写回。

```cuda
// Register blocking for GEMM: 每线程计算 TM×TN 个输出元素
float accum[TM][TN] = {0.0f};
for (int k = 0; k < K; k++) {
    float a_frag[TM], b_frag[TN];
    // load fragments
    for (int m = 0; m < TM; m++) a_frag[m] = ...;
    for (int n = 0; n < TN; n++) b_frag[n] = ...;
    // outer product accumulate
    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
            accum[m][n] += a_frag[m] * b_frag[n];
}
```

### 8.6 Warp Divergence 最小化

- 同一 Warp 内的分支 divergence 会导致所有路径串行执行。
- 让分支按 Warp 粒度对齐（以 32 线程为单位）。
- 利用 predication 替代分支。

### 8.7 延迟隐藏

- **Thread-Level Parallelism (TLP)**：增加活跃 Warp 数（提高 Occupancy）。
- **Instruction-Level Parallelism (ILP)**：每个线程执行更多独立指令。
- **Memory-Level Parallelism (MLP)**：每个线程发出多个独立 load 请求。

```cuda
// ILP: 循环展开增加独立指令
#pragma unroll 4
for (int i = 0; i < N; i += 4) {
    float a0 = input[i+0], a1 = input[i+1];
    float a2 = input[i+2], a3 = input[i+3];
    sum0 += a0; sum1 += a1; sum2 += a2; sum3 += a3;
}
```

### 8.8 Kernel Fusion（核函数融合）

将多个 Kernel 合并为一个，避免中间结果写回 Global Memory：

- **Vertical Fusion**：消除 producer-consumer 之间的全局内存往返。
- **Horizontal Fusion**：合并多个并行度不足的小 Kernel，提高 SM 利用率。

```cuda
// BEFORE: 3 次全局内存读写
layernorm_kernel<<<...>>>(input, normed);      // read input, write normed
gelu_kernel<<<...>>>(normed, activated);       // read normed, write activated
linear_kernel<<<...>>>(activated, output);     // read activated, write output

// AFTER: 融合为 1 个 Kernel
fused_ln_gelu_linear<<<...>>>(input, output);  // 中间结果留在寄存器/Shared
```

---

## 9. 高级编程特性

### 9.1 Warp Shuffle 操作

Warp 内线程直接交换寄存器数据，**无需 Shared Memory**：

```cuda
// __shfl_sync:        获取 Warp 中指定 lane 的值
// __shfl_up_sync:     获取 lane - delta 的值
// __shfl_down_sync:   获取 lane + delta 的值
// __shfl_xor_sync:    获取 lane ^ laneMask 的值

// Warp 级别归约
float val = local_value;
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
}
// lane 0 持有 Warp 内所有线程 local_value 的总和

// Warp 广播
float broadcast = __shfl_sync(0xFFFFFFFF, val, 0);  // lane 0 广播给所有线程
```

### 9.2 Atomic 操作

```cuda
atomicAdd(&global_counter, 1);        // 原子加
atomicMax(&global_max, local_max);    // 原子取最大
atomicCAS(&addr, compare, val);       // Compare-And-Swap

// FP16 atomicAdd（CC 7.0+）
atomicAdd(reinterpret_cast<__half*>(addr), half_val);
```

**性能提示**：Atomic 操作到全局内存代价高昂。优先在 Shared Memory 或 Warp 级别做归约，最后一个 Warp 结果再 Atomic 到全局。

### 9.3 Cooperative Groups

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void myKernel(...) {
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(tb);

    // Warp 级归约
    float val = local_value;
    for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
        val += warp.shfl_down(val, offset);
    }

    tb.sync();  // 等同于 __syncthreads()
}
```

### 9.4 Tensor Core 编程 (WMMA)

```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// Warp Matrix Multiply-Accumulate: D = A × B + C
// 每个 Warp（32 线程）协作完成一个矩阵乘
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

fill_fragment(c_frag, 0.0f);
load_matrix_sync(a_frag, A_smem, 16);
load_matrix_sync(b_frag, B_smem, 16);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(C_smem, c_frag, 16, mem_row_major);
```

**WMMA 支持的数据类型与形状**（架构相关）：

| 架构 | 矩阵形状 (M×N×K) | A/B 类型 | C/D 类型 |
|------|-------------------|----------|----------|
| Volta (CC 7.0) | 16×16×16 | FP16 | FP16/FP32 |
| Turing (CC 7.5) | 16×16×16, 8×32×16, 32×8×16 | FP16, INT8, INT4, INT1 | FP16/FP32, INT32 |
| Ampere (CC 8.0) | 16×16×16 + 更多 | FP16, BF16, TF32, FP64 | FP16/FP32, FP64 |
| Hopper (CC 9.0) | WGMMA 指令 | FP16, BF16, FP8 (E4M3/E5M2), INT8 | FP16/FP32 |

### 9.5 Inline PTX Assembly

```cuda
// 使用 PTX 内联汇编进行精细控制
__device__ __forceinline__ float fast_exp(float x) {
    float result;
    asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(x * 1.4426950408889634f));
    return result;
}

// 使用 PTX load 指令控制缓存行为
__device__ __forceinline__ float ldg(const float* ptr) {
    float val;
    asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(val) : "l"(ptr));
    return val;
}
```

### 9.6 Hopper 特有功能 (CC 9.0)

#### TMA (Tensor Memory Accelerator)

- 硬件加速的张量数据搬运，支持 1D–5D 张量。
- 从 Global Memory 到 Shared Memory 的异步拷贝，不占用计算资源。

```cuda
// 使用 cp.async.bulk（TMA）进行张量数据传输
// 通过 CUTLASS 或 CuTe 库封装使用更为常见
```

#### Warp Group MMA (WGMMA)

- 比 WMMA 更高效的矩阵乘，4 个 Warp 组成 Warp Group 协作。
- 支持 FP8 (E4M3, E5M2) Tensor Core 操作。

#### Asynchronous Pipeline

- Producer-Consumer 模式：数据加载 Warp 和计算 Warp 异步重叠。
- 通过 barrier / arrive-wait 机制同步。

---

## 10. Compute Capability 规格表

### 10.1 SM 资源限制

| 规格 | CC 8.0 (A100) | CC 8.6 (RTX 3090) | CC 8.9 (RTX 4090) | CC 9.0 (H100) | CC 10.0 (B200) |
|------|---------------|--------------------|--------------------|----------------|-----------------|
| 每线程最大寄存器数 | 255 | 255 | 255 | 255 | 255 |
| 每 SM 寄存器文件 | 64K × 32-bit | 64K × 32-bit | 64K × 32-bit | 64K × 32-bit | 64K × 32-bit |
| 每 SM 最大 Block 数 | 32 | 16 | 24 | 32 | 32 |
| 每 SM 最大 Warp 数 | 64 | 48 | 48 | 64 | 64 |
| 每 SM 最大线程数 | 2048 | 1536 | 1536 | 2048 | 2048 |
| 每 Block 最大线程数 | 1024 | 1024 | 1024 | 1024 | 1024 |
| Shared Memory / SM | 164 KB | 100 KB | 100 KB | 228 KB | 228 KB |
| 每 Block 最大 Shared | 163 KB | 99 KB | 99 KB | 227 KB | 227 KB |
| 静态 Shared Memory 限制 | 48 KB | 48 KB | 48 KB | 48 KB | 48 KB |

### 10.2 编译目标

| 架构 | nvcc flag | 代表 GPU |
|------|-----------|----------|
| Ampere | `-arch=sm_80` / `sm_86` | A100 / RTX 3090 |
| Ada Lovelace | `-arch=sm_89` | RTX 4090, L40 |
| Hopper | `-arch=sm_90` / `sm_90a` | H100, H200 |
| Blackwell | `-arch=sm_100` / `sm_120` | B200, GB200 |

**注意**：`sm_90a` 包含架构特定加速指令，**不兼容**其他架构。

### 10.3 Shared Memory 可配置选项 (KB)

| CC 8.0 | CC 8.6 / 8.9 | CC 9.0 |
|--------|---------------|--------|
| 0, 8, 16, 32, 64, 100, 132, 164 | 0, 8, 16, 32, 64, 100 | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |

---

## 11. 面向 LLM 的 Kernel 设计模式

### 11.1 典型 LLM 算子及优化策略

| 算子 | 计算特征 | 关键优化 |
|------|----------|----------|
| GEMM (Linear) | Compute-bound | Tiling + Register blocking + Tensor Core + Double buffering |
| Self-Attention | Memory-bound → Compute-bound (fused) | FlashAttention: online softmax + tiling, 避免 O(N²) 中间矩阵 |
| LayerNorm / RMSNorm | Memory-bound | Warp 级归约 + Kernel fusion |
| Softmax | Memory-bound | Online safe softmax + Warp shuffle 归约 |
| Activation (GeLU/SiLU) | Memory-bound | 与前后 Kernel 融合（如 fused bias + activation） |
| Embedding Lookup | Memory-bound | Coalesced access + 向量化 load |
| Quantize/Dequantize | Memory-bound | 融合进 GEMM Kernel + 向量化 |
| KV-Cache Attention | Memory-bound | PagedAttention: 分页管理 + 向量化 |

### 11.2 FlashAttention 核心思想

```
传统 Attention:
  S = Q × K^T             → 写 N×N 到 HBM
  P = softmax(S)           → 读/写 N×N
  O = P × V                → 读 N×N
  内存复杂度: O(N²)

FlashAttention:
  按 tile 分块计算：
  for each K_tile, V_tile:
      S_tile = Q_tile × K_tile^T    → 留在 SRAM/寄存器
      P_tile = online_softmax(S_tile) → 留在 SRAM/寄存器
      O_tile += P_tile × V_tile       → 累积在寄存器
  内存复杂度: O(N)
```

关键技术：
1. **Online Softmax**：无需全局 max/sum，分块计算并在线修正。
2. **Tiling**：将 Q/K/V 分块加载到 Shared Memory。
3. **Recomputation**：反向传播时重算 attention 而非存储。

### 11.3 Fused Softmax 模板

```cuda
template<int BLOCK_SIZE>
__global__ void fused_softmax(const float* __restrict__ input,
                               float* __restrict__ output,
                               int N, int D) {
    // 每个 Block 处理一行
    int row = blockIdx.x;
    const float* in_row = input + row * D;
    float* out_row = output + row * D;

    // Step 1: 找最大值（Warp 级归约）
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE) {
        max_val = fmaxf(max_val, in_row[i]);
    }
    // Warp 归约
    for (int offset = 16; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    // Block 归约
    __shared__ float smem_max[32];
    if (threadIdx.x % 32 == 0) smem_max[threadIdx.x / 32] = max_val;
    __syncthreads();
    if (threadIdx.x < 32) {
        max_val = (threadIdx.x < BLOCK_SIZE / 32) ? smem_max[threadIdx.x] : -INFINITY;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    __syncthreads();
    max_val = smem_max[0]; // broadcast (store lane0 result first)
    // 简化：假设 smem_max[0] 在 Step1 归约后已正确

    // Step 2: 计算 exp 和 sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE) {
        sum += expf(in_row[i] - max_val);
    }
    // 同样做 Warp + Block 归约得到 sum ...

    // Step 3: 归一化输出
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE) {
        out_row[i] = expf(in_row[i] - max_val) / sum;
    }
}
```

### 11.4 Mixed Precision 最佳实践

```cuda
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// FP16 计算：使用 half2 向量化
__global__ void fp16_gelu(const half* __restrict__ input,
                          half* __restrict__ output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 2;  // 每线程处理 2 个 half
    if (idx + 1 < N) {
        half2 val = *reinterpret_cast<const half2*>(input + idx);
        // GELU 近似: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        float2 fval = __half22float2(val);
        fval.x = fval.x * 0.5f * (1.0f + tanhf(0.7978845608f *
                 (fval.x + 0.044715f * fval.x * fval.x * fval.x)));
        fval.y = fval.y * 0.5f * (1.0f + tanhf(0.7978845608f *
                 (fval.y + 0.044715f * fval.y * fval.y * fval.y)));
        *reinterpret_cast<half2*>(output + idx) = __float22half2_rn(fval);
    }
}
```

### 11.5 高性能 GEMM Kernel 结构

```
┌────────────────── Grid ──────────────────┐
│  Block (bm, bn) 负责输出矩阵的                │
│  [bm*BM : (bm+1)*BM, bn*BN : (bn+1)*BN]    │
│                                              │
│  for k_tile in range(K / BK):               │
│    1. 协作加载 A_tile[BM×BK]                  │
│       和 B_tile[BK×BN] 到 Shared Memory      │
│    2. __syncthreads()                        │
│    3. 每线程从 Shared Memory 加载              │
│       a_frag[TM] 和 b_frag[TN] 到寄存器       │
│    4. 外积累加: accum[TM×TN] += outer_product │
│    5. __syncthreads()                        │
│                                              │
│  写回 accum[TM×TN] 到 Global Memory          │
└──────────────────────────────────────────┘

关键参数:
- BM, BN, BK: Block tile 大小 (Shared Memory 级别)
- TM, TN: Thread tile 大小 (寄存器级别)
- 每个 Block 需要 (BM * BK + BK * BN) * sizeof(dtype) 的 Shared Memory
- 每个线程需要 TM + TN + TM*TN 个寄存器
```

### 11.6 Double Buffering（双缓冲/流水线预取）

```cuda
// Shared Memory 双缓冲：加载下一 tile 的同时计算当前 tile
__shared__ float As[2][TILE_K][TILE_M];
__shared__ float Bs[2][TILE_K][TILE_N];

// 预加载第一个 tile
load_tile(As[0], Bs[0], 0);
__syncthreads();

for (int k = 0; k < K / TILE_K; k++) {
    int cur = k % 2;
    int nxt = (k + 1) % 2;

    // 异步预取下一个 tile（如果有）
    if (k + 1 < K / TILE_K) {
        load_tile(As[nxt], Bs[nxt], k + 1);
    }

    // 使用当前 tile 计算
    compute_tile(As[cur], Bs[cur], accum);
    __syncthreads();
}
```

### 11.7 Kernel 开发检查清单

在生成或审核 CUDA Kernel 时，依次检查以下要点：

- [ ] **正确性**：边界检查、数据类型正确、数值精度可接受
- [ ] **Block 大小**：是 32 的倍数（避免部分 Warp 浪费）
- [ ] **内存合并**：连续线程访问连续地址
- [ ] **数据对齐**：访问数据类型大小 ∈ {1,2,4,8,16}B 且自然对齐（`__align__`）
- [ ] **Bank Conflict**：Shared Memory 访问无冲突（或已 padding/swizzle）
- [ ] **Warp Divergence**：分支按 Warp 粒度对齐
- [ ] **寄存器压力**：无严重 spill（`--ptxas-options=-v` 查看）
- [ ] **Occupancy**：足够的活跃 Warp 隐藏延迟
- [ ] **向量化**：使用 `float4` / `half2` 等向量类型访问内存
- [ ] **Fusion 机会**：相邻算子是否可融合
- [ ] **Shared Memory 上限**：未超过架构限制
- [ ] **Tensor Core**：矩阵乘是否启用 Tensor Core（WMMA/WGMMA）
- [ ] **`__restrict__`**：Kernel 参数指针标记 `const __restrict__`
- [ ] **错误处理**：Host 端 Kernel launch 后检查错误

---

## 参考文献

1. [NVIDIA CUDA C++ Programming Guide (Legacy)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
2. [NVIDIA CUDA Programming Guide (New)](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
3. [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
4. [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
5. [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
6. [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)
7. [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
8. [Compute Capabilities Reference](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
9. [CUDA Runtime API Reference](https://docs.nvidia.com/cuda/cuda-runtime-api/)
10. [C/C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
11. [FlashAttention: Fast and Memory-Efficient Exact Attention (Tri Dao et al.)](https://arxiv.org/abs/2205.14135)
12. [FlashAttention-2](https://arxiv.org/abs/2307.08691)
13. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://tridao.me/publications/flash3/flash3.pdf)
14. [CUTLASS: CUDA Templates for Linear Algebra](https://github.com/NVIDIA/cutlass)
15. [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
16. [Using CUDA Warp-Level Primitives (NVIDIA Blog)](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
17. [Using Shared Memory in CUDA C/C++ (NVIDIA Blog)](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
