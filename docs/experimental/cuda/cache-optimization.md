# CUDA Cache 优化深度指南

> 面向 LLM 高性能 Kernel 开发的缓存层次结构、访问策略与优化实战全面解析
> 覆盖 L1/L2 架构、Cache Line/Sector 机制、Cache 提示与策略、L2 持久化、预取、Threadblock Swizzle、Kernel Fusion 与 LLM 实战

---

## 目录

1. [GPU 缓存层次总览](#1-gpu-缓存层次总览)
2. [L1 Cache / 统一 Shared Memory 架构](#2-l1-cache--统一-shared-memory-架构)
3. [L2 Cache 架构](#3-l2-cache-架构)
4. [Cache Line、Sector 与事务机制](#4-cache-linesector-与事务机制)
5. [PTX Cache 操作符 (Load/Store 缓存提示)](#5-ptx-cache-操作符-loadstore-缓存提示)
6. [只读数据路径：Texture Cache 与 `__ldg()`](#6-只读数据路径texture-cache-与-__ldg)
7. [Constant Cache](#7-constant-cache)
8. [L2 持久化控制 (Persistent L2 Cache)](#8-l2-持久化控制-persistent-l2-cache)
9. [预取技术 (Prefetching)](#9-预取技术-prefetching)
10. [Threadblock Swizzle 与 L2 局部性](#10-threadblock-swizzle-与-l2-局部性)
11. [Kernel Fusion 与 Cache 局部性](#11-kernel-fusion-与-cache-局部性)
12. [Cache Coherence 与 Memory Fence](#12-cache-coherence-与-memory-fence)
13. [LLM Kernel Cache 优化实战](#13-llm-kernel-cache-优化实战)
14. [诊断与分析](#14-诊断与分析)
15. [Cache 优化检查清单](#15-cache-优化检查清单)

---

## 1. GPU 缓存层次总览

### 1.1 缓存层次结构

```
Thread Registers (~1 cycle)
         │
         ▼
L1 Cache / Shared Memory / Texture Cache  (~20–35 cycles)
    (每 SM 私有, 统一池)
         │
         ▼
L2 Cache  (~150–273 cycles)
    (全 SM 共享, 全局一致)
         │
         ▼
HBM / Global Memory  (~400–800 cycles)
    (DRAM, 最大容量)
```

### 1.2 各架构缓存规格对比

| 规格 | Volta V100 | Ampere A100 | Hopper H100 | Blackwell B200 | RTX 5090 |
|------|:----------:|:-----------:|:-----------:|:--------------:|:--------:|
| **CC** | 7.0 | 8.0 | 9.0 | 10.0 | 12.0 |
| **L1/Tex/SMEM 池 (每 SM)** | 128 KB | 192 KB | 256 KB | 256 KB | 128 KB |
| **最大 SMEM (每 SM)** | 96 KB | 164 KB | 228 KB | 228 KB | 128 KB |
| **最大 SMEM (每 Block)** | 96 KB | 163 KB | 227 KB | 227 KB | 99 KB |
| **L2 Cache (全局)** | ~6 MB | 40 MB | 50 MB | 126 MB | 96 MB |
| **Constant Cache (每 SM)** | 8 KB | 8 KB | 8 KB | 8 KB | 8 KB |
| **SM 数** | 80 | 108 | 132 | 148 | 170 |
| **显存容量** | 32 GB HBM2 | 80 GB HBM2e | 80 GB HBM3 | 180 GB HBM3e | 32 GB GDDR7 |
| **显存带宽** | 900 GB/s | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s | 1.79 TB/s |

> **趋势：** L1/SMEM 池从 128→256 KB (2×)，L2 从 6→126 MB (21×)，但 Tensor Core 吞吐增长更快，cache 容量相对仍然是稀缺资源。

### 1.3 各缓存的核心特征

| 缓存层次 | 延迟 | 一致性 | 可编程性 | 容量/SM |
|---------|------|--------|---------|---------|
| **L1 Data Cache** | ~28–35 cycles | SM 内一致, 跨 SM 不一致 | 部分 (carveout, cache hints) | 28–128 KB (取决于 SMEM 分配) |
| **Shared Memory** | ~20–30 cycles | Block 内可见 | 完全可编程 | 0–228 KB |
| **Texture Cache** | ~28–35 cycles | 只读 | `__ldg()` / `const __restrict__` | 与 L1 统一 |
| **L2 Cache** | ~150–273 cycles | 全局一致 | 持久化 API, cache hints | 40–126 MB 全局 |
| **Constant Cache** | ~1 cycle (广播命中) | 只读 | `__constant__` 声明 | 8 KB |

---

## 2. L1 Cache / 统一 Shared Memory 架构

### 2.1 统一 L1/Texture/Shared Memory 池

自 Volta 以来，L1 Data Cache、Texture Cache 和 Shared Memory 共享同一块片上 SRAM。分配比例 (carveout) 在运行时可配置：

```
SM 片上 SRAM 池 (以 H100 为例, 256 KB/SM):
┌─────────────────────────────────────────────┐
│              Shared Memory                  │  ← 可编程, 0–228 KB
├─────────────────────────────────────────────┤
│         L1 Data Cache / Texture Cache       │  ← 硬件管理, 28–256 KB
└─────────────────────────────────────────────┘

Shared Memory 越多 → L1 Cache 越少 (反之亦然)
```

### 2.2 配置 L1/Shared Memory 分割

```cpp
// ============= 方法 1: 百分比 Carveout =============
// 75% 分配给 Shared Memory, 25% 留给 L1
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributePreferredSharedMemoryCarveout, 75);

// ============= 方法 2: 枚举偏好 (传统 API) =============
cudaFuncSetCacheConfig(myKernel, cudaFuncCachePreferShared);
// 可选: cudaFuncCachePreferNone / PreferShared / PreferL1 / PreferEqual

// ============= 方法 3: 设备级默认 =============
cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
```

**A100 可选 SMEM 大小 (KB)：** 0, 8, 16, 32, 64, 100, 132, 164
**H100 可选 SMEM 大小 (KB)：** 0, 8, 16, 32, 64, 100, 132, 164, 196, 228

### 2.3 超过 48 KB 的 Dynamic Shared Memory

静态 `__shared__` 声明限制为 48 KB (向后兼容)。使用更多需要显式 opt-in：

```cpp
// 允许 kernel 使用最多 227 KB 动态 Shared Memory (H100)
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);

// Kernel 声明
extern __shared__ char smem_buf[];

// 启动
myKernel<<<grid, block, dynamicSmemSize>>>();
```

### 2.4 何时偏好 L1 vs. Shared Memory

| 场景 | 推荐 | 理由 |
|------|------|------|
| GEMM (大 tile, 显式 tiling) | 偏好 Shared Memory | 程序员管理的数据复用, 不依赖 L1 |
| FlashAttention | 偏好 Shared Memory | Q/K/V tile 在 SMEM 中多次复用 |
| Elementwise Kernel (无 SMEM) | 偏好 L1 | 依赖隐式 L1 缓存的 Global 访问 |
| 混合 (SMEM + 大量 Global 随机访问) | 需要 profiling | L1 和 SMEM 争用同一池 |

---

## 3. L2 Cache 架构

### 3.1 L2 Cache 基本特性

- **位置：** 全 SM 共享, 在 L1 和 HBM 之间
- **一致性：** 全局一致 (所有 SM 看到相同的 L2 数据)
- **Store 路径：** Global Memory 的所有 store 都经过 L2 (不经 L1)
- **关联度：** 未公开 (Turing 测得 16-way; Ampere/Hopper 使用非二的幂集合数和非标准哈希)

### 3.2 L2 延迟对比

| 架构 | L2 命中延迟 (cycles) |
|------|:-------------------:|
| Turing | ~188 |
| Ampere (A100) | ~200 |
| Hopper (H100) | ~273 |

> Hopper 的 L2 延迟高于 Ampere, 可能因为更大的 L2 (50 vs 40 MB) 和更复杂的 crossbar。这使得 L2 命中率优化更加重要。

### 3.3 L2 分区架构

```
A100: 单一 L2 分区, 40 MB
H100: 单一 L2, 50 MB, partitioned crossbar (就近缓存 GPC 数据)
B200: 双芯片, 4 个 L2 分区 (每芯片 2 个), 总计 126 MB
      芯片间通过 NV-HBI (10 TB/s) 互联
      跨芯片 L2 访问有额外延迟, 但比 HBM 仍快得多
```

### 3.4 B200 的 126 MB L2 对 LLM 的意义

```
FP8 模型参数缓存:
  LLaMA-7B FP8 ≈ 7 GB → 126 MB / 7 GB = 1.8% 可缓存
  但 1 层 FFN 权重 (4096 × 11008 × 2 层) ≈ 90 MB FP8 → 可缓存!

KV Cache:
  1 token, 32 heads, 128 dim, FP16 = 32 × 128 × 2 × 2 = 16 KB
  126 MB / 16 KB ≈ 8000 tokens 的 KV Cache 可驻留 L2

→ Blackwell 的大 L2 使得单层权重和短序列 KV Cache 的 L2 缓存变得可行
```

---

## 4. Cache Line、Sector 与事务机制

### 4.1 基本粒度

```
L1 Cache Line = 128 字节, 对齐到 128 字节边界
L2 Sector     = 32 字节 (1 个 cache line = 4 个 sector)
DRAM 事务     = 32 字节 (sector 级)
```

### 4.2 L1 路径 vs. L2-Only 路径

当 Warp 执行 Global Memory 加载时, 硬件有两条缓存路径：

```
L1+L2 路径 (ld.global.ca, 默认):
  Warp 请求 → L1 查找 (128B cache line 粒度)
    → 命中: 返回数据 (~28–35 cycles)
    → 未命中: 从 L2/DRAM 加载整个 128B line → 填入 L1

L2-Only 路径 (ld.global.cg):
  Warp 请求 → 绕过 L1 → L2 查找 (32B sector 粒度)
    → 命中: 返回数据 (~150–273 cycles)
    → 未命中: 从 DRAM 加载 32B sector → 填入 L2

Store 路径:
  所有 Global store → 绕过 L1 → L2 → 写回 DRAM
  粒度: 32 字节 (sector)
```

### 4.3 Sector 级缓存的含义

现代 GPU 使用 **sector-based caching**——即使分配了整个 128B cache line, 只有实际被请求的 32B sector 会被标记为有效：

```
Cache Line [128B]:
  Sector 0 [32B]: 有效 ✅ (有线程请求)
  Sector 1 [32B]: 无效 ❌ (无线程请求, 不浪费带宽)
  Sector 2 [32B]: 有效 ✅
  Sector 3 [32B]: 无效 ❌

→ 按需加载 sector, 减少不必要的 DRAM 传输
→ 但 L1 tag 查找仍以 128B line 为单位
```

### 4.4 Cache Line 对齐与 Coalescing 的交互

```
完美合并 (stride=1):
  32 线程 × 4B = 128B → 恰好 1 个 128B line → 4 个 sector
  L1 路径: 1 次 tag 查找, 4 sectors
  L2 路径: 4 次 32B 请求

非对齐 (offset=1):
  跨越 2 个 128B line → 5 个 sector
  L1 路径: 2 次 tag 查找
  L2 路径: 5 次 32B 请求 (20% 多余传输)

Stride=2:
  请求分散在 2 个 128B line 中交替的 sector
  L1 路径: 2 个 line → 8 sectors, 仅用 4 → 50% 浪费
  L2 路径: 8 × 32B sectors → 50% 浪费
```

---

## 5. PTX Cache 操作符 (Load/Store 缓存提示)

### 5.1 Load 操作符

| 操作符 | 名称 | L1 行为 | L2 行为 | 适用场景 |
|--------|------|---------|---------|---------|
| `.ca` | Cache All | 缓存 (默认) | 缓存 (默认) | 有时间局部性的数据 (重复访问) |
| `.cg` | Cache Global | **绕过** | 缓存 | 跨 CTA 共享数据 (避免 L1 污染) |
| `.cs` | Cache Streaming | 优先驱逐 | 优先驱逐 | 流式数据 (仅访问一次) |
| `.lu` | Last Use | 标记可驱逐 | — | 最后一次访问, 后续不再需要 |
| `.cv` | Cache Volatile | **绕过** | **失效重取** | CPU-GPU 轮询通信 |
| `.nc` | Non-Coherent | 只读路径 | — | 只读数据 (texture 路径) |

### 5.2 Store 操作符

| 操作符 | 名称 | L1 行为 | L2 行为 | 适用场景 |
|--------|------|---------|---------|---------|
| `.wb` | Write-Back | 绕过 (默认) | 缓存, 延迟写回 | 默认 store |
| `.cg` | Cache Global | 绕过 | 缓存 | 同 `.wb` (SM 2.0+) |
| `.cs` | Cache Streaming | 绕过 | 优先驱逐 | 流式输出 (避免 L2 污染) |
| `.wt` | Write-Through | 绕过 | 直写到系统内存 | CPU 轮询 GPU 输出 |

### 5.3 C++ 内建函数

```cpp
#include <cuda_runtime.h>

// Load intrinsics (CUDA 8.0+)
float val_ca = __ldca(ptr);    // ld.global.ca — L1+L2 缓存 (默认)
float val_cg = __ldcg(ptr);    // ld.global.cg — 仅 L2
float val_cs = __ldcs(ptr);    // ld.global.cs — 流式, 优先驱逐
float val_nc = __ldg(ptr);     // ld.global.nc — 只读 texture 路径
```

### 5.4 内联 PTX 示例

```cpp
// 向量化流式加载 (4 × float, 绕过 L1, L2 优先驱逐)
__device__ __forceinline__ float4 load_streaming(const float4* addr) {
    float4 val;
    asm("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
        : "l"(addr));
    return val;
}

// 仅 L2 缓存的加载 (绕过 L1)
__device__ __forceinline__ float load_l2_only(const float* addr) {
    float val;
    asm("ld.global.cg.f32 %0, [%1];" : "=f"(val) : "l"(addr));
    return val;
}

// 流式存储 (L2 优先驱逐)
__device__ __forceinline__ void store_streaming(float* addr, float val) {
    asm("st.global.cs.f32 [%0], %1;" :: "l"(addr), "f"(val));
}
```

### 5.5 高级驱逐优先级 (Hopper+)

Hopper PTX 扩展了更细粒度的 L1 驱逐控制：

| 修饰符 | 含义 |
|--------|------|
| `.L1::evict_normal` | 标准驱逐优先级 |
| `.L1::evict_first` | 优先驱逐 (流式提示) |
| `.L1::evict_last` | 优先保留 (持久化提示) |
| `.L1::no_allocate` | 不分配 L1 (也不检查命中, 允许读到陈旧数据) |
| `.L1::evict_unchanged` | 不改变现有驱逐优先级 |

### 5.6 选择策略

```
选择 Cache 操作符:

数据会被同一 SM 多次访问?
├── 是 → .ca (默认, L1+L2 缓存)
└── 否 → 数据会被其他 SM 访问?
          ├── 是 → .cg (绕过 L1, 避免 L1 污染, L2 一致)
          └── 否 → 数据只访问一次?
                    ├── 是 → .cs (流式, 全部优先驱逐)
                    └── 否 → .ca (默认)

只读数据? → __ldg() 或 const __restrict__
CPU-GPU 轮询? → .cv (load) + .wt (store)
```

---

## 6. 只读数据路径：Texture Cache 与 `__ldg()`

### 6.1 机制

CC 3.5 起, GPU 提供一条**只读数据缓存路径** (历史上是 Texture Cache)。两种方式触发：

```cpp
// 方法 1: 显式 __ldg() 内建函数
float val = __ldg(&data[idx]);
// → PTX: ld.global.nc.f32

// 方法 2: const __restrict__ 指针 (编译器自动优化)
__global__ void kernel(const float* __restrict__ data) {
    float val = data[idx];
    // 编译器可能自动生成 ld.global.nc
}
```

### 6.2 与 L1 Cache 的区别

| 特性 | L1 Data Cache | Texture/只读 Cache |
|------|:------------:|:-----------------:|
| 写入 | 支持 (store bypass L1 但可 invalidate) | 不支持 (只读) |
| 一致性 | 需要 `__threadfence()` 维护 | 无需 (只读 = 自然一致) |
| 硬件实现 | Volta+ 与 Texture 统一 | Volta+ 与 L1 统一 |
| PTX 指令 | `ld.global.ca` | `ld.global.nc` |
| 适用场景 | 通用 | 权重、查找表等只读数据 |

### 6.3 何时使用 `__ldg()`

**推荐：**
- 模型权重 (整个 kernel 不修改)
- 量化缩放因子 (scale/zero-point)
- 查找表 (lookup table)
- 超过 64 KB 的常量数据 (Constant Cache 装不下)

**不推荐：**
- 会被其他并发 kernel 修改的数据
- 编译器已通过 `const __restrict__` 自动优化的路径
- CC < 3.5 (回退到普通加载)

### 6.4 支持的类型

内建 `__ldg()` 重载: `char`, `short`, `int`, `long long`, `float`, `double`, `int2`, `int4`, `float2`, `float4`, `double2`, 及对应的 unsigned 版本。

---

## 7. Constant Cache

### 7.1 特性

```
Constant Memory:
  容量: 64 KB (用户) + 64 KB (编译器, kernel 参数等)
  缓存: 每 SM 8 KB Constant Cache
  物理位置: Global Memory (DRAM) 上, 通过专用 cache 访问
  访问模式: 专为 Warp 级广播优化
```

### 7.2 广播机制

```
Warp 内所有线程访问同一地址:
  → 1 次 Constant Cache 读取 → 广播到 32 线程
  → 延迟 ≈ 寄存器 (~1 cycle, 命中时)

Warp 内线程访问不同地址:
  → 序列化为 N 次读取 (N = 不同地址数)
  → 最坏: 32 个不同地址 → 32 次序列化 → 32× 慢!
```

### 7.3 使用模式

```cpp
// 声明 (全局作用域)
__constant__ float model_config[16];   // 模型超参数
__constant__ float quant_scales[256];  // 量化 scale (若 ≤ 64KB)

// Host 端写入
cudaMemcpyToSymbol(model_config, h_config, sizeof(float) * 16);

// Device 端读取
__global__ void kernel() {
    float eps = model_config[0];  // 所有线程读同一地址 → 广播 ✅
}
```

### 7.4 限制与替代

| 限制 | 替代方案 |
|------|---------|
| 总容量仅 64 KB | `__ldg()` + Global Memory (无容量限制) |
| 非均匀访问序列化 | Shared Memory (无序列化问题) |
| 只能从 Host 写入 | Shared Memory (可在 kernel 内写入) |
| 8 KB cache 工作集 | 超出 → 回退到 Global Memory 延迟 |

> **现代替代：** Hopper+ 支持 `ldu` (Load Uniform) 指令, 可以对任何 Global Memory 地址做广播加载, 减少了对 Constant Memory 的依赖。

---

## 8. L2 持久化控制 (Persistent L2 Cache)

### 8.1 概述

CC 8.0+ (Ampere 起) 支持将 L2 Cache 的一部分预留为 **持久化区域 (set-aside)**，指定的数据在该区域中优先保留, 不易被驱逐。

### 8.2 API 详解

```cpp
// ============= Step 1: 设置 L2 持久化预留大小 =============
// 查询设备支持的最大持久化 L2
int maxPersisting;
cudaDeviceGetAttribute(&maxPersisting,
    cudaDevAttrMaxPersistingL2CacheSize, deviceId);

// 预留 L2 的 75% 用于持久化 (示例)
int l2CacheSize;
cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, deviceId);
size_t setAside = min((size_t)(l2CacheSize * 0.75), (size_t)maxPersisting);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, setAside);

// ============= Step 2: 配置 Access Policy Window =============
cudaAccessPolicyWindow window = {};
window.base_ptr   = kv_cache_ptr;        // 要持久化的内存区域起始
window.num_bytes  = kv_cache_size;        // 区域大小
window.hitRatio   = min(1.0f,
    (float)setAside / kv_cache_size);     // 见 8.3 说明
window.hitProp    = cudaAccessPropertyPersisting;   // 命中→持久化
window.missProp   = cudaAccessPropertyStreaming;     // 未命中→流式

// ============= Step 3: 绑定到 CUDA Stream =============
cudaStreamAttrValue attr;
attr.accessPolicyWindow = window;
cudaStreamSetAttribute(stream,
    cudaStreamAttributeAccessPolicyWindow, &attr);

// ============= 运行 kernel =============
myKernel<<<grid, block, 0, stream>>>(kv_cache_ptr, ...);

// ============= 清理: 重置为 Normal =============
window.hitProp  = cudaAccessPropertyNormal;
window.missProp = cudaAccessPropertyNormal;
attr.accessPolicyWindow = window;
cudaStreamSetAttribute(stream,
    cudaStreamAttributeAccessPolicyWindow, &attr);
```

### 8.3 hitRatio 机制

`hitRatio` 决定窗口内多大比例的访问获得 `hitProp` 属性：

```
hitRatio = 1.0:  100% 的访问获得 Persisting 属性
hitRatio = 0.5:  50% 的访问获得 Persisting, 50% 获得 Streaming

关键约束:
  hitRatio × num_bytes ≤ persistingL2CacheSize (set-aside)
  否则持久化区域不够用 → 内部驱逐 → thrashing

示例:
  set-aside = 30 MB, kv_cache_size = 40 MB
  hitRatio = 30 / 40 = 0.75
  → 75% 的 kv_cache 驻留 L2, 25% 用流式策略
```

### 8.4 Access Property 类型

| 属性 | 行为 |
|------|------|
| `cudaAccessPropertyPersisting` | 在 set-aside 区域中优先保留, 不易被驱逐 |
| `cudaAccessPropertyStreaming` | 优先驱逐, 用于一次性数据 |
| `cudaAccessPropertyNormal` | 重置为默认驱逐策略 |

### 8.5 LLM 应用：KV Cache 驻留 L2

```
LLM Decode 推理:
  每个 token 的 attention 需要读取完整 KV Cache
  KV Cache 大小 = num_layers × num_heads × head_dim × seq_len × 2 (K+V) × dtype_size

  Llama-7B, FP16, seq_len=2048:
  32 × 32 × 128 × 2048 × 2 × 2 = 1 GB → 远超 L2

  Llama-7B, FP16, seq_len=128:
  32 × 32 × 128 × 128 × 2 × 2 = 64 MB → A100 可部分缓存, B200 可完全缓存

策略:
  - 仅持久化当前层或最近若干层的 KV Cache
  - 设置 hitRatio < 1.0 以避免 thrashing
  - 配合 PagedAttention 使用 (物理页级 persistence)
```

### 8.6 限制

- MIG (Multi-Instance GPU) 模式下不可用
- MPS 模式下只能通过环境变量 `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT` 设置
- 多个并发 kernel 共享 set-aside 区域可能互相驱逐
- 必须在 kernel 之间重置 (`cudaAccessPropertyNormal`) 以避免影响后续 kernel

---

## 9. 预取技术 (Prefetching)

### 9.1 预取方式总览

| 技术 | 最低 CC | 目标缓存 | 机制 | 适用场景 |
|------|:------:|:--------:|------|---------|
| `prefetch.global.L2` | 2.0 | L2 | PTX 指令 | 通用 L2 预热 |
| `prefetch.global.L1` | 2.0 | L1 | PTX 指令 | L1 预热 |
| `cp.async` + `.L2::128B` | 8.0 | L2 | 异步拷贝附加预取 | GEMM 流水线 |
| `cp.async.bulk.prefetch.L2` | 9.0 | L2 | TMA 专用 L2 预取 | Hopper GEMM |
| `cudaMemPrefetchAsync` | Unified Memory | 页级 | 页迁移 | Unified Memory |

### 9.2 PTX 软件预取

```cpp
// L2 预取 (通用)
__device__ __forceinline__ void prefetch_l2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}

// L1 预取
__device__ __forceinline__ void prefetch_l1(const void* ptr) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(ptr));
}

// 使用模式: 在计算当前 tile 的同时预取下一个 tile
for (int tile = 0; tile < num_tiles; tile++) {
    // 预取下一个 tile 到 L2
    if (tile + 1 < num_tiles) {
        prefetch_l2(&data[(tile + 1) * tile_size]);
    }
    // 计算当前 tile (此时预取与计算重叠)
    compute_tile(&data[tile * tile_size]);
}
```

### 9.3 `cp.async` 的 L2 预取提示 (Ampere+)

```
PTX 格式:
  cp.async.ca.shared.global.L2::128B [dst], [src], 16;
  // .L2::128B = 预取 128 字节到 L2 (实际拷贝仍按请求大小)
  // 可选: .L2::64B, .L2::128B, .L2::256B
```

CUTLASS 默认使用 `.L2::128B` 预取提示。这在 GEMM 流水线中, 当 `cp.async` 加载当前阶段数据时, 同时触发下一阶段数据的 L2 预取。

### 9.4 TMA L2 预取 (Hopper)

```
PTX 格式:
  cp.async.bulk.prefetch.L2.global [tensor_map, {coords}], size;

功能:
  将 TMA 描述符指定的张量数据预取到 L2, 但不写入 Shared Memory
  → 用于多阶段流水线中的提前预热

CUTLASS API:
  SM90_TMA_LOAD::PREFETCH
```

### 9.5 预取距离优化

```
预取距离 = 当前迭代与预取迭代之间的距离

距离太小: 预取未完成 → 仍然 cache miss
距离太大: 预取数据被驱逐 → 无效 + 浪费 L2 容量

NVIDIA 推荐: 预取距离 = 6 迭代 → 接近最优性能 (~60% 提速 vs 无预取)

最优距离取决于:
  - 每迭代的计算量 (决定了预取有多少时间完成)
  - L2 容量和关联度 (决定了预取数据能保留多久)
  - 并发 Warp 数 (更多 Warp → L2 竞争更激烈)
```

### 9.6 何时预取有效

通过 Nsight Compute 确认以下条件：

1. 内存带宽未饱和
2. 主要 stall 原因是 "Stall Long Scoreboard" (等待 DRAM 数据)
3. 循环迭代间无数据依赖 (可提前发出预取)

**LLM 实例：** H20 GPU 上对 KV Cache 做 L2 预取, L2 命中率从基线提升到 43–82%, 注意力 kernel 效率提升 2.15×, 端到端吞吐提升最高 1.97×。

---

## 10. Threadblock Swizzle 与 L2 局部性

### 10.1 问题：朴素 CTA 排布导致 L2 Thrashing

GEMM 中, 每个 CTA 计算 C 的一个 tile, 需要读取 A 的一行和 B 的一列。朴素的行优先 CTA 排布导致：

```
朴素排布 (row-major CTA order):

  CTA(0,0)  CTA(0,1)  CTA(0,2)  CTA(0,3)  ...  CTA(0,N-1)
  CTA(1,0)  CTA(1,1)  CTA(1,2)  CTA(1,3)  ...  CTA(1,N-1)
  ...

执行顺序: CTA(0,0), CTA(0,1), CTA(0,2), ..., CTA(0,N-1), CTA(1,0), ...

问题:
  CTA(0,0) 读取 B 的第 0 列 → 缓存在 L2
  CTA(0,1) 读取 B 的第 1 列 → 可能驱逐 B 第 0 列
  ...
  CTA(0,N-1) 读取 B 的第 N-1 列
  CTA(1,0) 再次需要 B 的第 0 列 → 已被驱逐 → L2 miss!

→ 如果 B 的总列数据量 > L2 大小, 每次换行都是全量 L2 miss
```

### 10.2 Swizzle 解决方案

```
Swizzle 排布 (L-shape / Z-shape / Hilbert 等):

  Swizzle=2 的执行顺序:
  CTA(0,0)  CTA(0,1)  CTA(1,0)  CTA(1,1)  CTA(0,2)  CTA(0,3)  CTA(1,2)  CTA(1,3) ...
  └─ 2×2 块 ─┘         └─ 2×2 块 ─┘         └─ 2×2 块 ─┘         └─ 2×2 块 ─┘

优势:
  CTA(0,0) 和 CTA(1,0) 接近执行 → 共享 B 的第 0 列 (仍在 L2 中)
  CTA(0,0) 和 CTA(0,1) 接近执行 → 共享 A 的第 0 行 (仍在 L2 中)

→ B 列的 L2 复用距离从 N 降到 swizzle_size
→ A 行的 L2 复用距离从 1 保持为 1
```

### 10.3 CUTLASS 中的 Swizzle

```cpp
// CUTLASS 3.x Swizzle 配置
// max_swizzle_size 控制 swizzle 的宽度
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    // ...
    EpilogueTileType,
    cute::C<max_swizzle_size>  // swizzle 宽度
>::CollectiveOp;
```

### 10.4 实测效果

| 技术 | L2 命中率提升 | 性能提升 | 来源 |
|------|:-----------:|:--------:|------|
| Threadblock Swizzle (CUTLASS) | +23.9% (avg) | 1.29× (avg) | SwizzlePerf |
| Grouped Launch (MoE GEMM) | +60% | 1.33× | PyTorch MoE Blog |
| Sawtooth Wavefront Reorder | -50% L2 miss | 1.3→2.4 TFLOPS | arXiv 2601.16032 |
| HyTiS Tile Scheduling | — | 2.08× vs cuBLAS (A100) | SC'25 |

### 10.5 CTA Cluster 与 L2 (Hopper)

```
Hopper Thread Block Cluster:
  多个 CTA 调度到同一 GPC (GPU Processing Cluster)
  → 共享 GPC 本地的 L2 分区
  → TMA Multicast: 1 次 HBM 读取 → L2 → 同时分发到 cluster 内所有 CTA

效果:
  不用 Multicast: N 个 CTA 各读 B tile → N 次 HBM 访问
  使用 Multicast: 1 次 HBM → L2 → 广播 → N 个 CTA 同时收到
  L2 命中率理论提升: (N-1)/N
```

---

## 11. Kernel Fusion 与 Cache 局部性

### 11.1 为什么 Fusion 优化 Cache

```
不融合 (Independent Kernels):
  Kernel 1: 读 Global → 计算 → 写 Global (结果 A)
  Kernel 2: 读 Global (结果 A) → 计算 → 写 Global (结果 B)
  → 结果 A 写到 HBM 再读回 → 2 次 HBM 往返

融合 (Fused Kernel):
  Kernel: 读 Global → 计算 1 → 中间结果在寄存器/SMEM/L1 → 计算 2 → 写 Global
  → 中间结果不经 HBM → 节省 2 次 HBM 传输

内存流量节省:
  2 个独立 kernel: 读 A + 写 A + 读 A + 写 B = 4D bytes
  1 个融合 kernel: 读 A + 写 B = 2D bytes (节省 50%)
```

### 11.2 LLM 推理中的关键融合模式

| 融合模式 | 融合的操作 | 中间结果位置 | 节省 |
|---------|----------|------------|------|
| **Fused Attention (FlashAttention)** | QK^T + Softmax + @V | SMEM + Registers | 避免 N×N attention 矩阵写 HBM |
| **Fused Add + RMSNorm** | Residual Add + Norm | Registers | 1 次 HBM 往返 |
| **Fused SwiGLU** | Gate Proj + Up Proj + SiLU + Mul | Registers | 1 次 HBM 往返 |
| **Fused QKV Proj** | Q Proj + K Proj + V Proj | Registers → SMEM | 读 1 次 input |
| **Fused Dequant + GEMM** | Dequantize + MatMul | Registers | 避免 dequant 结果写 HBM |

### 11.3 FlashAttention 的 Cache 优化

FlashAttention 将 Q, K, V 按 tile 加载到 Shared Memory, 使得:

```
标准 Attention:
  1. S = Q × K^T → 写 S 到 HBM (N × N 矩阵)
  2. P = softmax(S) → 写 P 到 HBM
  3. O = P × V → 写 O 到 HBM
  HBM 访问 = O(N² + Nd)

FlashAttention:
  Loop over K/V tiles:
    1. 加载 K_tile, V_tile 到 SMEM (~50 KB)
    2. S_tile = Q_tile × K_tile^T → 在寄存器中
    3. P_tile = online_softmax(S_tile) → 在寄存器中
    4. O_accum += P_tile × V_tile → 在寄存器中
  HBM 访问 = O(N²d² / M) 其中 M = SMEM 大小

关键:
  - S (N×N) 和 P (N×N) 从未写入 HBM
  - 内存节省 = N² × 2 × sizeof(float) (对于 seq_len=4096, 这是 128 MB)
  - SMEM 工作集仅取决于 tile 大小和 head_dim, 不取决于 seq_len
```

### 11.4 Fusion 的代价与限制

```
Fusion 并非无成本:

1. 寄存器压力增加 (多个操作的中间结果共享寄存器)
2. 代码复杂度增加 (需要手动管理所有 tiling 和同步)
3. Occupancy 可能下降 (更多 SMEM + 更多寄存器)
4. 编译时间增加 (更大的 kernel)

何时不该融合:
  - 两个操作的访问模式不兼容 (不同的 tiling 维度)
  - 融合后寄存器 spill 超过 HBM 往返节省
  - 操作之间有全局同步需求 (需要 grid sync)
```

---

## 12. Cache Coherence 与 Memory Fence

### 12.1 GPU Cache 一致性模型

```
L1 Cache:
  ✅ SM 内一致 (同一 block 的线程看到一致的 L1 视图)
  ❌ 跨 SM 不一致 (不同 block/SM 的 L1 互不感知)
  → Global store 绕过 L1 并失效匹配 L1 行

L2 Cache:
  ✅ 全局一致 (所有 SM 共享同一 L2)
  → store 写入 L2 后, 其他 SM 可通过 L2 看到

HBM:
  ✅ 全局一致 (最终一致点)
```

### 12.2 Memory Fence 函数

| 函数 | 可见性范围 | PTX 指令 | 成本 |
|------|----------|----------|------|
| `__threadfence_block()` | 同一 Block | `MEMBAR.SC.CTA` | 低 |
| `__threadfence()` | 同一 GPU | `MEMBAR.SC.GPU` | 中 |
| `__threadfence_system()` | 系统 (CPU+GPU) | `MEMBAR.SC.SYS` | 高 |

### 12.3 Fence 的作用

```
Fence 保证的是 **顺序性**, 不是 **同步性**:

线程 A:                       线程 B (不同 Block):
  store X = 1                   load X → 可能看到 0 或 1
  __threadfence()               (无保证, 因为没有同步)
  store flag = 1

线程 A:                       线程 B (不同 Block):
  store X = 1                   while (load flag != 1) {}
  __threadfence()               __threadfence()
  store flag = 1                load X → 保证看到 1 ✅

关键: __threadfence() 保证 "flag=1 被看到时, X=1 也已经被看到"
      但不保证 "flag=1 何时被看到"
```

### 12.4 `__syncthreads()` vs `__threadfence_block()`

```
__syncthreads():
  1. 内存屏障 (fence) — 同 __threadfence_block()
  2. 执行屏障 (barrier) — 所有线程到达后才继续
  → 更强, 但更贵

__threadfence_block():
  1. 仅内存屏障
  → 不等待其他线程, 仅保证自己的写入对 block 内可见
```

### 12.5 跨 Block 通信的 Cache 注意事项

```cpp
// ❌ 错误: L1 cache 可能返回陈旧数据
__global__ void bad_inter_block(int* flag, float* data) {
    if (blockIdx.x == 0) {
        data[0] = 42.0f;
        __threadfence();     // 刷新到 L2
        flag[0] = 1;
    } else {
        while (flag[0] != 1) {}  // 可能从 L1 cache 读到旧的 flag
        float val = data[0];      // 可能从 L1 cache 读到旧的 data
    }
}

// ✅ 正确: 使用 volatile 或 ld.cv 绕过 L1
__global__ void good_inter_block(volatile int* flag, volatile float* data) {
    if (blockIdx.x == 0) {
        data[0] = 42.0f;
        __threadfence();
        flag[0] = 1;            // volatile store → 不缓存在 L1
    } else {
        while (flag[0] != 1) {}  // volatile load → 每次都从 L2 读
        __threadfence();
        float val = data[0];     // volatile load → 从 L2 读最新值
    }
}
```

---

## 13. LLM Kernel Cache 优化实战

### 13.1 GEMM: 多级 Cache Tiling

```
GEMM 的 Cache 层次利用:

Level 4: Global Memory (HBM) → 全矩阵 A[M,K], B[K,N], C[M,N]
Level 3: L2 Cache → Threadblock Swizzle 优化, B 列 tile 复用
Level 2: Shared Memory → CTA tile (e.g., 128×128×32)
Level 1: Registers → Warp tile (e.g., 64×64), MMA fragment

数据流 (Ampere 多阶段流水线):
  HBM → [cp.async + L2 预取] → SMEM (双/三缓冲)
  SMEM → [ldmatrix] → Registers (MMA fragment)
  Registers → [mma.sync] → Registers (累加器)
  Registers → [epilogue] → SMEM → HBM

Cache 优化要点:
  1. Swizzle CTA 排布 → 提升 L2 复用 (+60% hit rate)
  2. 多阶段流水线 → 重叠加载与计算
  3. cp.async .L2::128B → 预取下一阶段数据到 L2
  4. ldmatrix → Warp 级 SMEM 读取 (自动处理 bank conflict)
```

### 13.2 FlashAttention: L2 友好的 Tile 顺序

```
FlashAttention-2 Loop Order (L2 优化):

外层循环: Q blocks (行方向)
  内层循环: K/V blocks (列方向)

  Q_block 保持在 SMEM (不变)
  K/V_block 流式加载 → 每个仅用一次 → 可用 .cs 避免 L2 污染

  优势:
  - Q_block 只加载 1 次, 在 SMEM 中重复使用 O(N/Bc) 次
  - K/V 流式通过 → 不占 L2 有效容量
  - 最终 O_block 一次性写出
```

### 13.3 KV Cache: L2 持久化策略

```cpp
// ============= Decode Phase: KV Cache L2 Persistence =============
// 在 decode 阶段, 每个新 token 需要读取完整 KV Cache
// 保持 KV Cache 在 L2 中可显著减少 HBM 流量

void setup_kv_cache_persistence(
    void* kv_cache_ptr,
    size_t kv_cache_bytes,
    cudaStream_t stream
) {
    // 查询可用 L2 set-aside
    int l2CacheSize;
    cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, 0);
    size_t setAside = min(kv_cache_bytes, (size_t)(l2CacheSize * 0.75));
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, setAside);

    // 配置 persistence
    cudaAccessPolicyWindow window = {};
    window.base_ptr  = kv_cache_ptr;
    window.num_bytes = kv_cache_bytes;
    window.hitRatio  = min(1.0f, (float)setAside / kv_cache_bytes);
    window.hitProp   = cudaAccessPropertyPersisting;
    window.missProp  = cudaAccessPropertyStreaming;

    cudaStreamAttrValue attr;
    attr.accessPolicyWindow = window;
    cudaStreamSetAttribute(stream,
        cudaStreamAttributeAccessPolicyWindow, &attr);
}
```

### 13.4 RMSNorm / LayerNorm: L1 友好

```
RMSNorm 访问模式:
  Pass 1: 读 input[hidden_dim] → 累加 sum_sq → 写 sum_sq (寄存器)
  Pass 2: 读 input[hidden_dim] + 读 weight[hidden_dim] → 写 output[hidden_dim]

  input 被读 2 次 → L1 缓存可复用!

Cache 优化:
  1. 偏好 L1: 不使用 SMEM → 将 SMEM 容量让给 L1
     cudaFuncSetCacheConfig(rmsnorm_kernel, cudaFuncCachePreferL1);

  2. 如果 hidden_dim × sizeof(half) < L1 容量:
     第 2 pass 的 input 加载会命中 L1 ✅

  3. 如果 hidden_dim 太大 (>4096 with FP16, >32KB):
     考虑显式缓存到 SMEM (1 pass 方案):
     读 input → SMEM → 从 SMEM 计算 sum_sq → 从 SMEM 归一化输出
```

### 13.5 Streaming Kernel: 避免 Cache 污染

```cpp
// ============= Elementwise 流式 Kernel (一次性数据) =============
// 场景: output = input * scale (每个元素只访问一次)
// 优化: 使用 .cs 避免 L1/L2 cache 被一次性数据填满

__global__ void scale_streaming(
    float* __restrict__ output,
    const float* __restrict__ input,
    float scale, int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // 流式加载: 不污染 L1/L2
        float val = __ldcs(&input[tid]);

        val *= scale;

        // 流式存储: 不驻留 L2
        __stcs(&output[tid], val);  // 需要内联 PTX
    }
}

// 为什么这有用:
// 如果 scale_streaming 在 GEMM 之前运行,
// 使用 .cs 可以避免驱逐 GEMM 需要的 B 矩阵缓存行
```

### 13.6 量化推理: Weight 缓存策略

```
Weight-Only Quantization GEMV (batch_size=1):
  每个输出元素需要读取一整行 weight

  FP8 weight 矩阵: [4096, 4096] = 16 MB
  → A100 L2 (40 MB) 可完全缓存!
  → 多次 decode step 之间, weight 在 L2 中复用

策略:
  1. Weight 加载使用 .ca (默认) → 保留在 L1 + L2
  2. 输入 activation 使用 .cg → 仅 L2 (每步不同, 无需 L1)
  3. 输出使用 .cs → 流式写出, 不占 L2

  或者使用 L2 Persistence API:
  - 将 weight 设为 Persisting
  - 将 activation/output 设为 Streaming
  → weight 始终在 L2, activation/output 不干扰
```

---

## 14. 诊断与分析

### 14.1 Nsight Compute Cache Metrics

```bash
# L1 Cache 命中率
ncu --metrics l1tex__t_sector_hit_rate.pct ./kernel

# L2 Cache 命中率
ncu --metrics lts__t_sector_hit_rate.pct ./kernel

# 完整的内存分析
ncu --section MemoryWorkloadAnalysis_Tables ./kernel

# 细粒度 L1 和 L2 指标
ncu --metrics \
    l1tex__t_sector_hit_rate.pct,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    lts__t_sector_hit_rate.pct,\
    lts__t_sectors_srcunit_tex_op_read.sum,\
    dram__bytes_read.sum.per_second,\
    dram__bytes_write.sum.per_second \
    ./kernel
```

### 14.2 关键 Metric 解读

| Metric | 含义 | 目标值 |
|--------|------|--------|
| `l1tex__t_sector_hit_rate.pct` | L1 sector 命中率 | >80% (良好) |
| `lts__t_sector_hit_rate.pct` | L2 sector 命中率 | >70% (良好) |
| `dram__bytes_read.sum.per_second` | DRAM 读带宽 | 接近峰值 → memory-bound |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 实际 occupancy | 取决于 kernel 类型 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | SMEM bank conflict 数 | 越低越好 |

### 14.3 缓存命中率解读指南

```
L1 命中率分析:
  >90%  → 优秀, 工作集完全在 L1 中
  60–90% → 良好, 部分数据复用
  <60%  → 需要优化: 检查访问模式, 考虑 SMEM tiling
  ~0%   → 可能使用了 .cg (故意绕过 L1) → 不一定是问题

L2 命中率分析:
  >90%  → 工作集在 L2 中 → 可能 memory-bound 但不是 HBM-bound
  50–90% → 部分复用
  <50%  → 工作集远超 L2 或访问模式差
  → 考虑: Threadblock Swizzle, L2 Persistence, 预取

DRAM 带宽利用率:
  >80% → 已 memory-bound → 减少数据量 (量化, fusion) 或增加计算强度
  <50% → 可能 compute-bound 或 latency-bound (检查 occupancy)
```

### 14.4 注意: Nsight Compute 的 Cache Flush

Nsight Compute **默认在每次 replay pass 前刷新所有 GPU 缓存**。这意味着 profiled 命中率反映的是**冷缓存**行为, 与生产环境 (暖缓存, 连续 kernel 间 L2 复用) 可能有显著差异。

对于 L2 Persistence 分析, 需要特别注意这一点。可以通过 `--replay-mode application` 避免 cache flush, 但会引入其他测量误差。

---

## 15. Cache 优化检查清单

### 15.1 通用 Cache 优化

- [ ] 确认 kernel 是否 memory-bound (`dram__bytes.per_second` vs 峰值)
- [ ] 检查 L1 命中率 (>80% 目标) 和 L2 命中率 (>70% 目标)
- [ ] 确认访问模式是合并的 (`sectors_per_request` ≤ 5)
- [ ] 一次性数据使用 `.cs` / `.cg` 避免 cache 污染
- [ ] 只读数据使用 `__ldg()` 或 `const __restrict__`
- [ ] 均匀访问的小常量使用 `__constant__` (≤ 64 KB)

### 15.2 L1 / Shared Memory 优化

- [ ] SMEM-heavy kernel: 调高 SMEM carveout
- [ ] 无 SMEM kernel: 偏好 L1 (`cudaFuncCachePreferL1`)
- [ ] 数据被同一 Block 多次访问 → 显式缓存到 SMEM
- [ ] 超过 48 KB SMEM → `cudaFuncSetAttribute` opt-in

### 15.3 L2 Cache 优化

- [ ] GEMM kernel: 使用 Threadblock Swizzle 提升 L2 复用
- [ ] 热点数据 (KV Cache, Weight): 考虑 L2 Persistence API
- [ ] 多阶段流水线: 使用 `cp.async .L2::128B` 预取
- [ ] Hopper: 使用 `cp.async.bulk.prefetch.L2` TMA 预取
- [ ] 设置 `hitRatio` 使 `hitRatio × num_bytes ≤ set-aside`
- [ ] 并发 kernel: 避免互相驱逐 L2 persistent 数据

### 15.4 Cache 策略决策表

```
数据特征              → 推荐 Cache 策略

热点权重 (每步复用)   → .ca + L2 Persistence
流式输入 (仅用一次)   → .cs 或 .cg (避免 L1/L2 污染)
只读查找表            → __ldg() 或 const __restrict__
小常量 (<64KB)        → __constant__ (广播优化)
GEMM 的 B 矩阵       → .ca + Threadblock Swizzle
FlashAttention K/V    → 流式加载到 SMEM, .cs 存储
KV Cache (decode)     → L2 Persistence + 预取
Epilogue 输出         → .cs 流式写出
```

### 15.5 常见陷阱

```
🚩 陷阱 1: 忽略 Nsight Compute 的 cache flush
   Profiled L2 命中率是冷缓存数据, 实际可能更高
   → 对比 kernel 间的 L2 复用需要 application-level profiling

🚩 陷阱 2: 盲目最大化 SMEM
   更多 SMEM → 更少 L1 → 隐式缓存的 Global 访问变慢
   → 需要 profiling 找平衡点

🚩 陷阱 3: L2 Persistence 不重置
   上一个 kernel 的 Persistence 窗口影响后续 kernel
   → 每次使用后重置为 cudaAccessPropertyNormal

🚩 陷阱 4: 朴素 CTA 排布导致 L2 thrashing
   GEMM 中不用 Swizzle → B 矩阵列的 L2 复用距离 = N
   → 使用 Threadblock Swizzle

🚩 陷阱 5: 过度融合导致寄存器溢出
   Fusion 减少 HBM 流量, 但增加寄存器压力
   → 如果 spill 量超过 fusion 节省, 反而更慢
   → 参考 reduce-register-pressure.md

🚩 陷阱 6: 对 L2 延迟的错误预期
   H100 L2 延迟 ~273 cycles (比 A100 的 ~200 高)
   → L2 miss 的代价更大 → L2 优化更重要
```

---

## 参考资源

- [NVIDIA CUDA Programming Guide: L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)
- [NVIDIA Blog: Unlock GPU Performance — Global Memory Access in CUDA](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/)
- [NVIDIA Blog: Boosting Application Performance with GPU Memory Prefetching](https://developer.nvidia.com/blog/boosting-application-performance-with-gpu-memory-prefetching/)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [NVIDIA PTX ISA: Cache Operators](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [Lei Mao: CUDA L2 Persistent Cache](https://leimao.github.io/blog/CUDA-L2-Persistent-Cache/)
- [Lei Mao: CUDA Constant Memory](https://leimao.github.io/blog/CUDA-Constant-Memory/)
- [Modal GPU Glossary: Registers](https://modal.com/gpu-glossary/device-software/registers)
- [Modal GPU Glossary: Occupancy](https://modal.com/gpu-glossary/perf/occupancy)
- [Understanding GPU Caches (RasterGrid)](https://www.rastergrid.com/blog/gpu-tech/2021/01/understanding-gpu-caches/)
- [GPU L2 Cache Persistence (veitner.bearblog)](https://veitner.bearblog.dev/gpu-l2-cache-persistence/)
- [GPU Cache Hierarchy (charlesgrassi.dev)](https://charlesgrassi.dev/blog/gpu-cache-hierarchy/)
- [PERKS: Locality-Optimized Execution on GPUs (arXiv)](https://arxiv.org/pdf/2204.02064)
- [Sawtooth Wavefront Reordering (arXiv)](https://arxiv.org/html/2601.16032v1)
- [PyTorch Blog: Accelerating MoEs with Persistent Cache-Aware Grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)
- [Colfax Research: CUTLASS Tutorial — Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
- [Colfax Research: Tutorial — Hopper TMA](https://research.colfax-intl.com/tutorial-hopper-tma/)
- [arXiv: Accelerating LLM Inference via Async KV Cache Prefetching](https://arxiv.org/pdf/2504.06319)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Microbenchmarking Blackwell (arXiv)](https://arxiv.org/html/2512.02189v1)
- [Dissecting the Hopper Architecture (arXiv)](https://arxiv.org/html/2501.12084v1)

---

*本文档作为 LLM Kernel Agent 的 Cache 优化技能参考。与 `coalesced-memory-access.md`（合并访问）、`conflict-free-accesses.md`（Shared Memory Bank Conflict）、`vectorization.md`（向量化）、`reduce-register-pressure.md`（寄存器压力）、`tensor-memory-accelerator.md`（TMA）配合使用。*
