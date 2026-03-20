# Tensor Memory Accelerator (TMA) 深度解析

> 面向 LLM 高性能 Kernel 开发的 TMA 技术参考
> 覆盖硬件架构、Tensor Map 描述符、异步拷贝流程、Swizzle、Multicast、Reduce、Blackwell 演进

---

## 目录

1. [概述与动机](#1-概述与动机)
2. [数据搬运演进：Volta → Ampere → Hopper](#2-数据搬运演进volta--ampere--hopper)
3. [TMA 硬件架构](#3-tma-硬件架构)
4. [Tensor Map 描述符 (CUtensorMap)](#4-tensor-map-描述符-cutensormap)
5. [cuTensorMapEncodeTiled 参数详解](#5-cutensormapencodetiled-参数详解)
6. [TMA Load：Global → Shared Memory](#6-tma-loadglobal--shared-memory)
7. [TMA Store：Shared → Global Memory](#7-tma-storeshared--global-memory)
8. [1D Bulk Copy vs Tensor Tiled Copy](#8-1d-bulk-copy-vs-tensor-tiled-copy)
9. [Proxy 与 Fence：内存一致性模型](#9-proxy-与-fence内存一致性模型)
10. [Swizzle 机制](#10-swizzle-机制)
11. [Multicast 与 Thread Block Cluster](#11-multicast-与-thread-block-cluster)
12. [TMA Reduce (规约操作)](#12-tma-reduce-规约操作)
13. [Im2Col 模式](#13-im2col-模式)
14. [Blackwell (SM 10.0) 中的 TMA 演进](#14-blackwell-sm-100-中的-tma-演进)
15. [LLM Kernel 中的 TMA 应用](#15-llm-kernel-中的-tma-应用)
16. [约束条件与常见问题](#16-约束条件与常见问题)
17. [PTX 指令与 API 速查表](#17-ptx-指令与-api-速查表)

---

## 1. 概述与动机

### 1.1 什么是 TMA

Tensor Memory Accelerator (TMA) 是 NVIDIA Hopper (SM 9.0) 架构引入的**专用硬件拷贝引擎**，用于在 GPU 全局内存 (GMEM) 与共享内存 (SMEM) 之间执行**异步、批量**的数据传输。

```
传统方式 (Volta/Ampere):
  每个线程: LDG(GMEM→Reg) → STS(Reg→SMEM)     ← 所有线程参与, 消耗寄存器

TMA (Hopper+):
  单个线程: 发起 TMA 指令 → 硬件自动完成 GMEM→SMEM
  其余线程: 继续计算, 等完成信号后消费数据    ← 寄存器零开销
```

### 1.2 核心优势

| 特性 | 说明 |
|------|------|
| **单线程发起** | 仅需 1 个线程即可启动整个 tile 的搬运 |
| **零寄存器开销** | 数据不经过寄存器，GMEM 直达 SMEM |
| **硬件地址生成** | 多维寻址 (1D–5D) 由 TMA 硬件完成，释放 CUDA Core 计算资源 |
| **自动越界填充 (OOB Fill)** | 越界访问自动填零或 NaN，无需手动边界检查 |
| **Swizzle 支持** | 硬件自动重排 SMEM 布局，消除 Bank Conflict |
| **Multicast** | 一次加载广播到 Cluster 内多个 CTA 的 SMEM |
| **异步执行** | 与计算完全重叠，配合 mbarrier 同步 |

### 1.3 支持架构

| 架构 | Compute Capability | TMA 支持 |
|------|-------------------|----------|
| Volta / Turing | 7.0 / 7.5 | 不支持 |
| Ampere | 8.0 / 8.6 | 不支持 (使用 cp.async) |
| **Hopper** | **9.0** | **完整支持** |
| **Blackwell DC** | **10.0** | **完整支持 + TMEM 集成** |
| Blackwell Consumer | 12.0 | 支持 (不支持 TMEM/tcgen05) |

---

## 2. 数据搬运演进：Volta → Ampere → Hopper

### 2.1 演进路径

```
Volta/Turing (SM 7.x):
  所有线程 → LDG(GMEM→Reg) → STS(Reg→SMEM) → __syncthreads()
  · 每线程计算自己的地址, 消耗寄存器
  · 数据必须经过寄存器中转
  · 同步: __syncthreads()

Ampere (SM 8.x):
  所有线程 → cp.async(GMEM→SMEM, 4/8/16B) → cp.async.commit_group → cp.async.wait_group
  · 绕过寄存器, GMEM 直达 SMEM (LDGSTS 指令)
  · 仍需每线程参与地址计算
  · 每次仅 4/8/16 字节
  · 同步: commit_group + wait_group

Hopper (SM 9.x):
  单线程 → cp.async.bulk.tensor(descriptor, coords) → mbarrier.try_wait
  · 硬件完成地址计算 (通过 Tensor Map 描述符)
  · 批量传输 (整个 tile, 可达 SMEM 上限)
  · 支持多维 (1D–5D), Swizzle, Multicast
  · 同步: mbarrier (硬件加速)
```

### 2.2 对比表

| 特性 | Volta/Turing | Ampere (`cp.async`) | Hopper (TMA) |
|------|-------------|-------------------|-------------|
| PTX 指令 | `ld.global` + `st.shared` | `cp.async` (LDGSTS) | `cp.async.bulk.tensor` |
| 拷贝粒度 | 4B per thread | 4/8/16B per thread | 整个 tile (批量) |
| 地址计算 | 每线程手动 | 每线程手动 | 硬件自动 (描述符) |
| 寄存器使用 | 数据经过寄存器 | 绕过 (部分) | 完全绕过 |
| 线程参与 | 所有线程 | 所有线程 | **单线程** |
| 维度支持 | 1D | 1D | 1D–5D |
| OOB 处理 | 手动判断 | 手动判断 | **硬件自动填充** |
| Swizzle | 手动 | 手动 | **硬件内置** |
| Multicast | 不支持 | 不支持 | **支持** |
| 同步机制 | `__syncthreads` | commit/wait group | **mbarrier** (硬件加速) |

---

## 3. TMA 硬件架构

### 3.1 TMA 单元在 SM 中的位置

```
SM (Streaming Multiprocessor) — Hopper
├── Warp Scheduler × 4
├── Register File (256 KB)
├── CUDA Cores (FP32 / INT32)
├── Tensor Core × 4
├── SFU / Load-Store Units
├── Shared Memory / L1 Cache (≤ 228 KB)
│
├── ▶ TMA Unit ◀               ← 专用硬件拷贝引擎
│   ├── 地址生成器 (多维坐标 → 线性地址)
│   ├── Swizzle 引擎
│   ├── OOB 检测 + 填充
│   ├── Multicast 分发器
│   └── Reduce 运算单元
│
└── SM-to-SM Network (DSMEM 通道)
```

### 3.2 数据流

```
GMEM (HBM3, 3 TB/s)
  │
  ├─────── L2 Cache ────────┐
  │                         │
  ▼                         ▼
TMA Unit ──────────→ Shared Memory (SMEM)
  │   (cp.async.bulk.tensor)    ↑
  │                             │
  ├── Multicast ──→ 其他 CTA 的 SMEM (via DSMEM)
  │
  └── Reduce ──→ GMEM (原子规约)
```

TMA 操作**不经过寄存器**，数据从 L2 Cache 直接写入 SMEM (或反向)。

---

## 4. Tensor Map 描述符 (CUtensorMap)

### 4.1 什么是 Tensor Map

Tensor Map 是 TMA 硬件所需的**数据结构描述符**，编码了多维张量在 GMEM 中的布局和 SMEM 中的存取方式。它在 **Host 端**创建，作为 kernel 参数传递到 Device。

```
CUtensorMap (128 bytes, opaque)
├── 张量数据类型 (FP16/BF16/FP8/INT8/...)
├── 维度数 (1–5)
├── 全局基地址 (GMEM)
├── 全局维度 (globalDim)
├── 全局步长 (globalStrides)
├── Box 尺寸 (boxDim) — 每次 TMA 传输的 tile 大小
├── 元素步长 (elementStrides)
├── Interleave 模式
├── Swizzle 模式
├── L2 Promotion 策略
└── OOB 填充模式
```

### 4.2 创建流程

```
Host 端:
  cuTensorMapEncodeTiled(&tensorMap, dataType, rank,
      globalAddr, globalDim, globalStrides,
      boxDim, elementStrides,
      interleave, swizzle, l2Promotion, oobFill);

传递到 Device:
  myKernel<<<grid, block>>>(tensorMap_as_grid_constant, ...);
  // tensorMap 必须标记为 const __grid_constant__

Device 端使用:
  cp.async.bulk.tensor.2d ... [tensorMap], {coord_x, coord_y}
```

### 4.3 更新描述符地址

当张量数据地址改变但布局不变时，可只更新基地址而不重建描述符：

```c
cuTensorMapReplaceAddress(&tensorMap, newGlobalAddress);
```

---

## 5. cuTensorMapEncodeTiled 参数详解

### 5.1 完整签名

```c
CUresult cuTensorMapEncodeTiled(
    CUtensorMap*             tensorMap,        // [out] 描述符
    CUtensorMapDataType      tensorDataType,   // 数据类型
    cuuint32_t               tensorRank,       // 维度数 (1–5)
    void*                    globalAddress,    // GMEM 基地址
    const cuuint64_t*        globalDim,        // GMEM 各维度大小
    const cuuint64_t*        globalStrides,    // GMEM 步长 (字节)
    const cuuint32_t*        boxDim,           // tile 大小 (元素)
    const cuuint32_t*        elementStrides,   // SMEM 元素步长
    CUtensorMapInterleave    interleave,       // 交错模式
    CUtensorMapSwizzle       swizzle,          // Swizzle 模式
    CUtensorMapL2promotion   l2Promotion,      // L2 提升策略
    CUtensorMapFloatOOBfill  oobFill           // 越界填充
);
```

### 5.2 各参数详解

#### tensorDataType

| 枚举值 | 说明 |
|--------|------|
| `CU_TENSOR_MAP_DATA_TYPE_UINT8` | 无符号 8-bit 整数 |
| `CU_TENSOR_MAP_DATA_TYPE_UINT16` | 无符号 16-bit 整数 |
| `CU_TENSOR_MAP_DATA_TYPE_UINT32` | 无符号 32-bit 整数 |
| `CU_TENSOR_MAP_DATA_TYPE_INT32` | 有符号 32-bit 整数 |
| `CU_TENSOR_MAP_DATA_TYPE_UINT64` | 无符号 64-bit 整数 |
| `CU_TENSOR_MAP_DATA_TYPE_FLOAT16` | FP16 |
| `CU_TENSOR_MAP_DATA_TYPE_FLOAT32` | FP32 |
| `CU_TENSOR_MAP_DATA_TYPE_FLOAT64` | FP64 |
| `CU_TENSOR_MAP_DATA_TYPE_BFLOAT16` | BF16 |
| `CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ` | FP32 flush-to-zero |
| `CU_TENSOR_MAP_DATA_TYPE_TFLOAT32` | TF32 |
| `CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ` | TF32 flush-to-zero |

#### tensorRank

- 张量维度数: 1–5
- 1D 张量使用 `cp.async.bulk.tensor.1d`，2D 使用 `cp.async.bulk.tensor.2d`，以此类推

#### globalDim 与 globalStrides

```
2D 矩阵示例 (M×K, 行主序, BF16):
  globalDim[0] = K          (列数, 最快变化维度)
  globalDim[1] = M          (行数)
  globalStrides[0] = K * sizeof(bf16)   (行步长, 字节)
  // globalStrides 长度 = rank - 1
  // 第 0 维步长隐含为 elementSize
```

**约束**: `globalStrides` 的每个元素必须是 **16 的倍数** (字节)。

#### boxDim

每次 TMA 传输的 tile 大小（元素数）：

```
加载 128×64 的 BF16 tile:
  boxDim[0] = 64     (快维: 64 × 2B = 128B)
  boxDim[1] = 128    (慢维)
```

**约束**:
- 每个维度: 1 ≤ boxDim[i] ≤ **256**
- `boxDim[0] × elementSize` 必须是 **16 字节的倍数**
- 使用 Swizzle 时: `boxDim[0] × elementSize` ≤ Swizzle 宽度

#### elementStrides

SMEM 中元素的步长（元素数而非字节），通常全部设为 **1**：

```c
uint32_t elementStrides[] = {1, 1};  // 连续放置
```

若设为 `{2, 1}`，则快维方向上每隔 1 个位置放一个元素（可用于仅加载复数的实部）。

**约束**: 1 ≤ elementStrides[i] ≤ 8

#### interleave

| 枚举值 | 说明 |
|--------|------|
| `CU_TENSOR_MAP_INTERLEAVE_NONE` | 无交错 (标准用法) |
| `CU_TENSOR_MAP_INTERLEAVE_16B` | 16 字节交错 (用于 sub-4B 数据类型加速) |
| `CU_TENSOR_MAP_INTERLEAVE_32B` | 32 字节交错 |

通常使用 `NONE`。交错模式可加速小于 4 字节的数据类型的加载。

#### swizzle

| 枚举值 | Swizzle 宽度 | 约束 |
|--------|-------------|------|
| `CU_TENSOR_MAP_SWIZZLE_NONE` | 无 | 无特殊约束 |
| `CU_TENSOR_MAP_SWIZZLE_32B` | 32 字节 | `boxDim[0] × elementSize` ≤ 32B |
| `CU_TENSOR_MAP_SWIZZLE_64B` | 64 字节 | `boxDim[0] × elementSize` ≤ 64B |
| `CU_TENSOR_MAP_SWIZZLE_128B` | 128 字节 | `boxDim[0] × elementSize` ≤ 128B |

详见 [§10 Swizzle 机制](#10-swizzle-机制)。

#### l2Promotion

| 枚举值 | 说明 |
|--------|------|
| `CU_TENSOR_MAP_L2_PROMOTION_NONE` | 无特殊 L2 策略 (默认) |
| `CU_TENSOR_MAP_L2_PROMOTION_64B` | 将 L2 cache policy 扩展到 64B 对齐 |
| `CU_TENSOR_MAP_L2_PROMOTION_128B` | 将 L2 cache policy 扩展到 128B 对齐 |
| `CU_TENSOR_MAP_L2_PROMOTION_256B` | 将 L2 cache policy 扩展到 256B 对齐 |

#### oobFill

| 枚举值 | 说明 |
|--------|------|
| `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` | 越界不填充 (默认) |
| `CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA` | 越界填 NaN (FMA 中视为 0) |

OOB Fill 是 TMA 的重要特性：当 tile 的一部分超出张量边界时，硬件自动将越界部分填零（或 NaN），**无需手动边界检查**。

### 5.3 完整示例

```cpp
#include <cuda.h>

// 2D BF16 矩阵, M=4096, K=4096, 每次加载 128×64 tile
CUtensorMap tensorMap;

uint64_t globalDim[2] = {4096, 4096};      // {K, M}
uint64_t globalStrides[1] = {4096 * 2};    // K * sizeof(bf16) = 8192 bytes
uint32_t boxDim[2] = {64, 128};            // {快维, 慢维} 元素数
uint32_t elementStrides[2] = {1, 1};

cuTensorMapEncodeTiled(
    &tensorMap,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,      // BF16
    2,                                      // 2D
    d_matrix,                               // GMEM 指针
    globalDim,
    globalStrides,
    boxDim,
    elementStrides,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,            // 128B swizzle
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
);
```

---

## 6. TMA Load：Global → Shared Memory

### 6.1 完整流程 (6 步)

```
┌─── Host 端 ──────────────────────────────────────────┐
│ Step 0: cuTensorMapEncodeTiled() 创建描述符            │
│         传递给 kernel 作为 const __grid_constant__      │
└──────────────────────────────────────────────────────┘

┌─── Device 端 (Kernel 内) ────────────────────────────┐
│                                                      │
│ Step 1: 初始化 mbarrier                               │
│   mbarrier.init(barrier, arrival_count);             │
│                                                      │
│ Step 2: fence.proxy.async  (确保 mbarrier 对 TMA 可见) │
│   fence.proxy.async.shared::cta;                     │
│                                                      │
│ Step 3: 设置预期传输字节数 (仅 1 个线程)               │
│   mbarrier.arrive.expect_tx(barrier, num_bytes);     │
│                                                      │
│ Step 4: 发起 TMA 拷贝 (仅 1 个线程)                   │
│   cp.async.bulk.tensor.2d ... [tensorMap], {x, y},   │
│       [barrier];                                     │
│                                                      │
│ Step 5: 等待完成 (所有线程)                            │
│   mbarrier.try_wait.parity(barrier, phase);          │
│                                                      │
│ Step 6: 数据就绪, 可消费 SMEM 中的 tile               │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 6.2 mbarrier 机制详解

mbarrier (异步屏障) 是驻留在 SMEM 中的 **64-bit 不透明对象**，维护两个计数器：

```
mbarrier (64-bit, 驻留 SMEM)
├── Arrival Count: 已到达的线程数 (递减至 0)
├── Transaction Bytes (tx-count): 已传输的字节数 (递减至 0)
└── Phase Bit: 每次屏障完成后翻转 (0 ↔ 1)

完成条件: Arrival Count == 0  AND  tx-count == 0
```

**工作流程**:
1. 初始化: 设置 arrival count = 需要到达的线程/操作数
2. `expect_tx`: 设置预期传输字节数 (同时消耗一个 arrival)
3. TMA 硬件完成传输时自动递减 tx-count
4. 其他线程调用 `arrive` 递减 arrival count
5. `try_wait`: 检查 phase bit 是否翻转 (即两个计数器都归零)

### 6.3 Prefetch 优化

TMA 描述符通过**常量缓存层次** (Constant Cache) 访问。可提前预取：

```c
// 预取 tensor map 到常量缓存 (可选, 减少首次 TMA 操作的延迟)
if (threadIdx.x == 0) {
    prefetch.tensormap [tensorMap];
}
```

### 6.4 PTX 伪代码

```
// 2D TMA Load
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem_ptr], [tensorMap, {coord_x, coord_y}], [mbar_ptr];
```

指令解读：
- `.2d`: 2 维张量
- `.shared::cluster`: 目标为 cluster 内的 SMEM
- `.global`: 源为 GMEM
- `.mbarrier::complete_tx::bytes`: 完成时递减 mbarrier 的 tx-count

---

## 7. TMA Store：Shared → Global Memory

### 7.1 与 TMA Load 的差异

| 差异 | TMA Load | TMA Store |
|------|----------|-----------|
| 方向 | GMEM → SMEM | SMEM → GMEM |
| 同步 | mbarrier | commit_group + wait_group |
| 关键 fence | 初始化 mbarrier 后 | SMEM 写入完成后 |

### 7.2 完整流程 (4 步)

```
Step 1: 计算结果写入 SMEM (所有线程)
  st.shared [smem_ptr], result;

Step 2: Async Proxy Fence + __syncthreads()
  fence.proxy.async.shared::cta;    // 使 SMEM 写入对 TMA 可见
  __syncthreads();                   // 确保所有线程的写入完成

Step 3: 发起 TMA Store (仅 1 个线程)
  cp.async.bulk.tensor.2d.global.shared::cta
      [tensorMap, {coord_x, coord_y}], [smem_ptr];
  cp.async.bulk.commit_group;        // 提交 store group

Step 4: 等待完成
  cp.async.bulk.wait_group 0;        // 等待所有 pending groups 完成
```

### 7.3 PTX 伪代码

```
// fence: 使 generic proxy 的 SMEM 写入对 async proxy 可见
fence.proxy.async.shared::cta;

// 2D TMA Store
cp.async.bulk.tensor.2d.global.shared::cta
    [tensorMap, {coord_x, coord_y}], [smem_ptr];

// 提交并等待
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0;
```

---

## 8. 1D Bulk Copy vs Tensor Tiled Copy

TMA 提供两种模式，分别针对 1D 连续数据和多维 tensor tile：

### 8.1 对比

| 特性 | 1D Bulk Copy (`cp.async.bulk`) | Tensor Tiled Copy (`cp.async.bulk.tensor`) |
|------|-------------------------------|-------------------------------------------|
| 描述符 | **不需要** Tensor Map | **需要** CUtensorMap |
| Host 端准备 | 无 | `cuTensorMapEncodeTiled` |
| 维度 | 仅 1D 连续 | 1D – 5D |
| 参数 | 源指针 + 目标指针 + 字节数 | 描述符 + 坐标 |
| Swizzle | 不支持 | 支持 |
| Multicast | 不支持 | 支持 |
| OOB Fill | 不支持 | 支持 |
| 地址计算 | 简单 (线性) | 硬件自动 (多维) |
| 适用场景 | 连续内存搬运 | GEMM tile、注意力矩阵 |

### 8.2 1D Bulk Copy 用法

```
// 不需要 Tensor Map, 直接传指针和大小
cp.async.bulk.shared::cluster.global [smem_dst], [gmem_src], num_bytes, [mbar];
```

### 8.3 坐标系统

Tensor Tiled Copy 使用**元素坐标**而非 tile 索引：

```
张量 shape = [1024, 512], boxDim = [64, 128]

加载第 (0,0) 个 tile: coords = {0, 0}
加载第 (1,0) 个 tile: coords = {64, 0}    ← 快维偏移 64 元素
加载第 (0,1) 个 tile: coords = {0, 128}   ← 慢维偏移 128 元素
```

---

## 9. Proxy 与 Fence：内存一致性模型

### 9.1 Proxy 概念

NVIDIA 将内存访问方法分为不同的 **Proxy（代理）**，不同 Proxy 之间的操作默认**不保证可见性**：

```
┌──────────────────────────────────────────────┐
│ Generic Proxy                                 │
│   · 常规 ld/st 指令 (线程的正常内存操作)       │
│   · 包括 ld.shared, st.shared, ld.global 等  │
└──────────────────────────────────────────────┘
              ↕ 需要 fence 保证可见性
┌──────────────────────────────────────────────┐
│ Async Proxy                                   │
│   · TMA 操作 (cp.async.bulk 系列)             │
│   · 由 TMA 硬件单元执行, 非线程执行           │
└──────────────────────────────────────────────┘
```

### 9.2 可见性规则 (关键！)

| 方向 | 是否自动可见 | 解决方案 |
|------|-------------|---------|
| Generic → Async | **否** | 需要 `fence.proxy.async.shared::cta` |
| Async → Generic | **是** | 无需 fence (TMA 完成后自动可见) |

**重要含义**:
- **TMA Load 后读 SMEM**: 不需要额外 fence，只需等待 mbarrier 完成
- **SMEM 写入后 TMA Store**: **必须**插入 `fence.proxy.async` 再发起 TMA Store

### 9.3 常见错误

```cpp
// ❌ 错误: 写入 SMEM 后直接发起 TMA Store, 没有 fence
st.shared [smem_ptr], data;
__syncthreads();
cp.async.bulk.tensor.2d.global.shared::cta ...;  // TMA 可能看不到 SMEM 数据!

// ✅ 正确: 插入 async proxy fence
st.shared [smem_ptr], data;
fence.proxy.async.shared::cta;   // ← 关键
__syncthreads();
cp.async.bulk.tensor.2d.global.shared::cta ...;
```

> **注意**: `__threadfence_block()` 不涉及 async proxy，**不能替代** `fence.proxy.async`！

---

## 10. Swizzle 机制

### 10.1 问题：SMEM Bank Conflict

SMEM 被组织为 **32 个 bank**，每 bank 宽 4 字节。当一个 warp 中多个线程同时访问同一 bank 的**不同地址**时，访问被**串行化** (bank conflict)。

```
不使用 Swizzle 的列访问 (bank conflict!):

列 0 在 SMEM 中的位置 (FP16, 每行 64 元素 = 128B):
  Row 0: bank 0-1    ← Thread 0
  Row 1: bank 0-1    ← Thread 1
  Row 2: bank 0-1    ← Thread 2
  ...                ← 8-way bank conflict!
```

### 10.2 Swizzle 原理

Swizzle 通过**XOR 操作**重排 SMEM 中数据的位置，使列方向访问分散到不同 bank：

```
Swizzle 后的列访问 (无 bank conflict):

列 0 在 SMEM 中的位置:
  Row 0: bank 0-1    ← Thread 0
  Row 1: bank 4-5    ← Thread 1  (XOR 偏移)
  Row 2: bank 8-9    ← Thread 2  (XOR 偏移)
  ...                ← 零 bank conflict!
```

### 10.3 三种 Swizzle 模式

| 模式 | 宽度 | XOR 位域 | CuTe 表示 | boxDim[0]×elemSize 约束 |
|------|------|---------|-----------|----------------------|
| `SWIZZLE_32B` | 32 字节 | bit5 ⊕ bit8 | `Swizzle<1,4,3>` | ≤ 32B |
| `SWIZZLE_64B` | 64 字节 | bit5-6 ⊕ bit8-9 | `Swizzle<2,4,3>` | ≤ 64B |
| `SWIZZLE_128B` | 128 字节 | bit5-7 ⊕ bit8-10 | `Swizzle<3,4,3>` | ≤ 128B |

### 10.4 Swizzle 的粒度

TMA Swizzle 以 **16 字节** (4 个 SMEM bank) 为单位进行重排：

```
128B Swizzle 示例 (8 个 16B 单元):

原始 GMEM 布局 (每行 128B):
  Row 0: [u0] [u1] [u2] [u3] [u4] [u5] [u6] [u7]
  Row 1: [u0] [u1] [u2] [u3] [u4] [u5] [u6] [u7]
  Row 2: [u0] [u1] [u2] [u3] [u4] [u5] [u6] [u7]
  ...

Swizzle 后 SMEM 布局:
  Row 0: [u0] [u1] [u2] [u3] [u4] [u5] [u6] [u7]    (不变)
  Row 1: [u1] [u0] [u3] [u2] [u5] [u4] [u7] [u6]    (XOR=1)
  Row 2: [u2] [u3] [u0] [u1] [u6] [u7] [u4] [u5]    (XOR=2)
  Row 3: [u3] [u2] [u1] [u0] [u7] [u6] [u5] [u4]    (XOR=3)
  Row 4: [u4] [u5] [u6] [u7] [u0] [u1] [u2] [u3]    (XOR=4)
  ...
```

效果：原本在同一列的 16B 单元现在分布在**不同 bank 组**，消除列方向 bank conflict。

### 10.5 重复周期

| 模式 | 重复周期 |
|------|---------|
| 32B Swizzle | 每 2 行重复 (256 bits / 32B) |
| 64B Swizzle | 每 4 行重复 (512 bits / 64B) |
| 128B Swizzle | 每 8 行重复 (1024 bits / 128B) |

### 10.6 Swizzle 与 WGMMA / tcgen05 的配合

TMA Swizzle 模式必须与 Tensor Core 指令的操作数布局**匹配**：

| 指令 | 推荐 Swizzle | 原因 |
|------|-------------|------|
| WGMMA (Hopper) | 128B | WGMMA 操作数在 SMEM 中需要 128B 对齐的 core matrix 布局 |
| tcgen05 (Blackwell) | 128B | 类似要求，但使用 TMEM 时有不同的 swizzle atom 尺寸 |

---

## 11. Multicast 与 Thread Block Cluster

### 11.1 什么是 TMA Multicast

TMA Multicast 将**同一个** GMEM tile 一次性加载到 Cluster 内**多个 CTA** 的 SMEM 中：

```
传统方式 (无 Multicast):
  CTA 0: TMA Load tile_A → SMEM_0     ← 1 次 GMEM 读取
  CTA 1: TMA Load tile_A → SMEM_1     ← 第 2 次 GMEM 读取 (相同数据!)
  CTA 2: TMA Load tile_A → SMEM_2     ← 第 3 次 GMEM 读取
  CTA 3: TMA Load tile_A → SMEM_3     ← 第 4 次 GMEM 读取
  总 GMEM 流量: 4×

TMA Multicast:
  TMA: Load tile_A → SMEM_0, SMEM_1, SMEM_2, SMEM_3  ← 1 次 GMEM 读取
  总 GMEM 流量: 1× (减少 4×)
```

### 11.2 Thread Block Cluster

Cluster 是 Hopper 引入的编程概念——多个 CTA 被调度到**同一 GPC** 的不同 SM 上，它们的 SMEM 通过 **SM-to-SM 网络**互连形成 **Distributed Shared Memory (DSMEM)**：

```
GPC (Graphics Processing Cluster)
├── SM_0 ─ CTA_0 ─ SMEM_0 ──┐
├── SM_1 ─ CTA_1 ─ SMEM_1 ──┤── DSMEM (低延迟互连)
├── SM_2 ─ CTA_2 ─ SMEM_2 ──┤
└── SM_3 ─ CTA_3 ─ SMEM_3 ──┘
```

### 11.3 Multicast Mask

16-bit 掩码，每 bit 对应 Cluster 中一个 CTA (最多 16 个)：

```c
// 4 个 CTA 全部参与 Multicast
uint16_t tma_mcast_mask = 0b1111;  // CTA 0,1,2,3

// 仅 CTA 0 和 CTA 2 参与
uint16_t tma_mcast_mask = 0b0101;  // CTA 0,2
```

### 11.4 协作加载模式

在 GEMM 中的典型用法——多个 CTA 分工加载，每个 CTA 的 TMA 单元都被利用：

```
Cluster: 4 个 CTA, 需要加载 4 行 tile

CTA 0: TMA multicast load row 0 → SMEM_0, SMEM_1, SMEM_2, SMEM_3
CTA 1: TMA multicast load row 1 → SMEM_0, SMEM_1, SMEM_2, SMEM_3
CTA 2: TMA multicast load row 2 → SMEM_0, SMEM_1, SMEM_2, SMEM_3
CTA 3: TMA multicast load row 3 → SMEM_0, SMEM_1, SMEM_2, SMEM_3

每个 CTA 只负责 1 行, 但每行都 multicast 到所有 CTA
→ GMEM 流量减少 4×, 所有 TMA 单元都在工作
```

### 11.5 mbarrier 与 Multicast

TMA Multicast 完成时会更新**所有参与 CTA** 的 mbarrier 的 tx-count，因此**不需要** `cluster_sync()` ——每个 CTA 的 mbarrier 独立跟踪整个 tile 的完成状态。

### 11.6 CuTe 的切分策略

CuTe 沿 `smem_tensor` 的**最慢变化维度**切分给各 CTA：

- 行主序: 沿 N 维切分
- 原因: 切分后每块数据在内存中**仍然连续**，对 TMA load 友好
- 若切分最快变化维度: 每块数据变成 strided 访问，不利于内存合并

### 11.7 性能提示

- Multicast 在 `sm_90a` 上性能最优，其他 target 可能显著降低
- 编译时务必指定 `-arch=sm_90a` (注意有 `a` 后缀)

---

## 12. TMA Reduce (规约操作)

### 12.1 概述

TMA 支持在 **SMEM → GMEM** 方向执行**元素级规约** (atomic reduction)，替代传统的 `atomicAdd`：

```
传统: 每线程 atomicAdd(gmem_ptr, value)     ← 高竞争, 逐元素
TMA:  cp.reduce.async.bulk.tensor ... add   ← 硬件批量规约, 整个 tile
```

### 12.2 支持的规约操作

| 操作 | PTX 后缀 | 说明 |
|------|---------|------|
| 加法 | `.add` | 加法规约 |
| 最小值 | `.min` | 取最小值 |
| 最大值 | `.max` | 取最大值 |
| 按位与 | `.and` | 按位与 |
| 按位或 | `.or` | 按位或 |

### 12.3 PTX 指令

```
// 2D TMA Reduce (add)
cp.reduce.async.bulk.tensor.2d.global.shared::cta.add
    [tensorMap, {coord_x, coord_y}], [smem_ptr];
```

### 12.4 优势

- **批量操作**: 一次规约整个 tile 到 GMEM，比逐元素 `atomicAdd` 减少竞争
- **异步执行**: 不阻塞 CUDA Core
- **硬件地址生成**: 同样由 TMA 描述符完成
- **DSMEM 原子操作**: 支持在 Cluster 内分布式共享内存中执行，延迟比 L2 更低

### 12.5 约束

- SMEM 源数据须 **128 字节对齐**
- 使用与 TMA Store 相同的 commit_group / wait_group 同步机制

---

## 13. Im2Col 模式

### 13.1 概述

TMA 提供专用的 Im2Col 模式，在硬件层面完成卷积运算中的 Im2Col 数据重排：

```
传统 Im2Col:
  1. 显式重排输入张量 → Im2Col 矩阵 (GMEM, 额外内存开销)
  2. 对 Im2Col 矩阵执行 GEMM

TMA Im2Col:
  1. 创建 Im2Col 描述符 (cuTensorMapEncodeIm2col)
  2. TMA 硬件自动完成重排 + 搬运, GMEM → SMEM
  3. 直接对 SMEM 数据执行 GEMM (隐式 GEMM)
```

### 13.2 API

```c
CUresult cuTensorMapEncodeIm2col(
    CUtensorMap*             tensorMap,
    CUtensorMapDataType      tensorDataType,
    cuuint32_t               tensorRank,           // 通常 4D (NCHW) 或 5D
    void*                    globalAddress,
    const cuuint64_t*        globalDim,
    const cuuint64_t*        globalStrides,
    const int*               pixelBoxLowerCorner,   // 卷积窗口下界偏移
    const int*               pixelBoxUpperCorner,   // 卷积窗口上界偏移
    cuuint32_t               channelsPerPixel,
    cuuint32_t               pixelsPerColumn,
    const cuuint32_t*        elementStrides,
    CUtensorMapInterleave    interleave,
    CUtensorMapSwizzle       swizzle,
    CUtensorMapL2promotion   l2Promotion,
    CUtensorMapFloatOOBfill  oobFill
);
```

### 13.3 PTX 指令

```
// Im2Col 模式的 TMA Load (4D)
cp.async.bulk.tensor.4d.shared::cluster.global.im2col
    [smem_ptr], [tensorMap, {c, w, h, n}], [mbar], {im2col_w, im2col_h};
```

### 13.4 适用场景

- CUTLASS 3.x 的 Implicit GEMM Conv2d / Conv3d
- 避免显式 Im2Col 的额外 GMEM 内存开销
- 支持 1D / 2D / 3D 卷积
- OOB Fill 自动处理 padding 区域

---

## 14. Blackwell (SM 10.0) 中的 TMA 演进

### 14.1 TMA 在 Blackwell 中的角色变化

```
Hopper (SM 9.0):
  TMA: GMEM ↔ SMEM
  WGMMA: SMEM operands → Register 累加器

Blackwell (SM 10.0):
  TMA: GMEM ↔ SMEM                      (不变)
  tcgen05.cp: SMEM → TMEM               (新增: 搬入 Tensor Memory)
  tcgen05.mma: TMEM × SMEM → TMEM       (新增: 累加器在 TMEM 中)
  tcgen05.st: TMEM → Register/SMEM      (新增: 搬出结果)
```

### 14.2 Tensor Memory (TMEM) — Blackwell 新增

| 属性 | 规格 |
|------|------|
| 容量 | 256 KB / SM |
| 组织 | 512 列 × 128 行 (lane) × 32-bit cell |
| 地址格式 | bit[31:16] = lane ID, bit[15:0] = column |
| 管理方式 | **显式**分配/释放 (需手动管理) |
| 用途 | 存放 MMA 累加器 (Matrix D) |

### 14.3 完整数据流对比

```
Hopper GEMM:
  GMEM ──TMA──→ SMEM ──ldmatrix──→ Reg ──WGMMA──→ Reg(累加器)

Blackwell GEMM:
  GMEM ──TMA──→ SMEM ──tcgen05.cp──→ TMEM ──┐
                  │                           ├──→ tcgen05.mma → TMEM(累加器)
                  └── operand B ──────────────┘
                                               │
                                         tcgen05.st → Reg/SMEM → TMA Store → GMEM
```

### 14.4 2-SM Cooperative MMA (CTA Pair)

Blackwell 引入 CTA Pair——两个 SM 共同执行一个 MMA：

```
TPC (Texture Processing Cluster = 2 SM)
├── SM_0 (CTA_0): TMA load A_half → SMEM_0  ─┐
├── SM_1 (CTA_1): TMA load A_half → SMEM_1  ─┤
│                                              │
└── tcgen05.mma.2sm: 使用两个 SM 的 SMEM       │
    共同计算, 结果存入两个 SM 的 TMEM          │
```

效果:
- 每个 SM 只需加载一半操作数 → SMEM 需求减半
- 等效 SMEM 容量**翻倍** (虽然物理上没增加)
- TMA Multicast 在此场景中用于高效分发操作数

### 14.5 关键差异总结

| 特性 | Hopper (SM 9.0) | Blackwell DC (SM 10.0) |
|------|----------------|----------------------|
| TMA | GMEM ↔ SMEM | GMEM ↔ SMEM (不变) |
| MMA 指令 | WGMMA (wgmma.mma_async) | tcgen05.mma |
| 累加器位置 | Register | **TMEM** (256 KB/SM) |
| MMA 发起粒度 | Warpgroup (128 线程) | **单线程** |
| 2-SM 协作 | 不支持 | **CTA Pair (TPC 粒度)** |
| 寄存器压力 | 累加器占寄存器 | 累加器在 TMEM，**释放寄存器** |
| 数据路径 | TMA → SMEM → Reg → WGMMA | TMA → SMEM → TMEM → tcgen05 |

> **Consumer Blackwell (SM 12.0)** 支持 TMA，但不支持 TMEM、tcgen05、wgmma，使用 mma.sync 扩展版。

---

## 15. LLM Kernel 中的 TMA 应用

### 15.1 GEMM / Linear Layer

TMA 是 Hopper/Blackwell 上 GEMM kernel 的**核心数据搬运引擎**：

```
Warp-Specialized GEMM Pipeline (Hopper):

Producer Warpgroup (数据搬运):
  while (tiles_remaining) {
      TMA.Load(A_tile) → SMEM_A    // 异步, 单线程发起
      TMA.Load(B_tile) → SMEM_B    // 异步, 单线程发起
      signal(mbarrier)             // 通知 consumer 数据就绪
  }

Consumer Warpgroup (计算):
  while (tiles_remaining) {
      wait(mbarrier)               // 等待数据就绪
      WGMMA(SMEM_A, SMEM_B → Acc) // Tensor Core 运算
  }

// Producer 和 Consumer 完全异步重叠!
// TMA 释放了 producer 线程的寄存器给 consumer 使用
```

### 15.2 Flash Attention

TMA 在 Flash Attention 中用于高效加载 Q/K/V tile：

```
FlashAttention-3 (Hopper):

TMA.Load(Q_block) → SMEM_Q      // 一次性加载整个 Q tile
循环 over K/V blocks:
  TMA.Load(K_block) → SMEM_K    // 异步加载下一个 K
  TMA.Load(V_block) → SMEM_V    // 异步加载下一个 V
  wait(mbarrier)                 // 等待数据就绪

  WGMMA: S = Q @ K^T            // 注意力分数
  Softmax(S)
  WGMMA: O += S @ V             // 输出累加

TMA.Store(O) → GMEM             // 结果写回
```

TMA 的 OOB Fill 在 Attention 中特别有用——sequence 末尾不完整的 block 自动填零。

### 15.3 多阶段流水线

TMA 天然支持**多阶段 (multi-stage) 流水线**：

```
Stage Pipeline (3 stages):

  SMEM_0: [加载中]  [计算中]  [空闲]   → 重新加载
  SMEM_1: [空闲]    [加载中]  [计算中] → 空闲
  SMEM_2: [计算中]  [空闲]    [加载中] → 计算中

每个 stage 有独立的 mbarrier, TMA 在不同 stage 的 SMEM buffer 间轮转
```

### 15.4 与 CUTLASS / CuTe 集成

```cpp
// Host: 构建 TMA copy 描述符
auto tma_load_A = make_tma_copy(
    SM90_TMA_LOAD{},
    gmem_layout_A,
    smem_layout_A,
    tile_shape,
    ClusterShape{}
);

// Device: 执行 TMA 拷贝
copy(tma_load_A, gmem_tensor_A, smem_tensor_A);
```

---

## 16. 约束条件与常见问题

### 16.1 关键约束汇总

| 约束 | 要求 |
|------|------|
| Compute Capability | ≥ 9.0 (Hopper) |
| `globalStrides` | 每个元素必须是 **16 字节的倍数** |
| `boxDim[i]` | 1 ≤ boxDim[i] ≤ **256** |
| `boxDim[0] × elementSize` | 必须是 **16 字节的倍数** |
| `elementStrides[i]` | 1 ≤ elementStrides[i] ≤ **8** |
| SMEM buffer 对齐 | 目标 SMEM 必须 **128 字节对齐** (`alignas(128)`) |
| GMEM 对齐 | `cudaMalloc` 返回的地址已满足 (256B 对齐) |
| GMEM 对齐 (Swizzle) | 使用 Swizzle 时 GMEM 必须 **128 字节对齐** |
| Swizzle 内维约束 | `boxDim[0] × elementSize` ≤ Swizzle 宽度 |
| Tensor Map 传递 | 必须标注 `const __grid_constant__` |
| Multicast 最优 target | 仅 `sm_90a` |
| mbarrier 对齐 | mbarrier 需 8 字节对齐 |

### 16.2 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `CUDA_ERROR_INVALID_VALUE` | `cuTensorMapEncodeTiled` 参数违反约束 | 检查 boxDim × elemSize 对齐、globalStrides 是否 16B 倍数 |
| TMA Load 数据不正确 | 缺少 `fence.proxy.async` | mbarrier 初始化后加 fence |
| TMA Store 写入旧数据 | SMEM 写入未对 async proxy 可见 | 加 `fence.proxy.async.shared::cta` |
| Multicast 性能不佳 | 非 `sm_90a` target | 编译时 `-arch=sm_90a` |
| 首次 TMA 延迟高 | 描述符未预取 | 使用 `prefetch.tensormap` |
| 小请求延迟高于 cp.async | TMA 地址生成开销未摊销 | TMA 适合大块传输；极小传输用 `cp.async` |
| Blackwell IMA/MMU fault | 描述符配置错误 (已知 driver bug) | 确保张量 backing 分配 ≥ 128 KB 且为 dense tensor |

### 16.3 何时不用 TMA

| 场景 | 原因 | 替代方案 |
|------|------|---------|
| Ampere / Turing 设备 | 硬件不支持 | `cp.async` (Ampere) 或 `ld.global` + `st.shared` |
| Consumer Blackwell (SM 12.0) | 支持 TMA | TMA 或 `cp.async` |
| 极小数据传输 (< 128B) | TMA 开销未摊销 | 直接寄存器搬运 |
| 非规则稀疏访问 | TMA 要求 16B 对齐步长 | 手动地址计算 |
| KV Cache 小块加载 | 每块过小 | 仅当每块 ≥ 16B 对齐时使用 TMA |

---

## 17. PTX 指令与 API 速查表

### 17.1 TMA 拷贝 PTX 指令

| 指令 | 方向 | 说明 |
|------|------|------|
| `cp.async.bulk.tensor.Nd.shared::cluster.global.tile` | GMEM → SMEM | Tiled TMA Load |
| `cp.async.bulk.tensor.Nd.shared::cluster.global.im2col` | GMEM → SMEM | Im2Col TMA Load |
| `cp.async.bulk.tensor.Nd.global.shared::cta` | SMEM → GMEM | TMA Store |
| `cp.reduce.async.bulk.tensor.Nd.global.shared::cta.{op}` | SMEM → GMEM | TMA Reduce (add/min/max/and/or) |
| `cp.async.bulk.shared::cluster.global` | GMEM → SMEM | 1D Bulk Copy (无描述符) |
| `cp.async.bulk.commit_group` | — | 提交 Store group |
| `cp.async.bulk.wait_group N` | — | 等待最多 N 个 pending groups |
| `fence.proxy.async.shared::cta` | — | Async proxy fence |
| `prefetch.tensormap` | — | 预取 Tensor Map 到常量缓存 |

### 17.2 mbarrier PTX 指令

| 指令 | 说明 |
|------|------|
| `mbarrier.init.shared::cta.b64` | 初始化 mbarrier (设置 arrival count) |
| `mbarrier.arrive.expect_tx.shared::cta.b64` | 到达 + 设置预期 tx 字节数 |
| `mbarrier.arrive.shared::cta.b64` | 到达 (不设 tx) |
| `mbarrier.try_wait.parity.shared::cta.b64` | 等待 phase 翻转 (非阻塞) |

### 17.3 Host API (CUDA Driver API)

| API | 说明 |
|-----|------|
| `cuTensorMapEncodeTiled` | 创建 Tiled 模式 Tensor Map |
| `cuTensorMapEncodeIm2col` | 创建 Im2Col 模式 Tensor Map |
| `cuTensorMapEncodeIm2colWide` | 创建宽 Im2Col 模式 Tensor Map |
| `cuTensorMapReplaceAddress` | 更新 Tensor Map 的 GMEM 基地址 |

> **链接方式**: `cuTensorMapEncodeTiled` 属于 CUDA Driver API，使用时需链接 `-lcuda`，或通过 `cudaGetDriverEntryPointByVersion` 运行时获取函数指针。

### 17.4 CUTLASS / CuTe 封装

| CuTe API | 对应底层操作 |
|----------|------------|
| `make_tma_copy(SM90_TMA_LOAD{}, ...)` | 构建 TMA Load 描述符 |
| `make_tma_copy(SM90_TMA_STORE{}, ...)` | 构建 TMA Store 描述符 |
| `copy(tma_op, src, dst)` | 执行 TMA 拷贝 |
| `tma_store_fence()` | `fence.proxy.async.shared::cta` |
| `tma_store_arrive()` | `cp.async.bulk.commit_group` |
| `tma_store_wait<N>()` | `cp.async.bulk.wait_group N` |
