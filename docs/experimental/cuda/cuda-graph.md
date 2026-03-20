# CUDA Graph 深度解析

> 面向 LLM 高性能 Kernel 开发的 CUDA Graph 技术参考
> 覆盖生命周期、创建方式、节点类型、更新策略、Conditional Nodes、Device-Side Launch、LLM 推理实践

---

## 目录

1. [概述与动机](#1-概述与动机)
2. [生命周期：定义 → 实例化 → 执行](#2-生命周期定义--实例化--执行)
3. [Graph 创建方式一：Stream Capture](#3-graph-创建方式一stream-capture)
4. [Graph 创建方式二：Explicit Graph API](#4-graph-创建方式二explicit-graph-api)
5. [Graph Node 类型全景](#5-graph-node-类型全景)
6. [Graph 实例化与启动](#6-graph-实例化与启动)
7. [Graph 更新策略](#7-graph-更新策略)
8. [Conditional Nodes (CUDA 12.4+)](#8-conditional-nodes-cuda-124)
9. [Device-Side Graph Launch (CUDA 12.4+)](#9-device-side-graph-launch-cuda-124)
10. [内存管理节点](#10-内存管理节点)
11. [Edge Data 与 Programmatic Dependent Launch (CUDA 12.3+)](#11-edge-data-与-programmatic-dependent-launch-cuda-123)
12. [LLM 推理中的 CUDA Graph 实践](#12-llm-推理中的-cuda-graph-实践)
13. [性能数据与最佳实践](#13-性能数据与最佳实践)
14. [API 速查表](#14-api-速查表)

---

## 1. 概述与动机

### 1.1 什么是 CUDA Graph

CUDA Graph 是一种**工作提交模型**，将一系列 GPU 操作 (kernel launch、memcpy、memset 等) 及其依赖关系组织为**有向无环图 (DAG)**，并将**定义与执行分离**：

```
传统 Stream 模型:
  CPU ──launch──> K1 ──launch──> K2 ──launch──> K3 ──launch──> K4
       [开销]         [开销]         [开销]         [开销]

CUDA Graph 模型:
  CPU ──graphLaunch──> ┌─ K1 ─→ K2 ─┐
                       │             ├─→ K4
                       └─ K3 ───────┘
       [一次开销]      [GPU 连续执行, 无 CPU 干预]
```

### 1.2 为何需要 CUDA Graph

传统 Stream 模型下，每次 kernel launch 需要 CPU 端完成参数封装、驱动调用、命令提交等操作。这些 **CPU 端开销**在 kernel 执行时间很短时成为性能瓶颈：

| 场景 | 典型 Kernel 执行时间 | CPU Launch 开销 | 瓶颈 |
|------|---------------------|----------------|------|
| LLM Decode (单 token) | 2–10 μs | 3–7 μs/kernel | CPU bound |
| 大 GEMM (Prefill) | 100+ μs | 3–7 μs/kernel | GPU bound |
| 训练 (大 batch) | 50+ μs | 3–7 μs/kernel | GPU bound |

**LLM Decode 阶段**是 CUDA Graph 收益最大的场景：每步仅生成 1 个 token，单个 kernel 执行时间极短，而一个 Transformer forward pass 可能包含**数百次** kernel launch。

### 1.3 核心优势

- **消除逐次 launch 开销**: 数百个 kernel launch 合并为单次 `cudaGraphLaunch`
- **Driver 全局优化**: 驱动获得完整 DAG 信息，可优化调度、内存布局、并发度
- **减少 GPU idle**: 传统模型中 kernel 间存在 1–3 μs 的切换间隔 (gap)，Graph 模型下降至 ~60 ns/node
- **确定性执行**: 相同 Graph 每次执行行为完全一致，利于调试和 profiling

### 1.4 性能提升数据

| 工作负载 | 无 Graph | 有 Graph | 加速比 |
|---------|---------|---------|--------|
| LLaMA-7B Decode | 30 tokens/s | 69 tokens/s | **2.3×** |
| 20 个短 kernel 序列 | 3.8 μs/kernel | 3.4 μs/kernel | 1.12× |
| DL Inference 通用 | — | — | **2–10×** |

---

## 2. 生命周期：定义 → 实例化 → 执行

CUDA Graph 的使用分为三个阶段：

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. 定义      │────→│  2. 实例化    │────→│  3. 执行      │
│  (Definition) │     │(Instantiation)│     │ (Execution)  │
│              │     │              │     │              │
│ cudaGraph_t  │     │cudaGraphExec_t│     │cudaGraphLaunch│
│ 描述拓扑+参数 │     │ 预初始化、验证 │     │ 提交到 Stream │
│ 可修改       │     │ 快照,不可改拓扑│     │ 可重复执行    │
└──────────────┘     └──────────────┘     └──────────────┘
     ↑                                          │
     └──── 需要改拓扑时重新定义 + 重新实例化 ─────┘
```

### 2.1 定义 (Definition)

创建 `cudaGraph_t` 对象，定义节点和依赖。两种方式：
- **Stream Capture**: 捕获 stream 上提交的操作
- **Explicit API**: 使用 `cudaGraphAdd*Node` 手动构建

### 2.2 实例化 (Instantiation)

```c
cudaGraphExec_t execGraph;
cudaGraphInstantiate(&execGraph, graph, flags);
```

实例化**冻结**图拓扑，执行以下优化：
- 验证图结构合法性
- 预分配内部资源
- 预计算调度和 kernel launch 参数
- 生成 GPU 端命令缓冲 (command buffer)

实例化的开销通常为 **~400 μs**，但只执行**一次**，后续反复 launch 时摊销。

### 2.3 执行 (Execution)

```c
cudaGraphLaunch(execGraph, stream);
```

可在**任意 stream** 上反复执行，无需重新实例化。

### 2.4 完整最小示例

```cpp
// 1. 定义: Stream Capture
cudaGraph_t graph;
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    kernel_A<<<grid, block, 0, stream>>>(args_A);
    kernel_B<<<grid, block, 0, stream>>>(args_B);
    kernel_C<<<grid, block, 0, stream>>>(args_C);
cudaStreamEndCapture(stream, &graph);

// 2. 实例化
cudaGraphExec_t execGraph;
cudaGraphInstantiate(&execGraph, graph, 0);

// 3. 反复执行
for (int i = 0; i < num_iterations; i++) {
    cudaGraphLaunch(execGraph, stream);
}
cudaStreamSynchronize(stream);

// 清理
cudaGraphExecDestroy(execGraph);
cudaGraphDestroy(graph);
```

---

## 3. Graph 创建方式一：Stream Capture

### 3.1 基本原理

Stream Capture 将 stream 置于**捕获模式**：所有提交到 stream 的操作**不会执行**，而是被记录为图节点：

```
正常模式:  stream ──→ GPU 执行
捕获模式:  stream ──→ 记录到 cudaGraph_t (不执行)
```

### 3.2 API

```c
// 开始捕获
cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode);

// 结束捕获, 取出 graph
cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph);
```

### 3.3 捕获模式 (Capture Mode)

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `cudaStreamCaptureModeGlobal` | 所有可能与该 stream 交互的操作都被捕获 | 最安全，默认选择 |
| `cudaStreamCaptureModeThreadLocal` | 仅捕获发起捕获的线程提交的操作 | 多线程程序 |
| `cudaStreamCaptureModeRelaxed` | 放松对 stream/线程的要求 | 高级用法 |

### 3.4 多 Stream 捕获

通过 `cudaEventRecord` + `cudaStreamWaitEvent` 可在捕获期间**分叉 / 合并**多个 stream，创建并行分支：

```cpp
cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

kernel_A<<<..., stream1>>>(...);       // stream1: A
cudaEventRecord(fork_event, stream1);  // 分叉点

cudaStreamWaitEvent(stream2, fork_event); // stream2 等待 A 完成
kernel_B<<<..., stream1>>>(...);       // stream1: A → B
kernel_C<<<..., stream2>>>(...);       // stream2: A → C (与 B 并行)

cudaEventRecord(join_event, stream2);  // 合并点
cudaStreamWaitEvent(stream1, join_event);
kernel_D<<<..., stream1>>>(...);       // stream1: B,C → D

cudaStreamEndCapture(stream1, &graph);
```

生成的图结构：

```
    A ──→ B ──┐
    │         ├──→ D
    └──→ C ──┘
```

### 3.5 捕获限制

Stream Capture 期间**不允许**的操作：

| 禁止操作 | 原因 |
|---------|------|
| `cudaMalloc` / `cudaFree` | 非异步操作，不在 stream 上 |
| `cudaStreamSynchronize` | 捕获模式下无执行，同步无意义 |
| `cudaMemcpy` (同步版) | 必须使用 `cudaMemcpyAsync` |
| 遗留 stream (`cudaStreamLegacy`) | 无法进入捕获模式 |

> **注意**: `cudaMallocAsync` / `cudaFreeAsync` 是允许的（CUDA 11.2+），会生成内存分配/释放节点。

### 3.6 优势与劣势

| 维度 | 优势 | 劣势 |
|------|------|------|
| 便利性 | 无需修改已有代码结构 | 不返回节点句柄 |
| 库兼容 | 可捕获第三方库调用 | 部分库不支持捕获 |
| 更新 | 可重新捕获后整图更新 | 更新开销比单节点更新大 |

---

## 4. Graph 创建方式二：Explicit Graph API

### 4.1 基本流程

手动创建图对象、添加节点、指定依赖：

```cpp
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// 添加 Kernel Node A (无前驱)
cudaGraphNode_t nodeA;
cudaKernelNodeParams paramsA = { ... };
cudaGraphAddKernelNode(&nodeA, graph, nullptr, 0, &paramsA);

// 添加 Kernel Node B (依赖 A)
cudaGraphNode_t nodeB;
cudaKernelNodeParams paramsB = { ... };
cudaGraphNode_t depsB[] = { nodeA };
cudaGraphAddKernelNode(&nodeB, graph, depsB, 1, &paramsB);

// 添加 Kernel Node C (依赖 A)
cudaGraphNode_t nodeC;
cudaKernelNodeParams paramsC = { ... };
cudaGraphNode_t depsC[] = { nodeA };
cudaGraphAddKernelNode(&nodeC, graph, depsC, 1, &paramsC);

// 添加 Kernel Node D (依赖 B 和 C)
cudaGraphNode_t nodeD;
cudaKernelNodeParams paramsD = { ... };
cudaGraphNode_t depsD[] = { nodeB, nodeC };
cudaGraphAddKernelNode(&nodeD, graph, depsD, 2, &paramsD);
```

### 4.2 优势与劣势

| 维度 | 优势 | 劣势 |
|------|------|------|
| 控制力 | 精确控制拓扑和参数 | 代码冗长 |
| 节点句柄 | 返回 `cudaGraphNode_t` 可用于后续更新 | 需要手动管理依赖 |
| 更新 | 可单节点更新，开销小 | — |

### 4.3 混合模式

可以使用 `cudaStreamBeginCaptureToGraph` 将 stream capture 的内容直接添加到已有的 graph 中，结合两种方式的优点。

---

## 5. Graph Node 类型全景

### 5.1 完整节点类型表

| 枚举值 | 节点类型 | 创建 API | 说明 | 引入版本 |
|--------|---------|---------|------|---------|
| 0 | **Kernel** | `cudaGraphAddKernelNode` | GPU kernel 执行 | CUDA 10.0 |
| 1 | **Memcpy** | `cudaGraphAddMemcpyNode` | 内存复制 | CUDA 10.0 |
| 2 | **Memset** | `cudaGraphAddMemsetNode` | 内存初始化 | CUDA 10.0 |
| 3 | **Host** | `cudaGraphAddHostNode` | CPU 回调函数 | CUDA 10.0 |
| 4 | **Child Graph** | `cudaGraphAddChildGraphNode` | 嵌入子图 | CUDA 10.0 |
| 5 | **Empty** | `cudaGraphAddEmptyNode` | 空操作 (同步点) | CUDA 10.0 |
| 6 | **Event Wait** | `cudaGraphAddEventWaitNode` | 等待 CUDA Event | CUDA 11.1 |
| 7 | **Event Record** | `cudaGraphAddEventRecordNode` | 记录 CUDA Event | CUDA 11.1 |
| 8 | **Ext. Semaphore Signal** | `cudaGraphAddExternalSemaphoresSignalNode` | 信号外部 Semaphore | CUDA 11.2 |
| 9 | **Ext. Semaphore Wait** | `cudaGraphAddExternalSemaphoresWaitNode` | 等待外部 Semaphore | CUDA 11.2 |
| 10 | **Mem Alloc** | `cudaGraphAddMemAllocNode` | 内存分配 | CUDA 11.4 |
| 11 | **Mem Free** | `cudaGraphAddMemFreeNode` | 内存释放 | CUDA 11.4 |
| 12 | **Batch MemOp** | `cuGraphAddBatchMemOpNode` | 批量内存操作 | CUDA 12.0 |
| 13 | **Conditional** | `cudaGraphAddNode` + `cudaGraphNodeTypeConditional` | 条件执行 (IF/WHILE/SWITCH) | CUDA 12.4 |

### 5.2 常用节点详解

#### Kernel Node

最常用节点，封装一次 kernel launch：

```c
cudaKernelNodeParams params = {};
params.func = (void *)myKernel;
params.gridDim = dim3(grid);
params.blockDim = dim3(block);
params.sharedMemBytes = smem;
params.kernelParams = args;  // void** 参数数组

cudaGraphAddKernelNode(&node, graph, deps, numDeps, &params);
```

#### Host Node

在 CPU 端执行回调函数（用于数据验证、日志等）：

```c
cudaHostNodeParams hostParams = {};
hostParams.fn = myHostFunction;
hostParams.userData = myData;

cudaGraphAddHostNode(&node, graph, deps, numDeps, &hostParams);
```

> **注意**: Host Node 执行在 CPU，会阻塞依赖它的下游 GPU 节点。

#### Child Graph Node

将整个子图嵌入为单个节点，实现模块化组合：

```c
cudaGraphAddChildGraphNode(&node, graph, deps, numDeps, childGraph);
```

#### Empty Node

不执行任何操作，用作**同步栅栏**或**依赖汇聚点**：

```
  K1 ──┐                K1 ──┐
  K2 ──┤──→ K4    →    K2 ──┤──→ [Empty] ──→ K4
  K3 ──┘                K3 ──┘
```

---

## 6. Graph 实例化与启动

### 6.1 实例化 API

```c
// CUDA 12+ 签名 (简化版, 取消了旧的 errNode/logBuffer 参数)
cudaError_t cudaGraphInstantiate(
    cudaGraphExec_t* pGraphExec,
    cudaGraph_t      graph,
    unsigned long long flags    // 默认 0
);
```

### 6.2 实例化标志 (Flags)

| Flag | 说明 | 引入版本 |
|------|------|---------|
| `0` (默认) | 标准实例化，仅支持 Host 启动 | CUDA 10.0 |
| `cudaGraphInstantiateFlagDeviceLaunch` | 允许从 Device 端启动 (见 §9) | CUDA 12.4 |
| `cudaGraphInstantiateFlagAutoFreeOnLaunch` | 重新 launch 时自动释放未 free 的图内分配 | CUDA 11.4 |
| `cudaGraphInstantiateFlagUseNodePriority` | 使用节点级优先级 | CUDA 12.0 |

### 6.3 启动

```c
// Host-side launch
cudaGraphLaunch(execGraph, stream);

// 异步执行, 提交后 CPU 立即返回
// 需要 cudaStreamSynchronize(stream) 等待完成
```

### 6.4 Constant-Time Launch (CUDA 12.6+)

对于**直线型图** (Straight-Line Graph，即无分支的线性依赖链)，CUDA 12.6 优化了 launch 机制，使 launch 时间与图中节点数量**无关**：

```
传统 launch:  O(N) — 逐个处理节点
Constant-time: O(1) — 预处理后一次提交整个命令缓冲

节点间延迟改善: ~60 ns/node
```

这意味着即使图包含数百个节点，launch 开销也保持恒定。

---

## 7. Graph 更新策略

### 7.1 两种更新机制

```
┌─────────────────────┐        ┌─────────────────────┐
│ Whole Graph Update   │        │ Individual Node      │
│ (整图更新)           │        │ Update (单节点更新)   │
│                     │        │                     │
│ cudaGraphExecUpdate │        │ cudaGraphExecKernel  │
│                     │        │ NodeSetParams 等     │
│ 适用:               │        │ 适用:               │
│ · 大量节点变更       │        │ · 少量参数变更       │
│ · 拓扑未知(库捕获)   │        │ · 持有节点句柄       │
│ · 便利              │        │ · 高效              │
└─────────────────────┘        └─────────────────────┘
```

### 7.2 Whole Graph Update (整图更新)

用一个新的 `cudaGraph_t` 更新已有的 `cudaGraphExec_t`：

```c
cudaGraphExecUpdateResultInfo updateResult;
cudaGraphExecUpdate(execGraph, newGraph, &updateResult);

if (updateResult.result == cudaGraphExecUpdateSuccess) {
    // 更新成功, 直接 launch
    cudaGraphLaunch(execGraph, stream);
} else {
    // 拓扑不兼容, 需要重新实例化
    cudaGraphExecDestroy(execGraph);
    cudaGraphInstantiate(&execGraph, newGraph, 0);
    cudaGraphLaunch(execGraph, stream);
}
```

**拓扑兼容条件**:
- 节点数量相同
- 节点类型和连接关系一致
- 仅参数不同

### 7.3 Individual Node Update (单节点更新)

直接修改已实例化图中的特定节点参数：

```c
// 更新 Kernel Node 参数
cudaKernelNodeParams newParams = { ... };
cudaGraphExecKernelNodeSetParams(execGraph, kernelNode, &newParams);

// 更新 Memcpy Node 参数
cudaMemcpy3DParms newMemcpyParams = { ... };
cudaGraphExecMemcpyNodeSetParams(execGraph, memcpyNode, &newMemcpyParams);

// 更新 Memset Node 参数
cudaMemsetParams newMemsetParams = { ... };
cudaGraphExecMemsetNodeSetParams(execGraph, memsetNode, &newMemsetParams);

// 更新 Host Node 参数
cudaHostNodeParams newHostParams = { ... };
cudaGraphExecHostNodeSetParams(execGraph, hostNode, &newHostParams);

// 更新 Child Graph Node
cudaGraphExecChildGraphNodeSetParams(execGraph, childNode, newChildGraph);
```

### 7.4 Update-then-Fallback 模式 (推荐)

```
首次: 捕获 → 实例化 → 执行
后续: 重新捕获 → cudaGraphExecUpdate() ──成功──→ 执行
                                      └──失败──→ 销毁 → 重新实例化 → 执行
```

优势：
1. 避免不必要的重新实例化 (~400 μs)
2. 无需了解图的内部结构，update 函数隐式做拓扑比较

### 7.5 Kernel Node 更新限制

- 不能改变 kernel 函数的 owning context
- 不能从非 CDP (CUDA Dynamic Parallelism) kernel 改为使用 CDP 的 kernel
- Enable/disable 状态不受更新影响

---

## 8. Conditional Nodes (CUDA 12.4+)

### 8.1 动机

传统 CUDA Graph 是**静态** DAG，不支持运行时分支或循环。若需条件执行，必须将图拆分并返回 CPU 做决策——这消除了 Graph 的核心优势。

Conditional Nodes 解决了这个问题：**在 GPU 端**根据条件变量决定是否执行子图，**无需返回 CPU**。

```
传统做法 (需要 CPU 参与):
  GPU: Graph_A ──→ CPU 判断 ──→ Graph_B 或 Graph_C
                  [CPU 开销]

Conditional Node (全 GPU):
  GPU: ┌── Kernel(设置条件) ──→ IF(cond) ──→ Body Graph ──┐
       │                                                    │
       └──────────────────── 后续节点 ←───────────────────┘
```

### 8.2 三种条件类型

| 类型 | 行为 | Body 数量 | 引入版本 |
|------|------|----------|---------|
| **IF** | 条件非零则执行 body[0]；若提供 body[1] 则条件为零时执行 body[1] (ELSE) | 1 或 2 | 12.4 (ELSE: 12.8) |
| **WHILE** | 条件非零则反复执行 body[0] | 1 | 12.4 |
| **SWITCH** | 条件值为 n 则执行 body[n]；超出范围则不执行 | ≥1 | 12.8 |

### 8.3 核心概念

#### Conditional Handle

条件变量的载体，通过 `cudaGraphConditionalHandleCreate` 创建：

```c
cudaGraphConditionalHandle handle;
cudaGraphConditionalHandleCreate(
    &handle,
    graph,
    defaultLaunchValue,  // 初始值 (每次图启动时重置)
    flags                // cudaGraphCondAssignDefault: 使用 defaultLaunchValue
);
```

#### 在 Device 端设置条件

```c
__device__ void cudaGraphSetConditional(
    cudaGraphConditionalHandle handle,
    unsigned int value
);
```

### 8.4 IF 条件节点

```cpp
// 1. 创建条件句柄
cudaGraphConditionalHandle handle;
cudaGraphConditionalHandleCreate(&handle, graph, 0, 0);

// 2. 上游 kernel 设置条件
__global__ void setCondition(cudaGraphConditionalHandle h, int* flag) {
    cudaGraphSetConditional(h, *flag != 0 ? 1 : 0);
}

// 3. 创建 IF 条件节点
cudaGraphNodeParams cParams = {};
cParams.type = cudaGraphNodeTypeConditional;
cParams.conditional.handle = handle;
cParams.conditional.type = cudaGraphCondTypeIf;
cParams.conditional.size = 1;  // 1 = 仅 IF; 2 = IF/ELSE
cParams.conditional.ctx = context;

cudaGraphNode_t condNode;
cudaGraph_t bodyGraph;
cudaGraphAddNode(&condNode, graph, &setKernelNode, 1, &cParams);
// bodyGraph 由 cParams.conditional.phGraph_out[0] 返回

// 4. 填充 body graph
cudaGraphNode_t bodyKernel;
cudaKernelNodeParams bodyParams = { ... };
cudaGraphAddKernelNode(&bodyKernel, bodyGraph, nullptr, 0, &bodyParams);
```

### 8.5 WHILE 条件节点

```cpp
__global__ void loopKernel(cudaGraphConditionalHandle handle, int* counter) {
    if (--(*counter) == 0) {
        cudaGraphSetConditional(handle, 0);  // 退出循环
    }
}

// 创建条件句柄, 默认值 1 (= 进入循环)
cudaGraphConditionalHandle handle;
cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault);

// 创建 WHILE 节点
cudaGraphNodeParams cParams = {};
cParams.type = cudaGraphNodeTypeConditional;
cParams.conditional.handle = handle;
cParams.conditional.type = cudaGraphCondTypeWhile;
cParams.conditional.size = 1;
```

执行流程：

```
开始 → 检查条件 ──[非零]──→ 执行 Body ──→ 检查条件 ──[非零]──→ ...
                                                    └──[零]──→ 继续
           └──[零]──→ 继续
```

### 8.6 SWITCH 条件节点 (CUDA 12.8+)

```cpp
// 创建 SWITCH 节点，3 个分支
cParams.conditional.type = cudaGraphCondTypeSwitch;
cParams.conditional.size = 3;  // body[0], body[1], body[2]

// 条件值 == 0 → 执行 body[0]
// 条件值 == 1 → 执行 body[1]
// 条件值 == 2 → 执行 body[2]
// 条件值 >= 3 → 不执行
```

### 8.7 Body Graph 限制

Body Graph 中**允许**的节点类型：
- Kernel Nodes
- Empty Nodes
- Child Graph Nodes
- Memset Nodes
- Memcpy Nodes
- Conditional Nodes (支持嵌套)

**不允许**: Host Nodes、Memory Alloc/Free Nodes、External Semaphore Nodes

所有 kernel 必须属于**同一 CUDA Context**。

---

## 9. Device-Side Graph Launch (CUDA 12.4+)

### 9.1 概述

Device-Side Graph Launch 允许**在 GPU kernel 内部**启动另一个 CUDA Graph，无需返回 CPU。

```
传统 Host Launch:
  CPU ──cudaGraphLaunch──→ GPU 执行 Graph

Device-Side Launch:
  GPU kernel ──cudaGraphLaunch──→ GPU 执行另一个 Graph
  (无 CPU 参与, 延迟更低)
```

### 9.2 两种 Device Launch 模式

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| **Fire-and-Forget** | 立即派发，与当前 graph 并行执行 | 独立子任务 |
| **Tail Launch** | 等待当前 graph (含所有 fire-and-forget) 完成后执行 | 顺序迭代 |

### 9.3 性能优势

- Device launch 延迟比 Host launch **低 2× 以上**
- 不受图结构复杂度影响 (Host launch 开销随图宽度增大)

### 9.4 启用 Device Launch

```c
cudaGraphExec_t execGraph;
cudaGraphInstantiate(&execGraph, graph, cudaGraphInstantiateFlagDeviceLaunch);
```

### 9.5 Tail Launch 自重启 (Self-Relaunch)

Tail Launch 特殊用法——图启动自身，实现**循环调度器**：

```
┌─────────────────────────────────────────┐
│ Scheduler Kernel:                        │
│   1. 检查工作队列                         │
│   2. Fire-and-forget 派发子图            │
│   3. Tail launch 自身 (自重启)            │
│      → 等待所有 fire-and-forget 完成      │
│      → 重新执行 Scheduler Kernel          │
└─────────────────────────────────────────┘
```

限制：
- 每个 device graph 同时只允许**一个** pending launch + **一个** self-relaunch
- 不能在 GPU 端使用 `cudaDeviceSynchronize` 等待 device launch 完成

### 9.6 Device-Launchable Graph 限制

- 所有节点必须在**同一设备**上
- 仅允许以下节点类型: Kernel、Empty、Memcpy、Memset、Child Graph、Conditional
- 不允许使用 CUDA Dynamic Parallelism (CDP) 或嵌套 Device Graph Launch
- 仅支持 device memory 和 pinned host memory 的 copy/memset

---

## 10. 内存管理节点

### 10.1 概述

CUDA 11.4 引入了 Graph 内的内存分配/释放节点，支持 **GPU 有序生命周期 (GPU-ordered lifetime)** 语义。

### 10.2 两种创建方式

| 方式 | API | 返回地址 |
|------|-----|---------|
| Stream Capture | 捕获 `cudaMallocAsync` / `cudaFreeAsync` | 捕获时返回 |
| Explicit API | `cudaGraphAddMemAllocNode` / `cudaGraphAddMemFreeNode` | `nodeParams.dptr` |

### 10.3 关键特性

- **地址固定**: 分配的虚拟地址在图的整个生命周期内**不变**，包括重复实例化和 launch
- **自动复用**: 生命周期不重叠的分配可**共享物理内存**
- **跨图复用**: CUDA 可跨多个 graph 复用物理内存

### 10.4 使用模式

```cpp
// 方式 1: Stream Capture
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    void* d_ptr;
    cudaMallocAsync(&d_ptr, size, stream);     // → Alloc Node
    myKernel<<<..., stream>>>(d_ptr, ...);     // → Kernel Node
    cudaFreeAsync(d_ptr, stream);              // → Free Node
cudaStreamEndCapture(stream, &graph);

// 方式 2: Explicit API
cudaMemAllocNodeParams allocParams = {};
allocParams.bytesize = size;
allocParams.poolProps.allocType = cudaMemAllocationTypePinned;
allocParams.poolProps.location.type = cudaMemLocationTypeDevice;
allocParams.poolProps.location.id = 0;

cudaGraphAddMemAllocNode(&allocNode, graph, nullptr, 0, &allocParams);
void* d_ptr = allocParams.dptr;  // 获取分配地址
```

### 10.5 限制

含内存节点的图有额外限制：
- 不能删除图的节点和边
- 同一时间只能存在**一个**实例化
- 图不能被 clone
- 仅当所有权转移给父图时才能作为 Child Graph Node

### 10.6 AutoFreeOnLaunch

使用 `cudaGraphInstantiateFlagAutoFreeOnLaunch` 标志实例化时，重新 launch 会自动释放未 free 的内存：

```
Producer Graph (包含 alloc 但不 free):
  Launch 1: alloc → compute → [未 free]
  Launch 2: [自动 free 上次的 alloc] → alloc → compute → [未 free]
  ...
```

> 注意：图销毁时仍需手动 free，`AutoFreeOnLaunch` 不改变销毁行为。

---

## 11. Edge Data 与 Programmatic Dependent Launch (CUDA 12.3+)

### 11.1 Edge Data 结构

CUDA 12.3 为图边引入了注解数据 (`cudaGraphEdgeData`)，由三部分组成：

| 字段 | 含义 | 默认值 |
|------|------|--------|
| `from_port` | 上游节点的**输出端口**，控制何时触发边 | 0 (完全完成) |
| `to_port` | 下游节点的**输入端口**，控制哪部分依赖 | 0 (整个节点) |
| `type` | 边类型，修饰端点关系 | 标准序列化 |

默认 (零初始化) 表示**标准完全序列化 + 内存可见性**。

### 11.2 Kernel Node 输出端口

仅 Kernel Node 定义了非零输出端口：

| 端口 | 说明 |
|------|------|
| `cudaGraphKernelNodePortDefault` | Kernel 执行完成后激活 |
| `cudaGraphKernelNodePortLaunchOrder` | 所有 block **开始执行**后激活 |
| `cudaGraphKernelNodePortProgrammatic` | 所有 block 调用 `cudaTriggerProgrammaticLaunchCompletion()` 或终止后激活 |

### 11.3 Programmatic Dependent Launch

通过 Edge Data 将上游 kernel 的 Programmatic 端口连接到下游 kernel：

```cpp
cudaGraphEdgeData edgeData = {};
edgeData.from_port = cudaGraphKernelNodePortProgrammatic;
edgeData.type = cudaGraphDependencyTypeProgrammatic;

cudaGraphAddDependencies(graph, &upstreamNode, &downstreamNode, &edgeData, 1);
```

下游 kernel 中调用 `cudaGridDependencySynchronize()` 等待上游信号。

应用场景：**kernel 重叠 (overlap)** — 上游 kernel 在部分 block 完成后就让下游开始执行。

```
传统依赖:    Kernel_A [==========] → Kernel_B [==========]
Programmatic: Kernel_A [======    ]
                       Kernel_B   [==========]
                                ↑ 上游触发后立即开始
```

### 11.4 相关 API

```c
// 带 Edge Data 的依赖管理
cudaGraphAddDependencies(graph, from[], to[], edgeData[], numDeps);
cudaGraphRemoveDependencies(graph, from[], to[], edgeData[], numDeps);
cudaGraphGetEdges(graph, from[], to[], edgeData[], &numEdges);
cudaGraphNodeGetDependencies(node, deps[], edgeData[], &numDeps);
cudaGraphNodeGetDependentNodes(node, deps[], edgeData[], &numDeps);
```

---

## 12. LLM 推理中的 CUDA Graph 实践

### 12.1 为什么 Decode 是 CUDA Graph 最佳场景

```
LLM 推理两阶段:

Prefill (处理 prompt):
  · 大量 token 并行 → 大 GEMM → GPU bound
  · kernel 执行时间长 → CPU 开销可忽略
  · 输入长度可变 → Graph 难以复用

Decode (逐 token 生成):
  · 每步仅 1 token/序列 → 小 GEMM → CPU bound
  · kernel 执行时间 2-10 μs → 数百次 launch 开销成瓶颈
  · batch size 固定 → 形状可预测 → Graph 完美适配
```

### 12.2 Graph Capture 策略

```
┌─────────────────────────────────────────────────────┐
│         LLM Inference CUDA Graph Strategy           │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Decode 阶段:                                       │
│    · 预热执行 → Stream Capture → 实例化              │
│    · 每个 batch_size 缓存一个 Graph                  │
│    · 请求 batch_size 不匹配时 padding 到最近的       │
│      预捕获尺寸                                      │
│                                                     │
│  Prefill 阶段:                                      │
│    · 通常不使用 Graph (形状变化太大)                  │
│    · 或使用 Piecewise Graph (排除 Attention)         │
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Attention 问题:                                     │
│    · Attention kernel 的 grid 维度取决于每序列的      │
│      query 分布，不仅仅是 total tokens              │
│    · 若捕获到 Graph，replay 时会 bake-in 错误的      │
│      launch config                                  │
│    · 解决: Piecewise Graph 将 Attention 排除在外     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 12.3 Graph Padding 策略

由于每个 batch size 需要独立的 Graph，为减少缓存数量，采用**padding**：

```
预捕获 batch sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256]

实际请求 batch_size = 5:
  → Padding 到 8
  → 使用 batch_size=8 的 Graph
  → 多余 3 个位置做空计算 (浪费小于 fallback 到 eager 模式)
```

### 12.4 vLLM 的 CUDA Graph 模式

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| `NONE` | 关闭 Graph | 调试 |
| `PIECEWISE` | Attention 保持 eager，其余 Graph 化 | 灵活，需 piecewise compilation |
| `FULL` | 全量 Graph (含 Attention) | 小模型、短 prompt |
| `FULL_DECODE_ONLY` | 仅 Decode 阶段全量 Graph | P/D 分离 Decode 实例 |
| `FULL_AND_PIECEWISE` | Decode 全量 + Prefill 分段 | **默认模式**，最高性能 |

### 12.5 TensorRT-LLM 的 Graph 策略

```
TRT-LLM 双策略:

1. Monolithic CUDA Graph (Decode):
   · 整个 forward pass 一次性捕获
   · 使用 torch.cuda.CUDAGraph 原生捕获
   · 包括 FlashAttention (Decode 时形状固定)

2. Piecewise CUDA Graph (Prefill/Mixed):
   · 通过 torch.compile fullgraph 模式实现
   · Attention 被切分为独立段，保持 eager
   · 自定义 op 需包装为 torch custom op
```

### 12.6 内存开销

每个 Graph 实例会占用额外 GPU 内存 (存储内部 command buffer、参数快照等)。多个 batch size 的 Graph 缓存会显著增加内存使用：

```
总内存开销 ≈ N_batch_sizes × 单个 Graph 内存
           ≈ 9 个预捕获尺寸 × ~50-200 MB/Graph (视模型大小)
```

在 GPU 内存紧张时，需要在 Graph 加速比和可用 KV Cache 之间权衡。

---

## 13. 性能数据与最佳实践

### 13.1 开销对比

```
┌─────────────────────────────────────────────────────┐
│            Launch 开销对比 (典型值)                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  单次 Kernel Launch (CPU 端)      3 – 7 μs          │
│  Kernel 间切换延迟               1 – 3 μs          │
│  Graph Launch (Host, 首次)       ~400 μs (含实例化)  │
│  Graph Launch (Host, 后续)       数 μs (整图一次)    │
│  Graph Launch (Device)           < Host 的 1/2      │
│  Graph 内节点间延迟 (优化后)       ~60 ns/node       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 13.2 最佳实践

#### 何时使用 CUDA Graph

| 场景 | 建议 | 原因 |
|------|------|------|
| LLM Decode | **强烈推荐** | CPU bound，短 kernel 序列重复执行 |
| 训练循环 (固定形状) | 推荐 | 每个 step 拓扑不变 |
| LLM Prefill | 谨慎 (Piecewise) | 形状可变，Attention 不兼容 |
| 单个大 GEMM | 不需要 | GPU bound，launch 开销可忽略 |
| 算法有大量分支 | 需配合 Conditional Nodes | 传统 Graph 不支持动态控制流 |

#### 实践要点

1. **预分配内存**: Graph 捕获期间避免动态分配（使用 `cudaMallocAsync` 或预分配 buffer）
2. **最小化 Graph 数量**: 每个 Graph 消耗内存，控制 batch size 级别数量
3. **Profile 对比**: 用 Nsight Systems 对比 Graph / 非 Graph 的 timeline
4. **避免 Graph 内 Host 同步**: Host Node 会阻塞 GPU 流水线
5. **Update 优于 Reinstantiate**: 拓扑不变时优先使用 `cudaGraphExecUpdate` 或单节点更新
6. **隔离非确定性操作**: RNG 等操作应在 Graph 外部或使用固定 seed

#### 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|---------|
| Graph 内地址失效 | 每次 launch 使用相同地址 | 使用 Graph 内存节点或固定 buffer |
| Attention 形状 bake-in | 捕获时形状固化 | Piecewise Graph 排除 Attention |
| 过多 Graph 缓存 | 内存爆炸 | 限制预捕获 batch size 数量 |
| 未预热即捕获 | 首次运行触发 JIT 编译被捕获 | 先 eager 预热再捕获 |
| 捕获期间同步调用 | `cudaStreamSynchronize` 导致捕获失败 | 使用异步 API |

---

## 14. API 速查表

### 14.1 Graph 生命周期

| API | 说明 |
|-----|------|
| `cudaGraphCreate` | 创建空图 |
| `cudaGraphDestroy` | 销毁图 |
| `cudaGraphInstantiate` | 实例化为可执行图 |
| `cudaGraphLaunch` | 在 stream 上启动可执行图 |
| `cudaGraphExecDestroy` | 销毁可执行图 |
| `cudaGraphClone` | 克隆图 |

### 14.2 Stream Capture

| API | 说明 |
|-----|------|
| `cudaStreamBeginCapture` | 开始 stream 捕获 |
| `cudaStreamEndCapture` | 结束捕获，返回 graph |
| `cudaStreamBeginCaptureToGraph` | 捕获到已有 graph |
| `cudaStreamIsCapturing` | 查询 stream 是否在捕获 |
| `cudaStreamGetCaptureInfo` | 获取捕获状态信息 |

### 14.3 Node 创建

| API | 节点类型 |
|-----|---------|
| `cudaGraphAddKernelNode` | Kernel |
| `cudaGraphAddMemcpyNode` | Memcpy |
| `cudaGraphAddMemsetNode` | Memset |
| `cudaGraphAddHostNode` | Host Callback |
| `cudaGraphAddChildGraphNode` | Child Graph |
| `cudaGraphAddEmptyNode` | Empty |
| `cudaGraphAddEventRecordNode` | Event Record |
| `cudaGraphAddEventWaitNode` | Event Wait |
| `cudaGraphAddExternalSemaphoresSignalNode` | Ext. Semaphore Signal |
| `cudaGraphAddExternalSemaphoresWaitNode` | Ext. Semaphore Wait |
| `cudaGraphAddMemAllocNode` | Memory Alloc |
| `cudaGraphAddMemFreeNode` | Memory Free |
| `cudaGraphAddNode` + `cudaGraphNodeTypeConditional` | Conditional |

### 14.4 Graph 更新

| API | 说明 |
|-----|------|
| `cudaGraphExecUpdate` | 整图更新 (拓扑不变时) |
| `cudaGraphExecKernelNodeSetParams` | 更新 Kernel 节点参数 |
| `cudaGraphExecMemcpyNodeSetParams` | 更新 Memcpy 节点参数 |
| `cudaGraphExecMemsetNodeSetParams` | 更新 Memset 节点参数 |
| `cudaGraphExecHostNodeSetParams` | 更新 Host 节点参数 |
| `cudaGraphExecChildGraphNodeSetParams` | 更新 Child Graph |
| `cudaGraphExecEventRecordNodeSetEvent` | 更新 Event Record 节点 |
| `cudaGraphExecEventWaitNodeSetEvent` | 更新 Event Wait 节点 |

### 14.5 图拓扑查询

| API | 说明 |
|-----|------|
| `cudaGraphGetNodes` | 获取图中所有节点 |
| `cudaGraphGetRootNodes` | 获取根节点 (无前驱) |
| `cudaGraphGetEdges` | 获取所有边 (含 Edge Data) |
| `cudaGraphNodeGetType` | 查询节点类型 |
| `cudaGraphNodeGetDependencies` | 查询节点的前驱 |
| `cudaGraphNodeGetDependentNodes` | 查询节点的后继 |
| `cudaGraphAddDependencies` | 添加依赖边 |
| `cudaGraphRemoveDependencies` | 删除依赖边 |

### 14.6 Conditional Node

| API | 说明 |
|-----|------|
| `cudaGraphConditionalHandleCreate` | 创建条件句柄 |
| `cudaGraphSetConditional` | (Device) 设置条件值 |

### 14.7 版本演进

| CUDA 版本 | 新增 Graph 功能 |
|-----------|----------------|
| 10.0 | CUDA Graph 初始引入 (Kernel/Memcpy/Memset/Host/Child/Empty) |
| 11.1 | Event Record / Wait Nodes |
| 11.2 | External Semaphore Nodes; `cudaMallocAsync` 可捕获 |
| 11.4 | Memory Alloc / Free Nodes; `AutoFreeOnLaunch` |
| 12.0 | Batch MemOp Nodes; 简化 `cudaGraphInstantiate` 签名 |
| 12.3 | Edge Data (`cudaGraphEdgeData`); Programmatic Dependent Launch |
| 12.4 | **Conditional Nodes** (IF/WHILE); **Device-Side Graph Launch** |
| 12.6 | **Constant-Time Launch** (直线型图优化) |
| 12.8 | Conditional IF/ELSE; **SWITCH** 节点; Blackwell 支持 |
