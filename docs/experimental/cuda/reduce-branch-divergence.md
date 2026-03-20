# 减少 Warp 分支分歧 (Reduce Branch Divergence) 深度指南

> 面向 LLM 高性能 Kernel 开发的 Warp Divergence 分析、优化策略与实战模式
> 覆盖 SIMT 分歧机制、硬件重汇合演进、Predication、循环展开、无分支编程、Warp 级统一控制流与 LLM Kernel 实战

---

## 目录

1. [Warp Divergence 概述](#1-warp-divergence-概述)
2. [硬件机制：分歧处理与重汇合](#2-硬件机制分歧处理与重汇合)
3. [Predication (谓词执行)](#3-predication-谓词执行)
4. [循环展开与分支消除](#4-循环展开与分支消除)
5. [无分支编程技术](#5-无分支编程技术)
6. [Warp 级统一控制流](#6-warp-级统一控制流)
7. [数据重组织与分歧消除](#7-数据重组织与分歧消除)
8. [Warp Shuffle 替代条件分支](#8-warp-shuffle-替代条件分支)
9. [LLM Kernel 实战：分歧优化模式](#9-llm-kernel-实战分歧优化模式)
10. [诊断与分析工具](#10-诊断与分析工具)
11. [优化检查清单](#11-优化检查清单)

---

## 1. Warp Divergence 概述

### 1.1 什么是 Warp Divergence

**Warp Divergence (分支分歧)** 是指同一 Warp 中 32 个线程在分支指令处走向不同路径，导致硬件必须**串行化**执行各路径。被 mask 掉的线程闲置等待，执行资源浪费。

```
无分歧 (Uniform):                 有分歧 (Divergent):
┌────────────────────────┐         ┌────────────────────────┐
│ Warp: 32 threads       │         │ Warp: 32 threads       │
│ if (cond) {            │         │ if (threadIdx.x < 16)  │
│   // 全部走同一路径    │         │   Path A  ← 16 threads │
│   Path A: 32/32 active │         │ else                   │
│ }                      │         │   Path B  ← 16 threads │
│ 执行时间: T            │         │ 执行时间: 2T           │
└────────────────────────┘         └────────────────────────┘
```

### 1.2 关键性质

| 性质 | 说明 |
|------|------|
| **Warp 内**限制 | 分歧仅影响同一 Warp 的线程；不同 Warp 独立执行互不干扰 |
| **串行化**执行 | 分歧路径被逐一执行，非活跃线程被 mask 掉 |
| **最坏情况** | N 条不同路径 → N× 串行化 (理论最坏 32 路 → 32× 减速) |
| **自动重汇合** | 硬件保证所有路径执行完后线程重新聚合，程序员无需手动干预 |
| **编译器优化** | 编译器对短分支自动使用 Predication，消除实际分支 |

### 1.3 性能影响量化

```
分歧惩罚 = 被串行化执行的额外路径所耗费的周期

示例: if-else 两路分歧，各路径 100 条指令
  无分歧: 100 instr × 1 路 = 100 instr
  有分歧: 100 instr × 2 路 = 200 instr (效率 50%)

示例: switch-case 4 路分歧，各路径 50 条指令
  全部走同一路: 50 instr
  4 路全分歧:   50 × 4 = 200 instr (效率 25%)
```

此外，分歧还有隐性开销：
- **分支管理开销**：BSSY/BSYNC 指令对 + 重汇合 mask 维护
- **指令缓存压力**：多路径代码增大 I-Cache working set
- **寄存器压力**：编译器需为所有路径分配寄存器

### 1.4 为什么 LLM Kernel 需要关注

| LLM Kernel 类型 | 典型分歧场景 | 严重程度 |
|----------------|------------|---------|
| **Softmax** | 边界检查、max/sum reduction 尾部处理 | 低—中 |
| **RMSNorm / LayerNorm** | reduction 尾部、尾部元素处理 | 低 |
| **量化/反量化** | 不同量化组的 scale 选择、packed 解码条件 | 中 |
| **Attention mask** | Causal mask 导致线程活跃/非活跃分化 | 中—高 |
| **Mixture of Experts (MoE)** | Token routing 到不同 Expert → 极端分歧 | 高 |
| **Speculative Decoding** | 验证阶段 accept/reject 分支 | 中—高 |
| **Top-k / Top-p Sampling** | 排序与阈值比较 | 中 |

---

## 2. 硬件机制：分歧处理与重汇合

### 2.1 Pre-Volta: SIMT Stack (SSY/SYNC)

Volta 之前 (Pascal 及更早)，整个 Warp 共享**单一 PC**，使用 **SIMT Stack** 管理分歧：

```
SIMT Stack 工作流程:
                                                SIMT Stack
1. 遇到分支前，推入 SSY (Set Sync):           ┌──────────────────┐
   SSY 记录重汇合点 (IPDom PC) + mask          │ Entry: RPC=0x120 │
                                               │   mask=0xFFFF... │
2. 分支发生，分成两路:                          ├──────────────────┤
   - fall-through (not-taken) → 先执行           │ Taken mask+PC    │
   - taken path → 压栈等待                       │ (等待执行)       │
                                               └──────────────────┘
3. fall-through 到达重汇合点 → pop 栈顶
4. 执行 taken path → 到达重汇合点 → pop
5. 所有路径完成 → SYNC 指令重汇合

限制:
- 线程在不同路径间**不能通信**
- 对称锁/barrier 可能**死锁** (同一 Warp 不同线程互相等待)
- 重汇合点固定为 IPDom → 可能延迟重汇合
```

### 2.2 Volta+: Convergence Barrier (BSSY/BSYNC)

从 Volta (CC 7.0) 起，每个线程拥有**独立 PC 和调用栈**，使用 **Convergence Barrier** 替代 SIMT Stack：

```
Convergence Barrier 机制 (SASS 指令):

BSSY Bx, target_pc    ─ 初始化 barrier register Bx
                         Bx ← 当前活跃线程的 mask (重汇合 mask)
                         target_pc 指向 BSYNC 指令地址

  ... 分歧代码 ...

BSYNC Bx              ─ 线程到达此处后阻塞
                         直到 Bx 记录的所有线程都到达

辅助指令:
YIELD                  ─ 切换到兄弟路径执行 (避免死锁)
BREAK @P, Bx           ─ 从重汇合 mask 中移除线程 (避免永不到达的死锁)
WARPSYNC mask          ─ 显式同步指定线程 (不依赖 Bx)
BMOV Bx, Rx            ─ 在 barrier 寄存器和通用寄存器间转移 mask
                         (嵌套分歧时保存/恢复)
```

### 2.3 Convergence Optimizer (调度优化器)

Volta+ 硬件中的 **Schedule Optimizer (Convergence Optimizer)** 动态将执行相同指令的活跃线程重新分组为 SIMT 单元：

```
Convergence Optimizer 的作用:

                ┌─ T0,T1,...T15 执行 Path A ──┐
  分歧发生 ──→  │                              │──→ Optimizer 重新分组
                └─ T16,...T31 执行 Path B ──┘      ↓
                                                最早可能的时机重汇合
                                                (不必等到 IPDom)

优势 vs Pre-Volta:
- Sub-warp 粒度重汇合 (不必全 32 线程)
- 允许分歧线程间通信 (__syncwarp)
- 编译器协助避免死锁 (YIELD 插入)
- 尽早重汇合 → 更高 SIMT 利用率
```

### 2.4 各架构分歧处理对比

| 特性 | Pre-Volta (≤ CC 6.x) | Volta/Turing (CC 7.x) | Ampere (CC 8.x) | Hopper (CC 9.0) |
|------|:---:|:---:|:---:|:---:|
| PC 模型 | Warp 共享 | 线程独立 | 线程独立 | 线程独立 |
| 分歧管理 | SIMT Stack (SSY/SYNC) | Convergence Barrier (BSSY/BSYNC) | BSSY/BSYNC | BSSY/BSYNC |
| 重汇合粒度 | Warp (32 线程) | Sub-warp | Sub-warp | Sub-warp |
| 死锁风险 | 高 (锁在同一 Warp) | 低 (YIELD + BREAK) | 低 | 低 |
| 分歧路径交织 | 否 (严格串行) | 是 (YIELD) | 是 | 是 |
| Warp 内同步 | 隐式 (不可靠) | `__syncwarp()` / WARPSYNC | 同左 | 同左 + ELECT |

> **关键结论**：虽然 Volta+ 的 Independent Thread Scheduling 使分歧处理更灵活、避免了死锁，但**分歧的性能惩罚本身并未消除**——被 mask 掉的线程仍然浪费执行周期。优化分歧依然是必要的。

---

## 3. Predication (谓词执行)

### 3.1 什么是 Predication

**Predication** 是编译器将短分支转化为**条件执行**的优化：所有线程都执行 if/else 两侧的指令，但通过谓词寄存器 (predicate register) 控制每条指令的结果是否写入。

```
源码:                                编译后 (Predication):
if (threadIdx.x < 16)                setp.lt.u32 %p, %tid, 16
    a = x + 1;                       @%p  add.f32 %a, %x, 1.0
else                                 @!%p add.f32 %a, %y, 2.0
    a = y + 2;
```

**没有实际分支指令**，没有 BSSY/BSYNC 开销，没有路径串行化。

### 3.2 PTX 谓词机制

```
PTX 谓词指令:

1. 声明谓词寄存器:
   .reg .pred %p, %q;

2. 设置谓词 (setp):
   setp.lt.s32 %p, %r1, %r2;     // %p = (%r1 < %r2)
   setp.eq.f32 %p, %f1, 0.0;     // %p = (%f1 == 0.0)
   setp.ne.and.s32 %p, %r1, 0, %q;  // %p = (%r1 != 0) AND %q

3. 条件执行 (任何指令前加 @%p 或 @!%p):
   @%p  mov.f32 %f1, %f2;         // 仅 %p=true 的线程写入
   @!%p add.f32 %f1, %f3, %f4;    // 仅 %p=false 的线程写入
   @%p  bra    LABEL;              // 条件分支 (仅需要时使用)

4. SASS 层面:
   PTX @%p → SASS @P0 (P0–P6 为硬件谓词寄存器)
   ISETP / FSETP / DSETP → 设置谓词
   @P0 FADD / @!P0 FMUL → 条件执行
```

### 3.3 编译器 Predication 策略

编译器 (ptxas) 自动决定使用 Predication 还是实际分支：

| 条件 | 编译器选择 | 原因 |
|------|----------|------|
| if body ≤ ~7 条指令 | **Predication** | 执行多余指令 < 分支管理开销 |
| if body > ~7 条指令 | **实际分支** (BRA + BSSY/BSYNC) | Predication 下太多无用指令 |
| if-else 两侧都短 | **双侧 Predication** | 全部执行，用 @%p/@!%p 选择 |
| 循环体内条件 | **倾向 Predication** | 循环 + 分支 = 双重惩罚 |
| 嵌套分支 | **实际分支** | Predication 无法高效嵌套 |

### 3.4 手动 Predication (Inline PTX)

当编译器未使用 Predication 而你确信应该使用时：

```cpp
// 编译器可能生成分支的代码:
float result;
if (mask & (1 << lane_id)) {
    result = value * scale;
} else {
    result = 0.0f;
}

// 手动 predication (inline PTX):
float result;
asm volatile(
    "{\n\t"
    "  .reg .pred p;\n\t"
    "  setp.ne.u32 p, %2, 0;\n\t"        // p = (bit != 0)
    "  @p  mul.f32 %0, %1, %3;\n\t"       // if p: result = value * scale
    "  @!p mov.f32 %0, 0f00000000;\n\t"    // else: result = 0.0
    "}\n\t"
    : "=f"(result)
    : "f"(value), "r"(mask & (1 << lane_id)), "f"(scale)
);
```

### 3.5 Predication 的权衡

```
Predication 权衡:

优势:
├── 消除分支管理开销 (BSSY/BSYNC/BRA)
├── 消除路径串行化 → 无分歧惩罚
├── 增加编译器指令调度自由度 (无控制流依赖)
└── 适合 if body 简短且 if/else 两侧计算量相近

劣势:
├── 两侧指令全部执行 (仅结果被 mask)
├── 功能单元被占用做"无用功"
├── 若一侧有昂贵操作 (如内存访问/SFU 调用)，浪费严重
└── 嵌套条件下指令膨胀严重

决策阈值 (近似):
  if body ≤ 7 instr  → Predication 更优
  if body > 7 instr  → 实际分支 + 分歧更优
  (编译器自动选择，极少需要手动干预)
```

---

## 4. 循环展开与分支消除

### 4.1 循环中的分歧来源

循环是 CUDA Kernel 中最常见的分歧来源之一：

```cpp
// 来源 1: 数据依赖的循环次数 → 不同线程迭代次数不同
for (int i = 0; i < data[threadIdx.x]; i++) {
    // 分歧: 每个线程循环次数不同
    // 先完成的线程被 mask 等待最慢的线程
}

// 来源 2: 循环内条件分支
for (int i = 0; i < N; i++) {
    if (arr[i + tid * N] > threshold) {  // 数据依赖条件
        // 分歧路径
    }
}

// 来源 3: 不均匀的 early break
for (int i = 0; i < N; i++) {
    if (found[tid]) break;  // 不同线程在不同时机 break
}
```

### 4.2 `#pragma unroll` 消除循环分歧

完全展开编译时已知次数的循环，消除循环条件分支：

```cpp
// 展开前: 每次迭代有循环条件判断 (i < 4?)
for (int i = 0; i < 4; i++) {
    sum += data[tid + i * stride];
}

// 展开后: 无循环条件 → 无条件分支
#pragma unroll
for (int i = 0; i < 4; i++) {
    sum += data[tid + i * stride];
}
// 编译为:
// sum += data[tid + 0 * stride];
// sum += data[tid + 1 * stride];
// sum += data[tid + 2 * stride];
// sum += data[tid + 3 * stride];
```

### 4.3 展开策略

| 场景 | 推荐 | 说明 |
|------|------|------|
| 编译时常量次数，≤ 32 次 | `#pragma unroll` (完全展开) | 消除所有循环开销 |
| 编译时常量次数，> 32 次 | `#pragma unroll 8` (部分展开) | 平衡 ILP 与代码膨胀 |
| 运行时变量次数 | `#pragma unroll` 无法完全展开；用模板参数固化为编译时常量 | 编译器可能部分展开但无法消除循环 |
| 循环体很大 (>20 行) | 慎用完全展开 | 指令缓存压力 + 寄存器膨胀 |

```cpp
// 模板参数固化循环次数 (运行时变为编译时):
template <int UNROLL_FACTOR>
__global__ void kernel(float* out, const float* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int idx = tid + i * gridDim.x * blockDim.x;
        if (idx < N) sum += in[idx];  // 边界检查仍然存在
    }
    out[tid] = sum;
}

// 调用时根据 N 选择实例化:
if (elements_per_thread == 4) kernel<4><<<grid, block>>>(...);
else if (elements_per_thread == 8) kernel<8><<<grid, block>>>(...);
```

### 4.4 循环展开的分歧消除原理

```
展开前 (PTX 含循环分支):
LOOP:
    ld.global.f32 %f1, [%rd1]
    add.f32 %f0, %f0, %f1
    add.s32 %r1, %r1, 1
    setp.lt.s32 %p, %r1, 4        ← 循环条件 (每次迭代检查)
    @%p bra LOOP                   ← 分支指令

展开后 (PTX 无循环分支):
    ld.global.f32 %f1, [%rd1]
    add.f32 %f0, %f0, %f1
    ld.global.f32 %f2, [%rd2]
    add.f32 %f0, %f0, %f2
    ld.global.f32 %f3, [%rd3]
    add.f32 %f0, %f0, %f3
    ld.global.f32 %f4, [%rd4]
    add.f32 %f0, %f0, %f4
    // 无任何分支指令
```

### 4.5 循环展开的注意事项

| 风险 | 说明 | 对策 |
|------|------|------|
| **寄存器膨胀** | 展开后更多变量同时 live → 寄存器需求增加 | 配合 `__launch_bounds__` 控制 |
| **指令缓存压力** | 展开后代码体积增大 → L0/L1 I-Cache miss | 部分展开 (`#pragma unroll 4`) |
| **编译时间** | 大循环完全展开 → 编译慢 | 部分展开或模板特化 |
| **Occupancy 下降** | 寄存器增加 → SM 能驻留的 Warp 减少 | 用 ncu 验证实际性能 |

---

## 5. 无分支编程技术

### 5.1 算术替代条件分支

用数学运算表达条件逻辑，彻底消除分支：

```cpp
// ---- 示例 1: 条件赋值 ----
// 有分支:
float result;
if (x > 0.0f) result = a;
else           result = b;

// 无分支 (用 ternary → 编译器通常生成 predication 或 FSEL):
float result = (x > 0.0f) ? a : b;

// ---- 示例 2: 条件累加 ----
// 有分支:
if (flag) sum += value;

// 无分支 (乘以 0/1):
sum += value * (float)flag;  // flag=0 → 加 0; flag=1 → 加 value

// ---- 示例 3: Clamp ----
// 有分支:
if (x < lo) x = lo;
if (x > hi) x = hi;

// 无分支:
x = fminf(fmaxf(x, lo), hi);  // fmin/fmax 编译为单条指令
```

### 5.2 内建函数替代分支

CUDA 提供的内建函数编译为**单条指令**，天然无分支：

| 操作 | 有分支写法 | 无分支替代 | SASS 指令 |
|------|----------|----------|----------|
| 取最小值 | `if (a < b) r = a; else r = b;` | `r = min(a, b)` / `fminf(a, b)` | `IMNMX` / `FMNMX` |
| 取最大值 | `if (a > b) r = a; else r = b;` | `r = max(a, b)` / `fmaxf(a, b)` | `IMNMX` / `FMNMX` |
| 绝对值 | `if (x < 0) x = -x;` | `x = abs(x)` / `fabsf(x)` | `FABS` (修饰符) |
| 条件选择 | `if (c) r = a; else r = b;` | 三目运算符 `r = c ? a : b` | `SEL` / `FSEL` |
| Clamp | 多个 if | `fminf(fmaxf(x, lo), hi)` | 2× `FMNMX` |
| 条件取负 | `if (sign) x = -x;` | `copysignf(fabsf(x), sign)` | `FABS` + `FSEL` |

### 5.3 位操作替代分支

```cpp
// ---- 条件清零 (无分支) ----
// 有分支: if (i < 0) i = 0;
// 无分支:
int mask = ~(i >> 31);   // i<0 → mask=0x00000000; i≥0 → mask=0xFFFFFFFF
i = i & mask;

// ---- 条件选择 (纯整数，无分支) ----
// 有分支: r = cond ? a : b;
// 无分支 (适用于 cond 为 0/1):
r = b ^ ((a ^ b) & -cond);   // cond=1 → r=a; cond=0 → r=b

// ---- 绝对值 (整数，无分支) ----
int sign = x >> 31;
int abs_x = (x ^ sign) - sign;
```

### 5.4 查找表替代 switch/case

`switch`/`case` 在线程走不同 case 时造成严重分歧。用查找表 (LUT) 替代：

```cpp
// ---- 有分歧: switch/case ----
float scale;
switch (quant_group_type) {
    case 0: scale = 1.0f; break;
    case 1: scale = 0.5f; break;
    case 2: scale = 0.25f; break;
    case 3: scale = 0.125f; break;
    default: scale = 1.0f; break;
}

// ---- 无分歧: Constant Memory LUT ----
__constant__ float SCALE_LUT[4] = {1.0f, 0.5f, 0.25f, 0.125f};

float scale = SCALE_LUT[quant_group_type];
// 如果所有线程查同一索引 → Constant Cache broadcast (1 cycle)
// 如果不同线程查不同索引 → 仍然比 switch 更快 (内存访问 vs 控制流串行化)

// ---- 无分歧: Shared Memory LUT (高频使用时) ----
__shared__ float scale_lut[4];
if (threadIdx.x < 4)
    scale_lut[threadIdx.x] = host_scale_table[threadIdx.x];
__syncthreads();
float scale = scale_lut[quant_group_type];
```

### 5.5 数学公式替代条件

```cpp
// ---- ReLU: max(0, x) ----
// 有分支:
if (x < 0.0f) x = 0.0f;
// 无分支:
x = fmaxf(x, 0.0f);

// ---- LeakyReLU: x > 0 ? x : alpha * x ----
// 有分支:
y = (x > 0.0f) ? x : alpha * x;
// 无分支 (利用 fmax + fmin):
y = fmaxf(x, 0.0f) + alpha * fminf(x, 0.0f);

// ---- 安全除法 (避免除零) ----
// 有分支:
float r = (denom != 0.0f) ? num / denom : 0.0f;
// 无分支 (添加极小值):
float r = num / (denom + 1e-12f);  // 避免条件检查

// ---- Causal mask (Attention) ----
// 有分支:
if (col > row) score = -INFINITY;
// 无分支:
float mask_val = (col > row) ? -INFINITY : 0.0f;  // 编译器 predicate
score += mask_val;
// 更好 (完全无分支):
float mask_val = __int_as_float(0xFF800000 & -(col > row));  // -INF if true
score += mask_val;
```

---

## 6. Warp 级统一控制流

### 6.1 原则：将分歧条件提升到 Warp/Block 边界

分歧的根源是**同一 Warp 内**线程走不同路径。如果能将条件分支对齐到 Warp 边界，每个 Warp 内的线程一致地走同一路径，则完全消除分歧：

```
分歧条件在线程级:              分歧条件在 Warp 级:
┌─ Warp 0 ─┐                 ┌─ Warp 0 ─┐
│ T0: if   │ ← 分歧!         │ 全部 Path A│ ← 无分歧
│ T1: else │                 │ (warpId=0) │
│ ...      │                 └───────────┘
└──────────┘                 ┌─ Warp 1 ─┐
                             │ 全部 Path B│ ← 无分歧
                             │ (warpId=1) │
                             └───────────┘
```

### 6.2 Warp 对齐分支

```cpp
// ---- 按 warpId 分支 (无 Warp 内分歧) ----
int warp_id = threadIdx.x / 32;  // 或 threadIdx.x >> 5

if (warp_id < 2) {
    // Warp 0 和 Warp 1 的所有线程走此路径
    compute_path_A();
} else {
    // Warp 2+ 的所有线程走此路径
    compute_path_B();
}
// ← 无 Warp 内分歧 (分歧仅在 Warp 间，不影响性能)

// ---- Block 级分支 (完全无分歧) ----
if (blockIdx.x % 2 == 0) {
    // 偶数 Block 走此路径
    process_even();
} else {
    process_odd();
}
```

### 6.3 Warp 级投票 (Warp Vote)

使用 Warp Vote 函数让所有线程统一决策：

```cpp
// __all_sync(mask, predicate): 所有线程的 predicate 都为 true → true
// __any_sync(mask, predicate): 任一线程的 predicate 为 true → true
// __ballot_sync(mask, predicate): 返回每线程 predicate 的 32-bit mask

// ---- 示例: 统一 early exit ----
// 有分歧的 early exit:
if (my_value == 0) return;  // 某些线程退出，其余继续

// 无分歧的 early exit (全 Warp 统一决策):
bool all_zero = __all_sync(0xFFFFFFFF, my_value == 0);
if (all_zero) return;  // 全部为 0 才退出，否则全部继续
// → Warp 内所有线程走同一路径

// ---- 示例: 统一跳过不必要的计算 ----
bool any_needs_work = __any_sync(0xFFFFFFFF, has_work[tid]);
if (!any_needs_work) {
    // 整个 Warp 跳过此阶段
    goto next_stage;
}
// 全部执行 (即使某些线程不需要，也比分歧更高效)
do_work();  // 不需要的线程做的是"浪费但无害"的计算
```

### 6.4 Warp 统一条件 (Uniform Branch)

如果分支条件对 Warp 内所有线程相同 (**Uniform Condition**)，则不会发生分歧：

```cpp
// ---- Uniform 条件 (无分歧) ----
// 1. 常量 / 编译时已知:
if (BLOCK_SIZE > 128) { ... }         // constexpr → 编译时消除

// 2. 基于 blockIdx (Block 内所有线程相同):
if (blockIdx.x < num_full_blocks) { ... }

// 3. 基于 Shared Memory 变量 (Block 内所有线程可见):
__shared__ int iteration_count;
if (threadIdx.x == 0) iteration_count = compute_count();
__syncthreads();
for (int i = 0; i < iteration_count; i++) { ... }  // 所有线程迭代相同次数

// 4. Warp 级 reduction 后的值 (需使用 butterfly 或广播使所有线程拿到结果):
float warp_max = warp_reduce_max_broadcast(my_val);  // 用 __shfl_xor_sync 或末尾 __shfl_sync 广播
if (warp_max > threshold) { ... }                     // Uniform → 无分歧
// 注意: __shfl_down_sync 仅 lane 0 正确，不能直接用于此场景

// ---- Non-Uniform 条件 (有分歧) ----
if (data[threadIdx.x] > threshold) { ... }  // 每线程不同 → 分歧
if (threadIdx.x % 3 == 0) { ... }           // 不对齐 Warp → 分歧
```

### 6.5 Grid Stride Loop 减少边界分歧

```cpp
// ---- 传统: early exit 边界检查 ----
__global__ void kernel_v1(float* out, const float* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;  // 最后一个 Warp 可能分歧
    out[tid] = in[tid] * 2.0f;
}

// ---- Grid Stride Loop: 分歧仅在最后一次迭代 ----
__global__ void kernel_v2(float* out, const float* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        out[i] = in[i] * 2.0f;
    }
    // 所有迭代中线程保持统一活跃
    // 仅最后一轮可能有线程因 i >= N 退出
}
```

---

## 7. 数据重组织与分歧消除

### 7.1 按处理类型排序数据

当不同数据需要不同处理逻辑时，排序使相同类型的数据落在同一 Warp 中：

```cpp
// ---- 未排序: 同一 Warp 中混杂不同类型 → 分歧 ----
// tokens: [type_A, type_B, type_A, type_B, type_A, ...]
//          ^ Warp 0 内 A/B 交替 → 每个 Warp 都分歧

// ---- 排序后: 同类型数据连续 → 无分歧 ----
// tokens: [type_A, type_A, ..., type_A, type_B, type_B, ..., type_B]
//          ^ Warp 0: 全 A                ^ Warp K: 全 B

// 实现:
// 1. 预处理阶段按类型排序 (或用 radix sort 的前几 bit)
// 2. 记录原始索引用于最后散射回结果
```

### 7.2 Padding 对齐到 Warp 大小

```cpp
// ---- 问题: 不规则长度导致尾部 Warp 分歧 ----
// 序列长度: [100, 37, 200, 15, ...]
// 处理 seq_len=37 时，Warp 1 (threads 32–63) 中 threads 37–63 被 mask

// ---- 解决: Padding 到 32 的倍数 ----
int padded_len = (seq_len + 31) & ~31;  // 向上取整到 32 倍数
// 37 → 64, 100 → 128, 15 → 32
// Padding 元素用 0 或 identity 值填充

// ---- 或者: 用 mask 替代条件分支 ----
// 不用 if (tid < seq_len) 来跳过，而是:
float val = (tid < seq_len) ? input[tid] : 0.0f;  // predication
output[tid] = val * scale;  // 所有线程统一执行，padding 值不影响结果
```

### 7.3 MoE (Mixture of Experts) 分歧问题

MoE 是 LLM 中最严重的分歧场景——不同 Token 路由到不同 Expert：

```
原始执行 (极端分歧):
Warp 0: [Token→E0, Token→E3, Token→E1, Token→E0, ...]
         每个线程执行不同 Expert → 最多 N_expert 路分歧

优化策略:
1. Token Regrouping (Token 重排):
   ┌─ Warp 0: 全部执行 Expert 0 ─┐
   │  Warp 1: 全部执行 Expert 1  │  按目标 Expert 排序 token
   │  Warp 2: 全部执行 Expert 2  │  每个 Warp 只执行一个 Expert
   └─ ...                        ─┘  → 无分歧

2. Batch-per-Expert (每个 Expert 独立 Kernel Launch):
   for each expert:
       kernel<<<grid, block>>>(tokens_for_this_expert, expert_weights);
   // 完全无分歧，但 launch overhead + 并行度可能不足

3. Grouped GEMM (CUTLASS):
   将每个 Expert 视为一个 grouped GEMM problem
   用 CUTLASS grouped GEMM 自动处理不同大小的矩阵
```

---

## 8. Warp Shuffle 替代条件分支

### 8.1 Reduction 中的分歧消除

传统 tree reduction 中，每一轮只有一半线程参与，其余被 mask → 分歧：

```cpp
// ---- 有分歧的 reduction ----
__shared__ float sdata[256];
sdata[tid] = value;
__syncthreads();

for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {                      // ← 每轮只有一半线程活跃 → 分歧
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

```cpp
// ---- 无分歧的 Warp Shuffle reduction ----
float val = value;

// Warp 内 reduction: 全部 32 线程参与，无分歧
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
}
// lane 0 持有 warp-level sum
// 无条件分支，无 shared memory，无 __syncthreads
```

### 8.2 Shuffle Reduction 可视化

```
__shfl_down_sync 的工作方式 (以 8 线程简化):

Step 1: offset = 4
  T0  T1  T2  T3  T4  T5  T6  T7
  ↓   ↓   ↓   ↓
  +   +   +   +
  T4  T5  T6  T7
  ─────────────────────────────────
  T0  T1  T2  T3  (累加了 T4-T7)

Step 2: offset = 2
  T0  T1  T2  T3
  ↓   ↓
  +   +
  T2  T3
  ─────────────────────────────────
  T0  T1  (累加了 T2-T7)

Step 3: offset = 1
  T0  T1
  ↓
  +
  T1
  ─────────────────────────────────
  T0  (累加了 T0-T7 = 全 warp sum)

每一步所有线程都执行 __shfl_down_sync (无分歧!)
不需要的线程的结果自然被忽略
```

### 8.3 完整的 Block-Level Reduction (无分歧)

```cpp
// Warp reduce (无分歧):
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Block reduce (最小化分歧):
__device__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[32];  // 最多 32 个 warp

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    val = warp_reduce_sum(val);  // Step 1: warp 内 reduce (无分歧)

    if (lane == 0)               // Step 2: lane 0 写入 shared memory
        warp_sums[warp] = val;   // (仅此处有分歧，但仅 1 条指令 → predication)
    __syncthreads();

    // Step 3: 第一个 warp reduce 所有 warp 的结果
    int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.0f;
    if (warp == 0)
        val = warp_reduce_sum(val);  // (warp 0 执行，其余 warp idle → 跨 warp 无惩罚)

    return val;
}
```

### 8.4 广播替代条件赋值

```cpp
// ---- 有分歧: 只有 lane 0 计算，然后广播 ----
float result;
if (lane_id == 0) {        // ← 分歧: 1 线程 vs 31 线程
    result = expensive_compute();
}
result = __shfl_sync(0xFFFFFFFF, result, 0);  // 从 lane 0 广播

// ---- 更优: 所有线程计算同一值 (如果可行) ----
float result = cheap_compute(shared_input);  // 无分歧
// 如果计算便宜，让 32 线程全做比 1 线程做 + 广播更高效
```

---

## 9. LLM Kernel 实战：分歧优化模式

### 9.1 Softmax Kernel

```cpp
// Softmax 中的分歧点与优化:

// 1. Max reduction (见 Section 8 的 warp shuffle reduction)
float thread_max = -INFINITY;
for (int i = tid; i < seq_len; i += blockDim.x)
    thread_max = fmaxf(thread_max, input[i]);
float row_max = block_reduce_max(thread_max);

// 2. Exp + Sum reduction
float thread_sum = 0.0f;
for (int i = tid; i < seq_len; i += blockDim.x) {
    float val = __expf(input[i] - row_max);  // __expf → fast math, 无分支
    output[i] = val;
    thread_sum += val;
}
float row_sum = block_reduce_sum(thread_sum);

// 3. Normalize — 边界处理 (潜在分歧点)
// 有分歧的写法 (padded 循环 + 内部边界检查):
int padded = (seq_len + blockDim.x - 1) / blockDim.x * blockDim.x;
for (int i = tid; i < padded; i += blockDim.x) {
    if (i < seq_len)       // ← 最后一轮 Warp 内有些线程 i < seq_len，有些 i >= seq_len → 分歧
        output[i] /= row_sum;
}

// 无分歧的写法 (用循环条件自然截断 + 倒数预计算):
float inv_sum = __fdividef(1.0f, row_sum);  // 先计算倒数 (所有线程相同)
for (int i = tid; i < seq_len; i += blockDim.x) {
    output[i] *= inv_sum;  // 乘法比除法更快; 循环边界是唯一分歧点 (仅最后一轮)
}
```

### 9.2 RMSNorm Kernel

```cpp
// RMSNorm: y = x * w / sqrt(mean(x^2) + eps)

__global__ void rmsnorm_kernel(float* out, const float* in,
                                const float* weight, int hidden_dim) {
    int tid = threadIdx.x;
    float thread_ss = 0.0f;

    // Step 1: Sum of squares (向量化 + Grid stride)
    #pragma unroll 4
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = in[i];
        thread_ss += val * val;
    }

    // Step 2: Block reduce (无分歧 shuffle reduction)
    float ss = block_reduce_sum(thread_ss);

    // Step 3: 计算 rsqrt (所有线程统一 → Uniform 分支)
    __shared__ float s_rsqrt;
    if (tid == 0)   // 仅 1 条指令 → predication
        s_rsqrt = rsqrtf(ss / hidden_dim + 1e-6f);
    __syncthreads();
    float scale = s_rsqrt;  // 所有线程读取相同值

    // Step 4: Normalize (无分歧)
    #pragma unroll 4
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        out[i] = in[i] * scale * weight[i];
    }
}
```

### 9.3 Causal Attention Mask

```cpp
// Causal Mask: score[i][j] = (j <= i) ? score[i][j] : -INF

// 有分歧:
if (col > row) {
    score = -INFINITY;      // ← 同一 Warp 处理不同列时分歧
}

// 无分歧方案 1: 算术 mask
float mask = (col <= row) ? 0.0f : -INFINITY;  // predication
score += mask;

// 无分歧方案 2: 完全用乘法
float mask = (float)(col <= row);  // 0.0f or 1.0f, predication
score = score * mask + (1.0f - mask) * (-1e9f);

// 无分歧方案 3: Warp 对齐 (最优)
// 如果 tile 宽度 = 32 (一个 Warp 处理一行的 32 列):
// 对于 row >= tile_col_start + 31 的行，整个 Warp 都在 mask 内 → 无分歧
// 对于 row < tile_col_start 的行，整个 Warp 都被 mask → 无分歧
// 仅 row 在 tile 列范围内的行有 Warp 内分歧 → 用 predication 处理
```

### 9.4 量化/反量化 Kernel

```cpp
// INT8 Dequantization with per-group scale

// 有分歧: 条件检查 group 边界
if (elem_idx % group_size == 0) {
    scale = scales[group_idx];  // 仅 group 第一个元素加载 scale
}

// 无分歧: 每个线程独立计算自己的 group 和 scale
int group_idx = elem_idx / group_size;       // 整数除法，无分支
float scale = scales[group_idx];              // 每线程独立加载
float dequant = (float)qdata[elem_idx] * scale;

// 更优: 如果 group_size ≥ 32 且对齐，同一 Warp 中所有线程
// 属于同一 group → scale 加载被 L1 cache 合并
```

### 9.5 Top-k Sampling

```cpp
// Top-k: 从 logits 中选最大的 k 个

// 高分歧做法: 线程级排序 + 条件交换
// → 不同线程在不同位置交换 → 分歧严重

// 低分歧做法: Bitonic Sort (Warp 级)
// Bitonic sort 的每一步，同一 Warp 中所有线程做相同方向的比较
// → 比较方向由 (tid ^ stride) 决定，是确定性的

// 最低分歧: Radix-based Top-k
// 从最高位开始，统计 bit=1 的元素个数
// 如果 count >= k，保留 bit=1 的元素，否则保留全部
// 每步用 __ballot_sync 统计 → 无分歧
int bit_count = __popc(__ballot_sync(0xFFFFFFFF, (value >> bit) & 1));
```

---

## 10. 诊断与分析工具

### 10.1 Nsight Compute (ncu) 分歧指标

| 指标 | 说明 | 理想值 |
|------|------|--------|
| `smsp__sass_average_branch_targets_threads_uniform.pct` | 分支目标线程统一百分比 | **100%** |
| `smsp__branch_targets_diverged` | 分歧分支目标数 (含 fallthrough) | **0** |
| `smsp__sass_thread_inst_executed_op_control_pred_on.sum` | 谓词为 true 时执行的控制指令数 | 越低越好 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Warp 活跃率 | 高 |
| `smsp__inst_executed_pipe_control.sum` | 控制流指令总数 | 越低越好 |

```bash
# 收集分歧指标:
ncu --metrics \
    smsp__sass_average_branch_targets_threads_uniform.pct,\
    smsp__branch_targets_diverged,\
    smsp__inst_executed_pipe_control.sum \
    ./my_kernel

# Source-level 分歧分析:
ncu --section SourceCounters \
    --source-folders /path/to/source \
    ./my_kernel
```

### 10.2 SASS 反汇编检查

```bash
# 查看编译后的 SASS，检查 BRA/BSSY/BSYNC 指令:
cuobjdump -sass ./my_kernel.o | grep -E "BRA|BSSY|BSYNC|WARPSYNC|@P"

# 示例输出:
#   /*0050*/  @P0 FADD R4, R4, R5;         ← Predication (好: 无分歧)
#   /*0060*/  BSSY B0, `(.L_2);            ← 设置重汇合 barrier (有分支)
#   /*0070*/  @P1 BRA `(.L_1);             ← 条件分支 (可能分歧)
#   /*0080*/  ...
#   /*00a0*/  BSYNC B0;                    ← 重汇合点

# 目标: @P 修饰的普通指令 (predication) > BRA 指令 (分支)
```

### 10.3 PTX 检查

```bash
# 生成 PTX 检查分支模式:
nvcc -ptx -arch=sm_90 kernel.cu -o kernel.ptx

# 查找条件分支 vs predication:
grep -c "@%p bra" kernel.ptx    # 条件分支数
grep -c "@%p" kernel.ptx        # 谓词执行数 (含分支)
grep -c "setp" kernel.ptx       # 谓词设置数
```

### 10.4 分歧定位流程

```
分歧诊断流程图:

1. ncu 收集 branch_targets_diverged
   ├── = 0 → 无分歧问题 ✓
   └── > 0 → 继续排查
       │
2. ncu SourceCounters 定位分歧热点 (行号级)
   │
3. cuobjdump -sass 查看具体指令
   ├── BRA 条件分支 → 检查条件是否可对齐到 Warp
   ├── 循环条件分支 → 考虑 unroll
   └── BSSY/BSYNC 对 → 检查分支体大小
       ├── 小 (<7 instr) → 考虑手动 predication
       └── 大 → 考虑数据重排或算法重构
   │
4. 性能影响评估
   ├── control pipe 指令占比 > 10% → 优化分支
   ├── warp 活跃率低 → 分歧或 occupancy 问题
   └── 对比无分歧版本的 throughput 差异
```

---

## 11. 优化检查清单

### 11.1 编码阶段

| 检查项 | 优先级 | 说明 |
|--------|:------:|------|
| 短 if/else 使用 ternary 或 `fmin/fmax` | **高** | 让编译器生成 predication |
| `switch/case` 替换为 LUT | **高** | 查找表天然无分歧 |
| Reduction 使用 `__shfl_down_sync` | **高** | 替代 Shared Memory tree reduction |
| 循环次数固化为编译时常量 | **高** | 启用完全展开，消除循环分支 |
| 边界检查放在循环条件中 (Grid Stride) | **中** | 减少 early exit 分歧 |
| 分支条件对齐到 Warp 边界 | **中** | `warp_id` 而非 `threadIdx.x` |
| 数据按处理类型排序 | **中** | 同类型数据落在同一 Warp |
| 用 `__all_sync/__any_sync` 统一决策 | **中** | Warp 级一致的 early exit |
| 避免 Warp 内线程有不同循环次数 | **中** | Padding 到 Warp 对齐 |
| Causal mask 用算术而非 if | **低** | `score += mask_val` 替代条件赋值 |

### 11.2 验证阶段

| 检查项 | 工具 | 目标 |
|--------|------|------|
| `branch_targets_diverged` = 0 | ncu | 无分歧 |
| `branch_targets_threads_uniform` > 95% | ncu | 接近 100% |
| SASS 中 BRA 数量 vs @P 数量 | cuobjdump | @P 远多于 BRA |
| 控制流指令占比 | ncu | < 5% |
| 对比有/无分歧版本吞吐量 | ncu / nsys | 量化优化收益 |

### 11.3 常见错误

| 错误 | 说明 | 修正 |
|------|------|------|
| `if (threadIdx.x % N)` (N ≠ 32 的倍数) | 不对齐 Warp → 分歧 | 改为 `warp_id` 级条件 |
| 循环内 `if (data[tid] > thresh) break` | 不同线程不同时机退出 | 用 `__all_sync` 统一退出 |
| 多层嵌套 if-else | 路径指数增长 | 重构为查找表或数学公式 |
| 忽略尾部 Warp 分歧 | 小 N 时比例显著 | Padding 或 Grid Stride |
| 过度手动优化简单分支 | 编译器已自动 predication | 先用 SASS 验证再优化 |

### 11.4 决策流程图

```
遇到条件分支时的决策:

                        条件分支
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
       Uniform?      body ≤ 7 instr?   body > 7 instr?
       (Warp 内        (短分支)         (长分支)
        所有线程                │              │
        条件相同)              ▼              ▼
            │          编译器自动        考虑以下策略:
            ▼          predication      ├── 数据重排对齐 Warp
       无分歧 ✓        (无需优化) ✓     ├── 查找表替代 switch
                                       ├── 算术替代条件
                                       ├── Warp Vote 统一决策
                                       └── 提升条件到 Warp/Block 级
```

---

*参考资料: NVIDIA CUDA Programming Guide (Warps and SIMT), NVIDIA PTX ISA, Nsight Compute Profiling Guide, "Control Flow Management in Modern GPUs" (arXiv:2407.02944)*
