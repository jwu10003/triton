# Shared Memory Bank Conflict 与无冲突访问技术

> 面向 LLM 高性能 Kernel 开发的 Shared Memory Bank Conflict 深度解析与解决方案
> 覆盖 Padding、XOR Swizzle、CuTe Swizzle、TMA Swizzle 四种核心技术

---

## 目录

1. [Shared Memory Bank 基础](#1-shared-memory-bank-基础)
2. [Bank Conflict 原理](#2-bank-conflict-原理)
3. [常见冲突场景分析](#3-常见冲突场景分析)
4. [解决方案一：Padding](#4-解决方案一padding)
5. [解决方案二：XOR Swizzle](#5-解决方案二xor-swizzle)
6. [解决方案三：CuTe Swizzle 抽象](#6-解决方案三cute-swizzle-抽象)
7. [解决方案四：TMA Swizzle (Hopper+)](#7-解决方案四tma-swizzle-hopper)
8. [ldmatrix 与 128-bit 访问的 Bank Conflict](#8-ldmatrix-与-128-bit-访问的-bank-conflict)
9. [LLM Kernel 实战模式](#9-llm-kernel-实战模式)
10. [性能分析与调试](#10-性能分析与调试)
11. [方案对比与选型指南](#11-方案对比与选型指南)

---

## 1. Shared Memory Bank 基础

### 1.1 物理组织

Shared Memory 是 GPU SM 上的片上 SRAM，被划分为 **32 个 Bank**，每个 Bank 宽度 **4 字节 (32 bit)**，每个时钟周期可独立服务一次 4 字节读写。

```
Bank:    0     1     2     3    ...   30    31
        ┌─────┬─────┬─────┬─────┬───┬─────┬─────┐
Addr 0: │ 0-3 │ 4-7 │8-11 │12-15│...│120- │124- │  ← 第 0 行 (128 Bytes)
        │     │     │     │     │   │ 123 │ 127 │
        ├─────┼─────┼─────┼─────┼───┼─────┼─────┤
Addr128:│128- │132- │136- │140- │...│248- │252- │  ← 第 1 行 (128 Bytes)
        │ 131 │ 135 │ 139 │ 143 │   │ 251 │ 255 │
        ├─────┼─────┼─────┼─────┼───┼─────┼─────┤
        │ ... │ ... │ ... │ ... │...│ ... │ ... │
        └─────┴─────┴─────┴─────┴───┴─────┴─────┘
```

### 1.2 地址到 Bank 的映射公式

```
bank_id = (byte_address / 4) % 32
```

即连续的 4 字节 word 依次映射到 Bank 0, 1, 2, ..., 31, 0, 1, 2, ...

**推论：** 地址相差 128 字节 (= 32 × 4) 的整数倍的 word 落在同一个 Bank。

### 1.3 关键设计约束

- **Bank 数量 = Warp 大小 = 32**：这不是巧合，而是刻意设计，使得理想情况下每个线程恰好访问一个独立 Bank
- **Bank 宽度固定 4 字节** (SM 5.0+)：SM 3.x 可配置为 8 字节 (`cudaDeviceSetSharedMemConfig`)，但从 SM 5.0 起固定为 4 字节
- **Bank Conflict 仅发生在同一 Warp 内**：不同 Warp 的线程访问同一 Bank 不会冲突

---

## 2. Bank Conflict 原理

### 2.1 定义

当同一 Warp 内的**两个或多个线程**在同一条 Shared Memory 指令中访问**同一 Bank 的不同地址**时，发生 Bank Conflict。硬件将冲突的访问**串行化**为多次无冲突事务。

```
✅ 无冲突：32 个线程各访问不同 Bank（1 次事务）
✅ 广播：  多个线程访问同一 Bank 的 *相同地址*（1 次事务，硬件 multicast）
❌ N-way: N 个线程访问同一 Bank 的 *不同地址*（N 次事务，带宽降为 1/N）
```

### 2.2 冲突度 (Conflict Degree)

| 冲突度 | 含义 | 性能影响 |
|--------|------|---------|
| 1-way | 无冲突 | 1× (最优) |
| 2-way | 2 个线程命中同一 Bank | 2× 延迟 |
| 4-way | 4 个线程命中同一 Bank | 4× 延迟 |
| 8-way | 8 个线程命中同一 Bank | 8× 延迟 |
| 16-way | 16 个线程命中同一 Bank | 16× 延迟 |
| 32-way | 所有线程命中同一 Bank | 32× 延迟 (最差) |

### 2.3 两个特殊规则

**规则 1 — 广播 (Broadcast / Multicast)：**

多个线程读取**完全相同的地址**不会冲突，硬件直接广播数据到所有请求线程。SM 2.0+ 支持任意多播 (multicast)——只要目标是同一地址，任意数量线程同时访问也只需 1 次事务。

```cpp
// 所有线程读同一地址 → 广播，无冲突
float val = smem[0];  // 全 Warp 广播
```

**规则 2 — 仅同一 Warp 内冲突：**

不同 Warp 的线程同时访问同一 Bank 不产生冲突。Bank Conflict 是 Warp 粒度的问题。

### 2.4 Stride 与冲突度的关系

设 Warp 中线程 `t` 访问 `smem[t * stride]`，元素大小 `sizeof(T) = S` 字节：

```
bank(t) = (t * stride * S / 4) % 32
```

**冲突度 = 32 / gcd(stride × S / 4, 32)**（当 `S / 4` 为整数时）

| Stride (float, S=4) | bank(t) 公式 | 冲突度 | 说明 |
|---------------------|-------------|--------|------|
| 1 | `t % 32` | 1-way | 完美，每线程不同 Bank |
| 2 | `(2t) % 32` | 2-way | 偶数 Bank 冲突 |
| 3 | `(3t) % 32` | 1-way | 3 与 32 互素 → 无冲突 |
| 4 | `(4t) % 32` | 4-way | |
| 8 | `(8t) % 32` | 8-way | |
| 16 | `(16t) % 32` | 16-way | |
| 32 | `(32t) % 32 = 0` | 32-way | 最差！所有线程同一 Bank |
| 33 | `(33t) % 32 = t` | 1-way | 33 与 32 互素 → 无冲突 |

**规律：** stride 与 32 互素时无冲突；stride 为 2 的幂次方时冲突最严重。

---

## 3. 常见冲突场景分析

### 3.1 矩阵转置 — 列访问

```cpp
__shared__ float tile[32][32];  // 32×32 float 矩阵

// 写入（行访问）：无冲突
tile[threadIdx.y][threadIdx.x] = input[gy * width + gx];  // stride=1 ✅

__syncthreads();

// 读取（列访问）：32-way 冲突！
output[gx * width + gy] = tile[threadIdx.x][threadIdx.y];  // stride=32 ❌
```

**分析：** `tile[threadIdx.x][threadIdx.y]` 中，不同 `threadIdx.x` 访问的地址间隔为 `32 × 4 = 128` 字节，全部落在同一 Bank。

```
Thread 0 → tile[0][col]  → Bank = (0 × 32 + col) % 32 = col
Thread 1 → tile[1][col]  → Bank = (1 × 32 + col) % 32 = col  ← 同一 Bank！
Thread 2 → tile[2][col]  → Bank = (2 × 32 + col) % 32 = col  ← 同一 Bank！
...
Thread 31→ tile[31][col] → Bank = (31× 32 + col) % 32 = col  ← 32-way conflict
```

### 3.2 GEMM 中的 Shared Memory Tile

```cpp
// A tile: M×K，行优先存储
__shared__ half A_tile[64][64];  // 64 列 × 2 字节 = 128 字节/行

// 线程沿 K 维度读取（列访问）时：
// stride = 64 halfs = 128 bytes = 32 words → 32-way conflict
half val = A_tile[k][threadIdx.x];  // ❌
```

### 3.3 Reduction 中的 Stride 访问

```cpp
__shared__ float smem[256];

// 树形归约的早期阶段
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
        smem[threadIdx.x] += smem[threadIdx.x + stride];
        // 当 stride = 32: threadIdx.x + 32 与 threadIdx.x 映射到同一 Bank
        // 但这里每个线程访问两个地址，其中一个可能冲突
    }
    __syncthreads();
}
```

### 3.4 char / short 类型的隐式冲突

```cpp
__shared__ char smem[128];

// 32 个线程读 smem[0..31]
char val = smem[threadIdx.x];

// Thread 0-3 → 同一个 4-byte word → Bank 0 的相同地址 → 广播 ✅
// 但如果是写入:
smem[threadIdx.x] = val;
// Thread 0-3 写入 Bank 0 的同一 word 的不同字节 → 需要 read-modify-write
// 性能取决于硬件实现，建议用 32-bit 类型
```

### 3.5 double (8 字节) 类型

```cpp
__shared__ double smem[32];

// Thread t → smem[t]
// 地址 = t × 8 字节 → Bank = (t × 8 / 4) % 32 = (2t) % 32
// → 2-way conflict（偶数/奇数交替）
```

---

## 4. 解决方案一：Padding

### 4.1 基本原理

Padding 通过在每行末尾添加无用元素，打破相邻行到同一 Bank 的对齐：

```
原始: tile[row][col] → address = (row × N + col) × S
Padded: tile[row][col] → address = (row × (N+P) + col) × S
```

当 `(N+P) × S / 4` 与 32 互素时，列访问无冲突。

### 4.2 标准用法

```cpp
// ❌ 有冲突: 列宽 = 32 × 4 = 128 字节 = 32 banks 整数倍
__shared__ float tile[32][32];

// ✅ 无冲突: 列宽 = 33 × 4 = 132 字节, 33 与 32 互素
__shared__ float tile[32][32 + 1];  // Pad 1 个 float

// 对于 half (2 字节):
// ❌ 有冲突: 64 × 2 = 128 字节
__shared__ half tile[64][64];

// ✅ 无冲突: 72 × 2 = 144 字节, 72/2 = 36, 36 与 32 互素? 不是
// 需要 pad 使得 (N+P) × sizeof(half) / 4 与 32 互素
// (64+P) × 2 / 4 = (64+P) / 2 需要与 32 互素
// P=1 → 65/2 = 32.5 (非整数，不能直接这么算)
// 正确做法: pad 到使得 word 对齐的行宽与 32 互素
// 64 half = 32 words → 冲突; pad 2 half = 33 words → 无冲突
__shared__ half tile[64][64 + 2];  // Pad 2 个 half = 1 个 word
```

### 4.3 Padding 量的计算规则

设元素大小 `S` 字节，原始列数 `N`：

```
行宽 (words) = N × S / 4

需要: 行宽 (words) 与 32 互素 (即不能被 2 整除)

最小 Padding 量 (元素数) = 使得 (N + P) × S / 4 为奇数的最小 P
```

| 元素类型 | 原始列数 N | 原始行宽 (words) | Pad 量 (元素) | Padded 行宽 (words) |
|----------|-----------|-----------------|--------------|-------------------|
| `float` (4B) | 32 | 32 | 1 | 33 ✅ |
| `float` (4B) | 64 | 64 | 1 | 65 ✅ |
| `float` (4B) | 128 | 128 | 1 | 129 ✅ |
| `half` (2B) | 32 | 16 | — | 16 (已无冲突) |
| `half` (2B) | 64 | 32 | 2 | 33 ✅ |
| `half` (2B) | 128 | 64 | 2 | 65 ✅ |
| `double` (8B) | 16 | 32 | 1 | 34 → 17 ✅ |
| `int8_t` (1B) | 128 | 32 | 4 | 33 ✅ |

### 4.4 完整代码示例：矩阵转置

```cpp
template <int TILE_DIM>
__global__ void transpose_padded(float* out, const float* in, int width) {
    // Pad 1 个 float 消除列访问冲突
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 行优先写入 (stride=1, 无冲突)
    if (x < width && y < width)
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];

    __syncthreads();

    // 列优先读出 (原本 stride=TILE_DIM → 现在因 padding 变为 stride=TILE_DIM+1，与 32 互素)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < width && y < width)
        out[y * width + x] = tile[threadIdx.x][threadIdx.y];
}
```

### 4.5 Padding 的优缺点

| 优点 | 缺点 |
|------|------|
| 实现极简单，改一个维度数字即可 | **浪费 Shared Memory**（对 LLM kernel 中需要大 tile 的场景影响较大） |
| 理解直观，不改变索引逻辑 | 增加行宽 → 可能影响 cache line 对齐 |
| 对 load/store 双向均有效 | 某些情况 Padding 量较大 (如 `char` 类型需 pad 4 字节) |
| 编译器无额外指令开销 | 与 `ldmatrix` / TMA 配合时可能破坏对齐要求 |

---

## 5. 解决方案二：XOR Swizzle

### 5.1 核心思想

Swizzle 通过对地址进行 **XOR 变换**，将逻辑上连续的列元素分散到不同 Bank，而不浪费存储空间。

核心映射：

```
swizzled_col = original_col XOR original_row
```

或更一般地：

```
new_addr = old_addr XOR f(row)
```

其中 `f(row)` 是行号的某个位操作函数。

### 5.2 为什么是 XOR？

**XOR 的双射 (Bijection) 性质保证无数据丢失：**

给定常量 `c`，函数 `f(x) = x ⊕ c` 在域 `[0, 2^m - 1]` 上是双射。这意味着：
- 每个输入映射到唯一输出（不会两个元素写到同一位置）
- 通过再次 XOR 相同常量可以逆变换（`f(f(x)) = x ⊕ c ⊕ c = x`）

**与其他方案对比：** 加法或移位也可消除冲突，但 XOR 在 GPU 上只需 1 条指令，且天然可逆。其本质是一个 Sudoku-like 映射——每行中每个 Bank 恰好被访问一次。

### 5.3 Swizzle 可视化

以 8×8 矩阵为例（简化为 8 个 Bank）：

**无 Swizzle（列访问 = 冲突）：**

```
        Col 0  Col 1  Col 2  Col 3  Col 4  Col 5  Col 6  Col 7
Row 0: [B0]   [B1]   [B2]   [B3]   [B4]   [B5]   [B6]   [B7]
Row 1: [B0]   [B1]   [B2]   [B3]   [B4]   [B5]   [B6]   [B7]
Row 2: [B0]   [B1]   [B2]   [B3]   [B4]   [B5]   [B6]   [B7]
Row 3: [B0]   [B1]   [B2]   [B3]   [B4]   [B5]   [B6]   [B7]
...
↑ 列访问: 所有行的 Col 0 → 全在 B0 → 8-way conflict！
```

**XOR Swizzle（`new_col = col XOR row`）：**

```
        Phys0  Phys1  Phys2  Phys3  Phys4  Phys5  Phys6  Phys7
Row 0: [L0]   [L1]   [L2]   [L3]   [L4]   [L5]   [L6]   [L7]    (0^row=0)
Row 1: [L1]   [L0]   [L3]   [L2]   [L5]   [L4]   [L7]   [L6]    (col^1)
Row 2: [L2]   [L3]   [L0]   [L1]   [L6]   [L7]   [L4]   [L5]    (col^2)
Row 3: [L3]   [L2]   [L1]   [L0]   [L7]   [L6]   [L5]   [L4]    (col^3)
Row 4: [L4]   [L5]   [L6]   [L7]   [L0]   [L1]   [L2]   [L3]    (col^4)
Row 5: [L5]   [L4]   [L7]   [L6]   [L1]   [L0]   [L3]   [L2]    (col^5)
Row 6: [L6]   [L7]   [L4]   [L5]   [L2]   [L3]   [L0]   [L1]    (col^6)
Row 7: [L7]   [L6]   [L5]   [L4]   [L3]   [L2]   [L1]   [L0]    (col^7)

↑ 列访问 (逻辑 Col 0): L0 在 Phys0, Phys1, Phys2, ..., Phys7 → 全不同 Bank ✅
↑ 行访问 (任意 Row): 物理位置连续，Bank 0-7 → 无冲突 ✅
```

### 5.4 实现代码

```cpp
// ============= 写入: Global → Shared (Swizzled) =============
template <int TILE_M, int TILE_N>
__device__ void store_swizzled(float smem[][TILE_N], const float* gmem,
                                int row, int col, int ld) {
    int swizzled_col = col ^ row;  // XOR swizzle
    smem[row][swizzled_col] = gmem[row * ld + col];
}

// ============= 读取: Shared (Swizzled) → Register =============
template <int TILE_M, int TILE_N>
__device__ float load_swizzled(float smem[][TILE_N], int row, int col) {
    int swizzled_col = col ^ row;  // 相同的 XOR 变换
    return smem[row][swizzled_col];
}

// ============= 完整矩阵转置示例 =============
__global__ void transpose_swizzled(float* out, const float* in, int N) {
    __shared__ float tile[32][32];  // 不需要 padding！

    int bx = blockIdx.x * 32, by = blockIdx.y * 32;
    int tx = threadIdx.x, ty = threadIdx.y;

    // 写入: row-major, 用 swizzle 打散 Bank
    int gx = bx + tx, gy = by + ty;
    if (gx < N && gy < N) {
        int sw_col = tx ^ ty;        // swizzle: col XOR row
        tile[ty][sw_col] = in[gy * N + gx];
    }
    __syncthreads();

    // 读取: 转置索引, 同样 swizzle
    gx = by + tx;
    gy = bx + ty;
    if (gx < N && gy < N) {
        int sw_col = ty ^ tx;        // 读取的逻辑 col=ty, row=tx
        out[gy * N + gx] = tile[tx][sw_col];
    }
}
```

### 5.5 通用 XOR Swizzle 公式

对于更一般的情况（元素不是 4 字节、列数不是 32），需要控制 XOR 的位范围：

```cpp
// 通用 swizzle: 只对地址的 Bank 位 (bit 2..6) 做 XOR
__device__ int swizzle_offset(int row, int col, int elem_bytes) {
    int linear = row * COLS + col;
    int byte_offset = linear * elem_bytes;

    // 提取 bank 位 (bit 2..6)
    int bank_bits = (byte_offset >> 2) & 0x1F;

    // 用 row 的低 5 位做 XOR
    int row_bits = row & 0x1F;
    int new_bank_bits = bank_bits ^ row_bits;

    // 替换 bank 位
    int new_offset = (byte_offset & ~0x7C) | (new_bank_bits << 2);
    return new_offset;
}
```

### 5.6 Swizzle 的优缺点

| 优点 | 缺点 |
|------|------|
| **不浪费存储空间** | 实现复杂度高，索引计算需要 XOR |
| 对 load 和 store 双向有效 | 需要在所有访问点一致 swizzle，否则数据错乱 |
| 与 Padding 性能相当（约 20% 提升） | 调试时地址不再线性，观察数据困难 |
| XOR 只需 1 条指令 | 对非 2 的幂次维度需要更复杂的映射 |

---

## 6. 解决方案三：CuTe Swizzle 抽象

### 6.1 三参数模型

CUTLASS/CuTe 将 Swizzle 封装为 `Swizzle<BBits, MBase, SShift>` 三参数模型：

```
地址位布局:
  0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                                ^--^ MBase: 最低位中保持不变的位数
                           ^-^ ^-^ BBits: 参与 XOR 的掩码位数
                ^---------^  SShift: YYY 掩码相对 ZZZ 的偏移距离
```

**操作：** `ZZZ_new = ZZZ XOR YYY`

即从地址高位提取 `BBits` 位 (YYY)，与低位的 `BBits` 位 (ZZZ) 做 XOR，其余位不变。

### 6.2 参数计算公式

设：
- `S` = 元素大小（字节）
- `N_vec` = 向量宽度（向量化访问时的元素个数，如 `float4` → `N_vec = 4`）
- `X` = 行中的元素数（快速维度大小）

```
MBase = log2(N_vec)                    // 向量内元素不重排
BBits = log2(32 × 4 / S) − MBase      // XOR 覆盖所有 32 个 Bank
SShift = log2(X) − MBase              // YYY 位对应行号
```

**约束：** `S × 2^(MBase + BBits) = 128` (即 XOR 范围恰好覆盖 32 个 4-byte Bank)

### 6.3 常见配置示例

| 矩阵形状 | 元素类型 | 向量宽度 | MBase | BBits | SShift | CuTe 记法 |
|----------|---------|---------|-------|-------|--------|-----------|
| 32×32 float | 4B | 1 | 0 | 5 | 5 | `Swizzle<5,0,5>` |
| 32×64 float | 4B | 1 | 0 | 5 | 6 | `Swizzle<5,0,6>` |
| 64×64 half | 2B | 1 | 0 | 6 | 6 | `Swizzle<6,0,6>` |
| 64×64 half (v2) | 2B | 2 | 1 | 5 | 5 | `Swizzle<5,1,5>` |
| 64×64 half (v8) | 2B | 8 | 3 | 3 | 3 | `Swizzle<3,3,3>` |
| 128×128 half (v8) | 2B | 8 | 3 | 3 | 4 | `Swizzle<3,3,4>` |

### 6.4 Swizzle 应用过程（逐步）

以 `Swizzle<3, 4, 3>` 为例（CUTLASS GEMM 中常见）：

```
Step 1: bit_msk = (1 << 3) - 1 = 0b111

Step 2: yyy_msk = 0b111 << (4 + max(0,3)) = 0b111 << 7 = 0b0000_0011_1000_0000
         (bit 7,8,9 = YYY)

Step 3: zzz_msk = 0b111 << (4 - min(0,3)) = 0b111 << 4 = 0b0000_0000_0111_0000
         (bit 4,5,6 = ZZZ)

Step 4: 对于输入地址 addr:
         yyy = (addr >> 7) & 0b111    // 提取 bit 7-9
         zzz = (addr >> 4) & 0b111    // 提取 bit 4-6
         new_zzz = zzz ^ yyy          // XOR
         new_addr = addr 中把 bit 4-6 替换为 new_zzz
```

### 6.5 CuTe 代码用法

```cpp
#include <cute/swizzle.hpp>

using namespace cute;

// 定义 swizzle 布局
auto tileLayout = make_layout(
    make_shape(Int<32>{}, Int<32>{}),
    GenRowMajor{}
);

// 组合 swizzle
auto swizzledLayout = composition(Swizzle<5, 0, 5>{}, tileLayout);

// 使用 swizzled 布局写入/读取共享内存
// CuTe 自动处理索引变换
```

### 6.6 手动实现 CuTe 式 Swizzle

```cpp
template <int BBits, int MBase, int SShift>
struct ManualSwizzle {
    static constexpr int bit_msk = (1 << BBits) - 1;
    static constexpr int yyy_shift = MBase + (SShift > 0 ? SShift : 0);
    static constexpr int zzz_shift = MBase + (SShift < 0 ? -SShift : 0);
    static constexpr int yyy_msk = bit_msk << yyy_shift;

    __device__ static int apply(int offset) {
        // 提取 YYY 位，右移到 ZZZ 位置，XOR
        return offset ^ ((offset & yyy_msk) >> SShift);
    }

    __device__ static int inverse(int offset) {
        // XOR 是自逆的（对相同操作）
        return apply(offset);  // f(f(x)) = x 仅当 SShift 使 YYY/ZZZ 不重叠
    }
};

// 使用
__shared__ __align__(128) half smem[64 * 64];

// 写入
int linear_offset = row * 64 + col;
int swizzled_offset = ManualSwizzle<5, 1, 5>::apply(linear_offset);
smem[swizzled_offset] = value;

// 读取
half value = smem[ManualSwizzle<5, 1, 5>::apply(row * 64 + col)];
```

---

## 7. 解决方案四：TMA Swizzle (Hopper+)

### 7.1 硬件自动 Swizzle

Hopper (SM 9.0+) 的 TMA (Tensor Memory Accelerator) 引擎内置了硬件 Swizzle 支持。TMA 在数据从 Global Memory 传输到 Shared Memory 时，自动应用 Swizzle 模式，无需任何额外的索引计算指令。

```
Global Memory (线性) ──TMA──→ Shared Memory (Swizzled)
                           自动硬件重排
```

### 7.2 三种 Swizzle 模式

通过 `cuTensorMapEncode*()` 设置 Swizzle 模式：

| 模式 | 枚举值 | XOR 粒度 | Bounding Box 内维限制 | 对齐要求 (smem) |
|------|--------|---------|---------------------|---------------|
| 32B | `CU_TENSOR_MAP_SWIZZLE_32B` | 32 字节单元 | ≤ 32 字节 | 128 字节 |
| 64B | `CU_TENSOR_MAP_SWIZZLE_64B` | 64 字节单元 | ≤ 64 字节 | 256 字节 |
| 128B | `CU_TENSOR_MAP_SWIZZLE_128B` | 128 字节单元 | ≤ 128 字节 | 1024 字节 |
| None | `CU_TENSOR_MAP_SWIZZLE_NONE` | 不 swizzle | 无限制 | 16 字节 |

**对于 LLM Kernel 的 GEMM/Attention：** 通常使用 `128B` 模式，因为 Tensor Core 指令 (WGMMA) 每次消费 128 字节的 shared memory 行。

### 7.3 128B Swizzle 工作原理

在 128B 模式下，每行 128 字节被划分为 8 个 16 字节块。这 8 个块根据行号进行 XOR 重排：

```
逻辑行中的 8 个 16B 块: [chunk0] [chunk1] [chunk2] ... [chunk7]

Row 0 物理顺序: [c0] [c1] [c2] [c3] [c4] [c5] [c6] [c7]  (不变)
Row 1 物理顺序: [c1] [c0] [c3] [c2] [c5] [c4] [c7] [c6]  (chunk_id ^ 1)
Row 2 物理顺序: [c2] [c3] [c0] [c1] [c6] [c7] [c4] [c5]  (chunk_id ^ 2)
Row 3 物理顺序: [c3] [c2] [c1] [c0] [c7] [c6] [c5] [c4]  (chunk_id ^ 3)
...
Row 7 物理顺序: [c7] [c6] [c5] [c4] [c3] [c2] [c1] [c0]  (chunk_id ^ 7)
Row 8: 回到 Row 0 的模式 (8 行为一个周期)
```

**效果：** 列方向访问时，连续 8 行中同一逻辑列的数据分布在不同的 16B 块（不同 Bank 组）中，消除冲突。

### 7.4 TMA 描述符配置

```cpp
#include <cuda.h>

CUtensorMap tensorMap;

// 配置 2D TMA 描述符 + 128B swizzle
cuTensorMapEncodeTiled(
    &tensorMap,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,      // 元素类型
    2,                                      // 维度
    globalAddress,                          // 全局内存基址
    globalDim,                              // 全局维度 {N, M}
    globalStrides,                          // 全局步长 (字节)
    boxDim,                                 // tile 维度 {tileN, tileM}
    elementStrides,                         // 元素步长 {1, 1}
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,            // ← 128B swizzle
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```

### 7.5 TMA Swizzle 的优势

| 特性 | 说明 |
|------|------|
| **零指令开销** | 硬件自动完成，不需要 XOR 计算指令 |
| **不浪费空间** | 与手动 Swizzle 相同，无 Padding |
| **与 WGMMA 完美配合** | Swizzle 后的布局正是 WGMMA 期望的 shared memory 描述符格式 |
| **自动边界处理** | 越界区域自动填零 |
| **单线程发起** | 仅 1 个线程设置 TMA，其余线程可做计算 |

### 7.6 读取 Swizzled 数据

写入由 TMA 自动完成，但如果需要手动从 swizzled 的 shared memory 中读取（如 epilogue 阶段），需要知道 swizzle 映射：

```cpp
// 从 128B-swizzled 共享内存读取
__device__ half read_swizzled_128B(const half* smem_base, int row, int col) {
    // 128B = 64 个 half，所以行内有 64 个元素
    // 划分为 8 个 chunk，每个 chunk 16B = 8 个 half
    int chunk_id = col / 8;
    int intra_chunk = col % 8;

    // XOR swizzle
    int swizzled_chunk = chunk_id ^ (row % 8);

    return smem_base[row * 64 + swizzled_chunk * 8 + intra_chunk];
}
```

---

## 8. ldmatrix 与 128-bit 访问的 Bank Conflict

### 8.1 宽访问的分阶段执行

当线程执行 128-bit (16 字节) 加载（如 `ld.shared.v4.f32` 或 `ldmatrix.x4`）时，硬件将 Warp 的 32 个线程分为 **4 个 Phase**，每 Phase 8 个线程，依次执行：

```
Phase 0: Thread  0- 7  各访问 16 字节 → 总 128 字节
Phase 1: Thread  8-15  各访问 16 字节 → 总 128 字节
Phase 2: Thread 16-23  各访问 16 字节 → 总 128 字节
Phase 3: Thread 24-31  各访问 16 字节 → 总 128 字节
```

### 8.2 等效 8-Bank 模型

每个线程访问 16 字节 = 4 个连续 Bank。在每个 Phase 内，有效 Bank 数变为 **32 / 4 = 8 个 16 字节宽的 "超级 Bank"**。

```
SuperBank 0: Bank  0- 3  (16 字节)
SuperBank 1: Bank  4- 7  (16 字节)
SuperBank 2: Bank  8-11  (16 字节)
SuperBank 3: Bank 12-15  (16 字节)
SuperBank 4: Bank 16-19  (16 字节)
SuperBank 5: Bank 20-23  (16 字节)
SuperBank 6: Bank 24-27  (16 字节)
SuperBank 7: Bank 28-31  (16 字节)
```

8 个线程需要访问 8 个不同的 SuperBank 才能无冲突。

### 8.3 ldmatrix 指令的访问模式

`ldmatrix.sync.aligned.x4.m8n8` 指令让每个线程提供一个地址，加载 16 字节 (8 个 half)。Warp 的 32 个线程共加载 32 × 16 = 512 字节，组成矩阵 fragment 供 `mma.sync` 使用。

**无 Swizzle 时的冲突：**

如果 shared memory 中矩阵按行优先线性存储（行宽 128 字节 = 8 个 SuperBank），则：

```
Phase 0 (Thread 0-7):
  T0 → Row 0, SuperBank 0  ← 指向 addr 0
  T1 → Row 1, SuperBank 0  ← 指向 addr 128
  T2 → Row 2, SuperBank 0  ← 指向 addr 256
  ...
  T7 → Row 7, SuperBank 0  ← 指向 addr 896

全部 8 个线程都在 SuperBank 0 → 8-way conflict！
```

**Swizzle 后无冲突：**

使用 128B XOR swizzle 后，每行的物理 chunk 顺序被 XOR 打散：

```
Phase 0 (Thread 0-7):
  T0 → Row 0, SuperBank 0^0 = 0
  T1 → Row 1, SuperBank 0^1 = 1
  T2 → Row 2, SuperBank 0^2 = 2
  ...
  T7 → Row 7, SuperBank 0^7 = 7

8 个线程 → 8 个不同的 SuperBank → 无冲突 ✅
```

### 8.4 向量化加载的硬件调度

对于 `ld.shared.v4.f32`（128-bit 向量加载），硬件有另一种优化：每个 Phase 中 8 个线程各需 4 个 Bank，硬件可以**重排各线程的 Bank 访问顺序**，将原本 4 cycle 的加载优化为无冲突的 4 cycle 流水。

```
Cycle 1: T0→B0, T1→B4, T2→B8,  ..., T7→B28
Cycle 2: T0→B1, T1→B5, T2→B9,  ..., T7→B29
Cycle 3: T0→B2, T1→B6, T2→B10, ..., T7→B30
Cycle 4: T0→B3, T1→B7, T2→B11, ..., T7→B31
```

每个 cycle 8 个线程各访问不同 Bank → 无冲突。因此 **stride-1 的 128-bit 向量加载天然无冲突**（硬件调度）。

但注意：**`ldmatrix` 的访问模式不是 stride-1**（每线程加载不同行的首元素），所以必须依赖 Swizzle 消除冲突。

---

## 9. LLM Kernel 实战模式

### 9.1 GEMM Tile 的 Swizzle 策略

**问题：** GEMM 中 A 矩阵 tile (M×K) 加载后，Tensor Core 消费时需要按 K 维度列式访问。

```
A tile [64×64] half，行优先存储:
  行宽 = 64 × 2 = 128 字节 = 32 个 Bank 的完整周期

  列访问 (K 维度): 相邻行同一列 → 同一 Bank → 严重冲突
```

**解决方案（按架构选择）：**

| 架构 | 推荐方案 | 实现 |
|------|---------|------|
| SM 8.0 (A100) | Padding 或 XOR Swizzle | `half A[64][64+2]` 或手动 XOR |
| SM 8.0 (A100) | CUTLASS Swizzle | `Swizzle<3,3,3>` + `ldmatrix` |
| SM 9.0 (H100) | TMA 128B Swizzle | `CU_TENSOR_MAP_SWIZZLE_128B` |
| SM 10.0 (B200) | TMA 128B Swizzle | 同 Hopper（TMA 复用） |

### 9.2 FlashAttention 的 Q/K/V Tile

```cpp
// FlashAttention: Q[Br×d], K[Bc×d], V[Bc×d]
// d = head_dim (通常 64 或 128)

// Q tile 加载后按列消费 (MMA 需要)
// 行宽 = d × sizeof(half) = 128 或 256 字节

// 方案 1: Padding (简单但浪费)
__shared__ half Q_tile[Br][d + 2];  // pad 2 half = 1 word

// 方案 2: Swizzle (零浪费)
// 使用 CuTe Swizzle<3, 3, 3> 对 d=64 的 tile
// 或 TMA 128B swizzle (Hopper)
```

### 9.3 Softmax 行归约的 Shared Memory

```cpp
// Softmax 中的行最大值/行求和需要跨 Warp 通信
__shared__ float row_max[NUM_ROWS];
__shared__ float row_sum[NUM_ROWS];

// 每个 Warp 的 lane 0 写入结果，其他 lane 读取
// 访问模式: stride-1, 无冲突 ✅ (每个 Warp 只有 1 个线程写)
```

### 9.4 Double Buffering 中的对齐考虑

```cpp
// 双缓冲需要两个 tile 在 shared memory 中
// 确保每个 buffer 的起始地址对齐

// 方案 1: 自然对齐
__shared__ __align__(128) half buf0[64][64];  // 对齐到 128B
__shared__ __align__(128) half buf1[64][64];

// 方案 2: 动态分配时手动对齐
extern __shared__ __align__(1024) char smem[];  // TMA 128B swizzle 需要 1024B 对齐
half* buf0 = reinterpret_cast<half*>(smem);
half* buf1 = reinterpret_cast<half*>(smem + ALIGNED_SIZE);
```

---

## 10. 性能分析与调试

### 10.1 检测 Bank Conflict

**Nsight Compute (推荐)：**

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared,\
              l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,\
              l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st \
    ./my_kernel
```

关键指标：

| 指标 | 含义 | 目标值 |
|------|------|--------|
| `shared_load_transactions_per_request` | 每次共享内存加载请求的平均事务数 | 1.0 (无冲突) |
| `shared_store_transactions_per_request` | 每次存储请求的平均事务数 | 1.0 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` | Bank Conflict 总次数 | 0 |
| `smsp__sass_average_data_bytes_per_waveslot_mem_shared` | 平均每波每槽传输字节 | 128 (理论最大) |

**nvprof (Legacy)：**

```bash
nvprof --events shared_ld_bank_conflict,shared_st_bank_conflict ./my_kernel
nvprof --metrics shared_efficiency ./my_kernel
```

### 10.2 验证 Swizzle 正确性

```cpp
// 调试辅助: 打印 swizzle 映射
__global__ void debug_swizzle() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Row -> Bank mapping for Col 0:\n");
        for (int row = 0; row < 8; row++) {
            int col = 0;
            int swizzled = col ^ row;
            int bank = swizzled % 32;
            printf("  Row %d: logical col %d -> physical col %d -> Bank %d\n",
                   row, col, swizzled, bank);
        }
    }
}
```

### 10.3 常见错误与排查

| 症状 | 可能原因 | 排查方法 |
|------|---------|---------|
| `shared_efficiency` < 50% | 列访问未 swizzle/pad | 检查 smem 布局和访问 stride |
| `bank_conflicts` 非零但少量 | 部分线程走不同路径 | 检查边界条件处理 |
| 数据计算结果错误 | Swizzle 写入与读取不匹配 | 确保所有访问点使用相同的 swizzle 函数 |
| `ldmatrix` 结果错乱 | Swizzle 与 `ldmatrix` 期望布局不兼容 | 按 CUTLASS 文档检查 fragment 布局 |
| TMA 加载后数据错位 | smem 对齐不满足 swizzle 要求 | 检查 `__align__` 值 (128B swizzle 需 1024B 对齐) |

---

## 11. 方案对比与选型指南

### 11.1 四种方案对比

| 维度 | Padding | XOR Swizzle | CuTe Swizzle | TMA Swizzle |
|------|---------|-------------|-------------|-------------|
| **架构要求** | 所有 SM | 所有 SM | 所有 SM | SM 9.0+ (Hopper) |
| **存储浪费** | 有 (~3%) | 无 | 无 | 无 |
| **指令开销** | 0 | 1 条 XOR/thread | 1-2 条位操作/thread | 0 (硬件) |
| **实现复杂度** | 极低 | 中 | 中高 (需理解 CuTe) | 低 (配置 TMA) |
| **与 ldmatrix 兼容** | 可能破坏对齐 | 需精确匹配 | 原生支持 | 原生支持 |
| **与 WGMMA 兼容** | 不兼容 | 需精确匹配 | 原生支持 | 原生支持 |
| **调试难度** | 低 | 中 | 中 | 低 |
| **性能** | 好 (~20% 提升) | 好 (~20% 提升) | 好 | 最优 (零开销) |

### 11.2 选型决策树

```
需要消除 Bank Conflict?
│
├─ SM 9.0+ (Hopper/Blackwell)?
│   ├─ 使用 TMA? → TMA Swizzle (CU_TENSOR_MAP_SWIZZLE_128B)  ★ 推荐
│   └─ 不使用 TMA (cp.async) → CuTe Swizzle / XOR Swizzle
│
├─ SM 8.x (Ampere/Ada)?
│   ├─ 使用 CUTLASS/CuTe? → CuTe Swizzle  ★ 推荐
│   ├─ 使用 ldmatrix? → XOR Swizzle (手动匹配 fragment 布局)
│   └─ 简单 Kernel? → Padding (最简实现)
│
└─ 通用/快速原型?
    └─ Padding  ★ 推荐 (改一个数字即可)
```

### 11.3 LLM Kernel 场景推荐

| Kernel 类型 | 推荐方案 | 理由 |
|------------|---------|------|
| GEMM (Hopper) | TMA 128B Swizzle | 与 WGMMA 完美配合，零开销 |
| GEMM (Ampere) | CuTe Swizzle | CUTLASS 原生支持，`ldmatrix` 兼容 |
| FlashAttention (Hopper) | TMA 128B Swizzle | Q/K/V tile 自动 swizzle |
| FlashAttention (Ampere) | XOR Swizzle / CuTe | 配合 `cp.async` + `ldmatrix` |
| Fused Softmax | Padding | 行归约为主，smem 用量小 |
| LayerNorm | Padding | 实现简单，smem 用量小 |
| 矩阵转置 | Padding 或 XOR Swizzle | 取决于性能要求 |
| Elementwise Fusion | 通常不需要 | 行访问 stride=1 天然无冲突 |

### 11.4 Kernel Agent 生成代码检查清单

1. **识别 Shared Memory 访问模式：**
   - 是否存在列式访问或 stride ≥ 2 的访问？
   - 行宽 (字节) 是否是 128 的倍数？

2. **选择解决方案：**
   - Hopper + TMA → 配置 Swizzle 模式
   - Ampere + Tensor Core → CuTe/XOR Swizzle
   - 简单场景 → Padding

3. **验证正确性：**
   - 所有 smem 访问点使用一致的 swizzle/padding
   - 对齐要求满足（特别是 TMA 的 1024B 对齐）

4. **性能验证：**
   - `shared_load_transactions_per_request` ≈ 1.0
   - `l1tex__data_bank_conflicts` = 0

---

## 参考资源

- [NVIDIA CUDA Programming Guide — Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-5-x)
- [Axel Feldmann — Notes About Nvidia GPU Shared Memory Banks](https://feldmann.nyc/blog/smem-microbenchmarks)
- [NVIDIA Developer Forum — How to understand the bank conflict of shared_mem](https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900)
- [Modal GPU Glossary — Bank Conflict](https://modal.com/gpu-glossary/perf/bank-conflict)
- [Lei Mao — CUDA Shared Memory Swizzling](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/)
- [Lei Mao — CuTe Swizzle](https://leimao.github.io/blog/CuTe-Swizzle/)
- [Simon Veitner — Understanding CuTe Swizzling](https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/)
- [Flash Attention from Scratch Part 4: Bank Conflicts & Swizzling](https://lubits.ch/flash/Part-4)
- [CUTLASS GitHub Discussion #1130](https://github.com/NVIDIA/cutlass/discussions/1130)
- [NVIDIA CUDA Driver API — Tensor Map Object Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- [PyTorch Blog — Deep Dive on the Hopper TMA Unit for FP8 GEMMs](https://pytorch.org/blog/hopper-tma-unit/)

---

*本文档作为 LLM Kernel Agent 的 Shared Memory 无冲突访问技能参考。配合 `official-std/` 目录下的 CUDA 编程指南、最佳实践、nvcc 编译器和 PTX ISA 文档共同使用。*
