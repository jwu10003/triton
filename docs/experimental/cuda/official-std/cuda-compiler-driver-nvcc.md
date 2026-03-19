# CUDA Compiler Driver NVCC — Kernel Agent Skill Reference

> 基于 [NVIDIA CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) 编写。
> 面向 LLM 高性能 Kernel 自动生成场景，聚焦 **nvcc 编译流程、架构选项、优化开关与实战编译命令**。
> 与 `cuda-c-programming-guide.md`（语言特性）和 `cuda-cpp-best-practices-guide.md`（性能优化）互补。

---

## 目录

1. [NVCC 概述](#1-nvcc-概述)
2. [编译流程与阶段](#2-编译流程与阶段)
3. [GPU 架构选项](#3-gpu-架构选项)
4. [编译阶段控制选项](#4-编译阶段控制选项)
5. [优化与性能选项](#5-优化与性能选项)
6. [浮点与精度选项](#6-浮点与精度选项)
7. [调试与诊断选项](#7-调试与诊断选项)
8. [C++ 语言特性选项](#8-c-语言特性选项)
9. [Pass-Through 选项](#9-pass-through-选项)
10. [分离编译与链接](#10-分离编译与链接)
11. [NVRTC 运行时编译](#11-nvrtc-运行时编译)
12. [环境变量与配置](#12-环境变量与配置)
13. [实战编译命令速查](#13-实战编译命令速查)

---

## 1. NVCC 概述

### 1.1 什么是 NVCC

**nvcc** 是 NVIDIA CUDA 编译器驱动（Compiler Driver），负责协调 CUDA C++ 程序的完整编译流程。它本身**不是一个单体编译器**，而是一个**编排器**，依次调用：

| 内部工具 | 职责 |
|----------|------|
| **cicc** | CUDA C++ → PTX 编译器（Device 前端+优化） |
| **ptxas** | PTX → cubin 汇编器（Device 后端） |
| **fatbinary** | 打包多个 PTX/cubin 为 fatbin |
| **cudafe++** | CUDA 前端：分离 Host/Device 代码 |
| **Host compiler** (gcc/g++/cl/clang) | 编译 Host C++ 代码 |
| **nvlink** | Device 代码链接器（分离编译模式） |
| **Host linker** (ld/link) | Host 对象文件链接 |

### 1.2 NVCC 版本标识

```cuda
// 编译时可用的预定义宏
__CUDACC__                    // nvcc 编译 .cu 文件时定义
__CUDACC_VER_MAJOR__          // 主版本号 (如 12)
__CUDACC_VER_MINOR__          // 次版本号 (如 6)
__CUDACC_VER_BUILD__          // 构建号
__CUDA_ARCH__                 // Device 代码编译时定义，值 = CC × 10
                              // 例: sm_90 → __CUDA_ARCH__ == 900
```

```bash
# 查看 nvcc 版本
nvcc --version
# 例: Cuda compilation tools, release 12.6, V12.6.77
```

---

## 2. 编译流程与阶段

### 2.1 CUDA 编译总流程

```
                        ┌─────────────────────────────────────┐
                        │          .cu 源文件                  │
                        └───────────────┬─────────────────────┘
                                        │
                                  ┌─────▼─────┐
                                  │ cudafe++  │  分离 Host/Device 代码
                                  └──┬────┬───┘
                                     │    │
                        ┌────────────▼┐  ┌▼────────────┐
                        │ Device 代码  │  │  Host 代码   │
                        └──────┬──────┘  └──────┬──────┘
                               │                │
                        ┌──────▼──────┐         │
                        │    cicc     │         │
                        │ CUDA → PTX  │         │
                        └──────┬──────┘         │
                               │                │
                        ┌──────▼──────┐         │
                        │   ptxas     │         │
                        │ PTX → cubin │         │
                        └──────┬──────┘         │
                               │                │
                        ┌──────▼──────┐         │
                        │  fatbinary  │         │
                        │ 打包 fatbin  │         │
                        └──────┬──────┘         │
                               │                │
                               └───────┬────────┘
                                       │
                                ┌──────▼──────┐
                                │ Host 编译器  │  合并 fatbin 到 Host 对象
                                │ (gcc/clang)  │
                                └──────┬──────┘
                                       │
                                ┌──────▼──────┐
                                │ Host 链接器  │
                                └──────┬──────┘
                                       │
                                ┌──────▼──────┐
                                │  可执行文件   │
                                └─────────────┘
```

### 2.2 两阶段编译模型

nvcc 采用 **两阶段编译模型**：

1. **Virtual Architecture → PTX**：CUDA C++ 编译为中间表示 PTX（虚拟架构的汇编）。
2. **PTX → cubin**：PTX 汇编为目标 GPU 的真实二进制（cubin）。

这种分离使得 **PTX 前向兼容** 成为可能：包含 PTX 的程序可在未来架构上通过 JIT 编译运行。

### 2.3 支持的输入文件类型

| 后缀 | 类型 | 处理方式 |
|------|------|----------|
| `.cu` | CUDA C++ 源文件 | 完整 CUDA 编译（Host + Device） |
| `.c` | C 源文件 | 仅 Host 编译 |
| `.cc`, `.cpp`, `.cxx` | C++ 源文件 | 仅 Host 编译 |
| `.ptx` | PTX 汇编文件 | ptxas 汇编为 cubin |
| `.cubin` | CUDA 二进制 | 直接嵌入 fatbin |
| `.fatbin` | Fat 二进制 | 直接使用 |
| `.o`, `.obj` | 对象文件 | 传递给链接器 |
| `.a`, `.lib` | 静态库 | 传递给链接器 |
| `.so`, `.dll` | 动态库 | 传递给链接器 |
| `.res` | 资源文件 | 传递给链接器 |

### 2.4 输出文件类型

| 输出 | 说明 | 包含内容 |
|------|------|----------|
| **可执行文件** | 默认输出 | Host 代码 + 嵌入的 fatbin |
| **fatbin** | `--fatbin` | 多个 PTX/cubin 的容器 |
| **cubin** | `--cubin` | 单一架构的 Device 二进制 |
| **PTX** | `--ptx` | 虚拟架构的文本汇编 |
| **对象文件** | `-c` | Host 对象 + 嵌入的 fatbin |
| **预处理输出** | `-E` | 预处理后的源代码 |

---

## 3. GPU 架构选项

### 3.1 核心概念

| 概念 | 说明 | 示例 |
|------|------|------|
| **Virtual Architecture** | PTX 指令集版本，以 `compute_XX` 命名 | `compute_80`, `compute_90` |
| **Real Architecture** | 实际 GPU 的 SM 版本，以 `sm_XX` 命名 | `sm_80`, `sm_90` |
| **PTX** | 虚拟架构的文本汇编，前向兼容 | 可 JIT 到更高 CC |
| **cubin** | 真实架构的二进制，同 major 内向上兼容 | sm_86 cubin 可运行在 sm_89 |
| **fatbin** | 包含多个 PTX/cubin 的容器 | 支持多架构 |

### 3.2 `-arch` 与 `-code`

```bash
# -arch: 指定虚拟架构（PTX 指令集版本）
# -code: 指定真实架构（生成 cubin 的目标）

# 生成 sm_80 的 cubin + compute_80 的 PTX
nvcc -arch=compute_80 -code=sm_80,compute_80 kernel.cu

# 简写：-arch=sm_XX 自动设置 -arch=compute_XX -code=sm_XX,compute_XX
nvcc -arch=sm_80 kernel.cu
```

**兼容规则**：`-arch=compute_X` 必须 ≤ `-code=sm_Y`（X ≤ Y）。

### 3.3 `-gencode`（多架构编译）

```bash
# 为多种架构生成 fatbin（推荐的生产构建方式）
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 \
     -o my_app kernel.cu
```

**最后一行 `code=compute_90`** 嵌入 PTX，提供前向兼容（JIT 到未来架构）。

### 3.4 架构后缀（CUDA 12.9+）

| 后缀 | 含义 | 兼容性 |
|------|------|--------|
| 无后缀 | 标准兼容 | cubin 在同 major CC 内向上兼容 |
| `f` | Family-specific | PTX/cubin 在同 major CC 内兼容 |
| `a` | Architecture-specific | **仅**在精确匹配的 CC 上运行 |

```bash
# sm_90a: Hopper 特有指令（TMA, WGMMA），不兼容其他架构
nvcc -arch=sm_90a kernel.cu

# sm_100f: Blackwell 家族兼容
nvcc -arch=sm_100f kernel.cu
```

### 3.5 完整架构映射表

| 架构 | Virtual (PTX) | Real (cubin) | 代表 GPU |
|------|--------------|--------------|----------|
| Volta | `compute_70` | `sm_70` | V100 |
| Turing | `compute_75` | `sm_75` | T4, RTX 2080 |
| Ampere | `compute_80` | `sm_80` | A100 |
| Ampere | `compute_86` | `sm_86` | RTX 3090, A40 |
| Ada Lovelace | `compute_89` | `sm_89` | RTX 4090, L40 |
| Hopper | `compute_90` | `sm_90` / `sm_90a` | H100, H200 |
| Blackwell | `compute_100` | `sm_100` / `sm_100f` | B200 |
| Blackwell | `compute_120` | `sm_120` | GB200 |

```bash
# 查看 nvcc 支持的所有架构
nvcc --list-gpu-arch    # 列出虚拟架构
nvcc --list-gpu-code    # 列出真实架构
```

### 3.6 兼容性规则详解

```
PTX 前向兼容：
  compute_70 的 PTX → 可 JIT 运行在 sm_70, sm_75, sm_80, sm_86, sm_89, sm_90, ...
  compute_90 的 PTX → 可 JIT 运行在 sm_90, sm_100, ...
                    → 不可运行在 sm_80, sm_89 (低于 PTX 版本)

cubin 同代兼容：
  sm_80 的 cubin → 可运行在 sm_80, sm_86, sm_87
                → 不可运行在 sm_89 (不同 major), sm_90
  sm_86 的 cubin → 可运行在 sm_86, sm_87
                → 不可运行在 sm_80 (同 major 但更低 minor)

sm_90a (Architecture-specific)：
  → 仅运行在 CC 9.0，不兼容任何其他架构
  → 没有前向兼容
```

### 3.7 JIT 编译

当 fatbin 中没有匹配目标 GPU 的 cubin 但包含适当的 PTX 时，CUDA 驱动会在首次加载时 **JIT 编译** PTX → cubin。

```
优点：无需预编译即可支持新架构
缺点：首次启动延迟（编译开销）
解决：CUDA 缓存 JIT 结果（~/.nv/ComputeCache/）
```

---

## 4. 编译阶段控制选项

### 4.1 阶段选项一览

| 选项 | 说明 | 输出 |
|------|------|------|
| `-E` | 仅预处理 | `.cpp.ii` |
| `-c` | 编译到对象文件（不链接） | `.o` / `.obj` |
| `--cuda` | CUDA 编译到中间 C++ | `.cu.cpp.ii` |
| `--ptx` | 编译到 PTX | `.ptx` |
| `--cubin` | 编译到 cubin | `.cubin` |
| `--fatbin` | 编译到 fatbin | `.fatbin` |
| `-dc` / `--device-c` | 编译为可重定位 Device 代码对象 | `.o`（含 RDC） |
| `-dlink` / `--device-link` | Device 代码链接 | `.o`（可执行 Device 代码） |
| `--lib` | 创建静态库 | `.a` / `.lib` |
| `--run` | 编译并立即运行 | — |

### 4.2 常用组合

```bash
# 仅生成 PTX（查看编译器输出）
nvcc --ptx -arch=sm_90 kernel.cu -o kernel.ptx

# 仅生成 cubin
nvcc --cubin -arch=sm_90 kernel.cu -o kernel.cubin

# 编译为对象文件（多文件项目）
nvcc -c -arch=sm_90 kernel.cu -o kernel.o

# 链接对象文件
nvcc kernel.o main.o -arch=sm_90 -o my_app

# 编译并立即运行（快速测试）
nvcc --run -arch=sm_90 test.cu
```

---

## 5. 优化与性能选项

### 5.1 优化级别

| 选项 | 说明 | 影响范围 |
|------|------|----------|
| `-O0` | 无优化（调试用） | Host + Device |
| `-O1` | 基本优化 | Host + Device |
| `-O2` | 标准优化（实际等同 -O3） | Host + Device |
| `-O3` | **最高优化（默认）** | Host + Device |

**nvcc 默认使用 `-O3`**。ptxas 也默认使用优化级别 3。

### 5.2 寄存器控制

```bash
# 限制每线程最大寄存器数
nvcc --maxrregcount=64 -arch=sm_90 kernel.cu

# 等价方式：通过 ptxas
nvcc -Xptxas=-maxrregcount=64 -arch=sm_90 kernel.cu
```

**权衡**：更少寄存器 → 更高 Occupancy → 可能 register spill 到 Local Memory。

代码中也可以使用 `__launch_bounds__` 达到类似效果：

```cuda
__global__ void __launch_bounds__(256, 4) myKernel(...) {
    // maxThreadsPerBlock=256, minBlocksPerSM=4
    // 编译器据此分配寄存器
}
```

### 5.3 ptxas 优化选项

```bash
# 查看 Kernel 资源使用量（寄存器、Shared Memory、Local Memory）
nvcc -Xptxas=-v -arch=sm_90 kernel.cu
# 输出示例:
# ptxas info: Compiling entry function 'myKernel'
# ptxas info: Used 42 registers, 8192 bytes smem, 0 bytes lmem, 352 bytes cmem[0]

# 警告寄存器溢出
nvcc -Xptxas=-warn-spills -arch=sm_90 kernel.cu

# 警告 Local Memory 使用
nvcc -Xptxas=-warn-lmem-usage -arch=sm_90 kernel.cu

# 允许编译器使用更多时间做优化（编译变慢，代码可能更快）
nvcc -Xptxas=-allow-expensive-optimizations=true -arch=sm_90 kernel.cu

# 控制 L1 缓存行为（全局加载的默认缓存策略）
nvcc -Xptxas=-dlcm=ca  kernel.cu   # 缓存到 L1+L2（默认）
nvcc -Xptxas=-dlcm=cg  kernel.cu   # 仅缓存到 L2（跳过 L1）
```

### 5.4 并行编译

```bash
# 多线程编译（多架构并行）
nvcc -t 8 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_90,code=sm_90 \
     kernel.cu

# split compilation（优化阶段并行）
nvcc -split-compile 4 -arch=sm_90 kernel.cu
```

---

## 6. 浮点与精度选项

### 6.1 完整选项表

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--use_fast_math` | off | **快捷开关**，隐含以下全部 |
| `--ftz={true\|false}` | `false` | 非规格化数（subnormal）刷零 |
| `--prec-div={true\|false}` | `true` | IEEE 精确除法 |
| `--prec-sqrt={true\|false}` | `true` | IEEE 精确开方 |
| `--fmad={true\|false}` | `true` | 启用 FMA 指令融合 |

```bash
# --use_fast_math 等价于:
nvcc --ftz=true --prec-div=false --prec-sqrt=false --fmad=true kernel.cu
```

### 6.2 效果说明

| 选项 | 启用时行为 | 性能影响 | 精度影响 |
|------|-----------|----------|----------|
| `--ftz=true` | subnormal → ±0 | 避免 subnormal 处理开销 | 极小值附近丢失精度 |
| `--prec-div=false` | 使用快速近似除法 | 除法更快 | ~2 ULP 误差 |
| `--prec-sqrt=false` | 使用快速近似开方 | 开方更快 | ~2 ULP 误差 |
| `--fmad=true` | `a*b+c` → FMA 单指令 | 更快且通常更精确 | 改变舍入行为 |
| `--use_fast_math` | 以上全部 + `func()` → `__func()` | 整体显著加速 | 所有 math 函数用近似版 |

### 6.3 LLM 场景建议

| 场景 | 推荐设置 | 理由 |
|------|----------|------|
| 推理 Kernel | `--use_fast_math` | 精度容忍度高，速度优先 |
| Softmax / GELU | 选择性 `__expf` | 对单个函数使用 intrinsic |
| 训练 Kernel | 默认设置 | 梯度累积需要精度保证 |
| FP32 参考实现 | `--ftz=false --prec-div=true` | 最大精度用于验证 |

---

## 7. 调试与诊断选项

### 7.1 调试选项

| 选项 | 说明 | 注意事项 |
|------|------|----------|
| `-G` / `--device-debug` | 生成 Device 调试信息 | **关闭所有 Device 优化**，性能严重下降 |
| `-g` | 生成 Host 调试信息 | 不影响 Device 代码 |
| `--generate-line-info` / `-lineinfo` | 生成 Device 行号信息 | **不影响性能**，推荐用于 profiling |
| `--dopt` | 启用调试模式下的 Device 优化 | 与 `-G` 配合使用 |

```bash
# 性能分析时使用（保留行号映射，不影响优化）
nvcc -lineinfo -arch=sm_90 kernel.cu

# 调试时使用（会禁用优化）
nvcc -G -g -arch=sm_90 kernel.cu

# 有限调试信息 + 保留优化
nvcc -G --dopt on -arch=sm_90 kernel.cu
```

**关键区别**：
- `-G` 会定义 `__CUDACC_DEBUG__` 宏。
- `-G` 覆盖 `-lineinfo`（两者同时指定时，`-G` 优先）。
- **Profile 时只用 `-lineinfo`，绝不用 `-G`**（否则性能数据无意义）。

### 7.2 编译过程诊断

```bash
# 显示编译命令（不执行）
nvcc --dryrun -arch=sm_90 kernel.cu

# 显示并执行编译命令
nvcc --verbose -arch=sm_90 kernel.cu
# 或
nvcc -v -arch=sm_90 kernel.cu

# 保留所有中间文件
nvcc --keep -arch=sm_90 kernel.cu
# 生成: kernel.cpp1.ii, kernel.cudafe1.cpp, kernel.ptx, kernel.cubin, ...

# 同时保留中间文件 + 清理目录
nvcc --keep --keep-dir=./tmp -arch=sm_90 kernel.cu
```

### 7.3 资源使用报告

```bash
# 查看每个 Kernel 的寄存器/内存使用（关键诊断命令）
nvcc -Xptxas=-v -arch=sm_90 kernel.cu 2>&1 | grep "ptxas info"

# 输出示例:
# ptxas info : Compiling entry function '_Z8myKernelPfS_i' for 'sm_90'
# ptxas info : Function properties for _Z8myKernelPfS_i
#     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
# ptxas info : Used 42 registers, 8192 bytes smem, 352 bytes cmem[0]
```

**关键指标解读**：

| 指标 | 含义 | 关注点 |
|------|------|--------|
| `Used N registers` | 每线程寄存器数 | > 64 可能影响 Occupancy |
| `N bytes smem` | 静态 Shared Memory | 加上动态部分不要超过架构限制 |
| `N bytes spill stores/loads` | 寄存器溢出 | **> 0 表示有 spill，需要关注** |
| `N bytes lmem` | Local Memory 使用 | > 0 表示有大数组或 spill |
| `N bytes cmem[0]` | 常量内存（Kernel 参数） | 通常无需关注 |

---

## 8. C++ 语言特性选项

### 8.1 C++ 标准

```bash
# 选择 C++ 方言（默认跟随 Host 编译器）
nvcc --std=c++17 -arch=sm_90 kernel.cu
nvcc --std=c++20 -arch=sm_90 kernel.cu

# CUDA 12.x+ 支持 C++17 和部分 C++20 特性
# CUDA 12.4+ 支持 C++20
```

### 8.2 扩展语言特性

| 选项 | 说明 | 典型用途 |
|------|------|----------|
| `--extended-lambda` | 允许 `__host__ __device__` 注解的 lambda | Thrust/CUB 回调 |
| `--expt-relaxed-constexpr` | Host 调用 `__device__ constexpr`，反之亦然 | 模板元编程 |
| `--expt-extended-lambda` | `--extended-lambda` 的别名 | 同上 |

```bash
# 使用 Device lambda（常见于 Thrust/CUB 编程）
nvcc --extended-lambda -arch=sm_90 kernel.cu

# 宽松 constexpr（Host/Device 间共享 constexpr 函数）
nvcc --expt-relaxed-constexpr -arch=sm_90 kernel.cu
```

启用时定义的宏：
- `--extended-lambda` → 定义 `__CUDACC_EXTENDED_LAMBDA_ENABLED_EXPERIMENTAL__`
- `--expt-relaxed-constexpr` → 定义 `__CUDA_ARCH_RELAXED_CONSTEXPR__`

### 8.3 Stream 模式

```bash
# 默认: Legacy (隐式) Default Stream
# 所有 Host 线程共享同一 Default Stream → 隐式同步

# Per-Thread Default Stream
# 每个 Host 线程有独立 Default Stream → 更好的并发
nvcc --default-stream per-thread -arch=sm_90 kernel.cu

# 等价于在代码中定义（必须在 #include <cuda_runtime.h> 之前）:
# #define CUDA_API_PER_THREAD_DEFAULT_STREAM
```

### 8.4 异常处理

```bash
# 禁用 C++ 异常（减少代码大小，部分嵌入场景使用）
nvcc --no-exceptions -arch=sm_90 kernel.cu
```

---

## 9. Pass-Through 选项

nvcc 允许将选项直接传递给内部工具链的各个组件：

| 选项 | 目标工具 | 常用示例 |
|------|----------|----------|
| `-Xcompiler` | Host 编译器 (gcc/clang/cl) | `-Xcompiler -fPIC -O3 -fopenmp` |
| `-Xptxas` | ptxas (PTX 汇编器) | `-Xptxas=-v,-warn-spills` |
| `-Xlinker` | Host 链接器 | `-Xlinker -rpath=/usr/local/lib` |
| `-Xnvlink` | nvlink (Device 链接器) | `-Xnvlink --verbose` |
| `-Xarchive` | 库管理器 (ar) | — |

```bash
# Host 编译器选项
nvcc -Xcompiler "-fPIC -O3 -Wall" -arch=sm_90 kernel.cu

# ptxas 选项
nvcc -Xptxas="-v,-warn-spills,-maxrregcount=64" -arch=sm_90 kernel.cu

# 链接器选项
nvcc -Xlinker "-rpath,/usr/local/cuda/lib64" -arch=sm_90 kernel.cu -o app

# Host 编译器指定
nvcc -ccbin /usr/bin/g++-12 -arch=sm_90 kernel.cu
```

---

## 10. 分离编译与链接

### 10.1 默认模式：Whole-Program Compilation

默认情况下，nvcc 使用 **整体程序编译模式**：

- 每个 `.cu` 文件中的 Device 代码构成自包含的 Device 程序。
- **不能**跨文件引用 `__device__` 函数或 `__device__` 变量。
- 不需要 Device 链接步骤。

### 10.2 分离编译模式

启用分离编译后，多个 `.cu` 文件可以引用彼此的 Device 符号：

```bash
# 步骤 1: 编译为可重定位 Device 代码对象
nvcc -dc -arch=sm_90 file1.cu -o file1.o
nvcc -dc -arch=sm_90 file2.cu -o file2.o

# 步骤 2: Device 链接
nvcc -dlink -arch=sm_90 file1.o file2.o -o device_link.o

# 步骤 3: Host 链接
g++ file1.o file2.o device_link.o -lcudart -o my_app

# 或者一步完成（nvcc 自动处理 Device 链接）
nvcc -arch=sm_90 file1.o file2.o -o my_app
```

### 10.3 `-dc` vs `-c`

| 选项 | 含义 | Device 代码 |
|------|------|-------------|
| `-c` | 编译为对象文件 | **可执行** Device 代码（自包含） |
| `-dc` | 编译为可重定位 Device 代码对象 | **可重定位** Device 代码（需要 Device 链接） |

`-dc` 等价于 `-c --relocatable-device-code=true`。

### 10.4 分离编译的限制

- 链接的所有对象必须使用**相同的 SM 目标架构**。
- 所有对象必须使用**相同的指针大小**（32 或 64 位）。
- 所有对象必须使用**相同的 ABI 版本**。
- 分离编译会增加少量 overhead（Device 链接 + 函数调用开销）。

### 10.5 何时使用分离编译

| 场景 | 推荐模式 |
|------|----------|
| 单一 Kernel 文件 | 整体编译（默认） |
| 多文件共享 `__device__` 函数 | 分离编译 |
| 需要 `extern __device__` 变量 | 分离编译 |
| Device 代码库（.a 静态库） | 分离编译 |
| 极致性能、最小 overhead | 整体编译 |

---

## 11. NVRTC 运行时编译

### 11.1 概述

**NVRTC (NVIDIA Runtime Compilation)** 是一个库，允许在**运行时**将 CUDA C++ 源码编译为 PTX：

```
       nvcc (离线编译)                    NVRTC (在线编译)
    ┌────────────────────┐           ┌────────────────────┐
    │ .cu → PTX → cubin  │           │ 源码字符串 → PTX    │
    │ (编译时)            │           │ (运行时)            │
    └────────────────────┘           └────────────────────┘
                                              │
                                     CUDA Driver API
                                     cuModuleLoadData()
                                     → JIT: PTX → cubin
```

### 11.2 NVRTC API 流程

```cuda
#include <nvrtc.h>
#include <cuda.h>

const char *kernelSource = R"(
extern "C" __global__ void myKernel(float *data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) data[tid] *= 2.0f;
}
)";

// 1. 创建程序
nvrtcProgram prog;
nvrtcCreateProgram(&prog, kernelSource, "kernel.cu", 0, NULL, NULL);

// 2. 编译为 PTX
const char *opts[] = {"--gpu-architecture=compute_90", "--use_fast_math"};
nvrtcResult result = nvrtcCompileProgram(prog, 2, opts);

// 3. 获取 PTX
size_t ptxSize;
nvrtcGetPTXSize(prog, &ptxSize);
char *ptx = new char[ptxSize];
nvrtcGetPTX(prog, ptx);

// 4. 通过 Driver API 加载并执行
CUmodule module;
CUfunction kernel;
cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
cuModuleGetFunction(&kernel, module, "myKernel");
cuLaunchKernel(kernel, gridDim, 1, 1, blockDim, 1, 1, 0, 0, args, 0);

// 5. 清理
nvrtcDestroyProgram(&prog);
delete[] ptx;
```

### 11.3 NVRTC vs nvcc

| 特性 | nvcc (离线) | NVRTC (在线) |
|------|-----------|-------------|
| 编译时机 | 构建时 | 运行时 |
| 输出 | PTX + cubin (fatbin) | 仅 PTX（由驱动 JIT） |
| API | 命令行工具 | C 库 API |
| Host 代码 | 支持 | **不支持** |
| `--extended-lambda` | 支持 | 不支持 |
| 使用场景 | 常规应用 | 动态代码生成、模板特化 |
| 依赖 | CUDA Toolkit | 仅 NVRTC 库 + Driver |

### 11.4 LLM 场景中的 NVRTC 用途

- **Auto-tuning**：根据运行时参数（tile 大小、数据类型）动态生成特化 Kernel。
- **算子融合**：运行时将多个算子模板拼接并编译为融合 Kernel。
- **PyTorch/Triton 后端**：Triton JIT 编译使用类似机制。

---

## 12. 环境变量与配置

### 12.1 编译环境变量

| 环境变量 | 说明 | 示例 |
|----------|------|------|
| `NVCC_PREPEND_FLAGS` | 在所有命令行选项之前插入 | `export NVCC_PREPEND_FLAGS='-lineinfo'` |
| `NVCC_APPEND_FLAGS` | 在所有命令行选项之后追加 | `export NVCC_APPEND_FLAGS='--use_fast_math'` |
| `NVCC_CCBIN` | 默认 Host 编译器路径 | `export NVCC_CCBIN=/usr/bin/g++-12` |
| `CUDA_PATH` | CUDA Toolkit 安装路径 | `export CUDA_PATH=/usr/local/cuda-12.6` |

```bash
# 全局注入 line info（方便 profiling）
export NVCC_PREPEND_FLAGS='-lineinfo'

# 全局指定 Host 编译器
export NVCC_CCBIN=/usr/bin/g++-12
```

**优先级**：命令行选项 > `NVCC_PREPEND_FLAGS` / `NVCC_APPEND_FLAGS` > 默认值。`--compiler-bindir` 命令行选项 > `NVCC_CCBIN`。

### 12.2 运行时环境变量

| 环境变量 | 说明 |
|----------|------|
| `CUDA_VISIBLE_DEVICES` | 控制可见 GPU 设备 |
| `CUDA_DEVICE_MAX_CONNECTIONS` | 最大并发硬件队列数（默认 8） |
| `CUDA_CACHE_DISABLE` | 禁用 JIT 缓存 |
| `CUDA_CACHE_PATH` | JIT 缓存目录 |
| `CUDA_CACHE_MAXSIZE` | JIT 缓存最大大小 |
| `CUDA_LAUNCH_BLOCKING` | 强制同步 Kernel launch（调试用） |

### 12.3 Response File

当命令行过长时，可使用 options file：

```bash
# 创建 options file
cat > nvcc_opts.txt << 'EOF'
-arch=sm_90
--use_fast_math
-Xptxas=-v
-lineinfo
-O3
-std=c++17
--extended-lambda
EOF

# 引用 options file
nvcc --options-file nvcc_opts.txt kernel.cu -o my_app
```

---

## 13. 实战编译命令速查

### 13.1 开发阶段

```bash
# 快速开发编译（单一架构，带调试行号）
nvcc -arch=sm_90 -lineinfo -O3 -std=c++17 kernel.cu -o app

# 查看资源使用
nvcc -arch=sm_90 -Xptxas=-v kernel.cu -o app 2>&1 | grep ptxas

# 查看生成的 PTX
nvcc --ptx -arch=compute_90 kernel.cu -o kernel.ptx

# 保留中间文件调试
nvcc --keep --keep-dir=./debug_tmp -arch=sm_90 kernel.cu -o app

# 编译命令预览（不执行）
nvcc --dryrun -arch=sm_90 kernel.cu
```

### 13.2 性能优化阶段

```bash
# 最大性能（LLM 推理 Kernel）
nvcc -arch=sm_90 \
     -O3 \
     --use_fast_math \
     -Xptxas=-v,-warn-spills \
     --maxrregcount=128 \
     -lineinfo \
     -std=c++17 \
     --extended-lambda \
     kernel.cu -o app

# 限制寄存器 + 查看 spill
nvcc -arch=sm_90 \
     -O3 \
     --maxrregcount=64 \
     -Xptxas=-v,-warn-spills,-warn-lmem-usage \
     kernel.cu -o app

# Hopper 特有指令（TMA, WGMMA）
nvcc -arch=sm_90a \
     -O3 \
     --use_fast_math \
     -Xptxas=-v \
     kernel.cu -o app
```

### 13.3 生产构建

```bash
# 多架构 fat binary（兼容 A100 到 B200）
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_100,code=sm_100 \
     -gencode arch=compute_100,code=compute_100 \
     -O3 \
     --use_fast_math \
     -lineinfo \
     -std=c++17 \
     -t 8 \
     kernel.cu -o app

# 带动态库链接
nvcc -arch=sm_90 \
     -O3 \
     -Xcompiler "-fPIC" \
     --shared \
     kernel.cu -o libkernel.so
```

### 13.4 分离编译

```bash
# 多文件分离编译
nvcc -dc -arch=sm_90 -O3 -std=c++17 gemm.cu -o gemm.o
nvcc -dc -arch=sm_90 -O3 -std=c++17 attention.cu -o attention.o
nvcc -dc -arch=sm_90 -O3 -std=c++17 layernorm.cu -o layernorm.o
nvcc -arch=sm_90 gemm.o attention.o layernorm.o main.o -o llm_inference
```

### 13.5 调试

```bash
# 调试构建（Device 调试 + Host 调试）
nvcc -G -g -arch=sm_90 -O0 kernel.cu -o app_debug

# 使用 compute-sanitizer 检查
compute-sanitizer --tool memcheck ./app
compute-sanitizer --tool racecheck ./app
compute-sanitizer --tool initcheck ./app

# cuda-gdb 调试
cuda-gdb ./app_debug
```

### 13.6 CMake 集成

```cmake
cmake_minimum_required(VERSION 3.18)
project(llm_kernel LANGUAGES CXX CUDA)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")

# 设置编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v,-warn-spills")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

# 添加库
add_library(kernels STATIC
    gemm.cu
    attention.cu
    layernorm.cu
)

# 设置 per-target 属性
set_target_properties(kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON    # 分离编译
    POSITION_INDEPENDENT_CODE ON     # -fPIC
)

# 寄存器限制（per-target）
target_compile_options(kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=128>
)
```

---

## 参考文献

1. [NVIDIA CUDA Compiler Driver NVCC (v13.2)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
2. [NVCC Documentation PDF (v13.1)](https://docs.nvidia.com/cuda/pdf/CUDA_Compiler_Driver_NVCC.pdf)
3. [CUDA Programming Guide — NVCC Section](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html)
4. [nvcc Compiler Switches — CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/nvcc-compiler-switches.html)
5. [Understanding PTX (NVIDIA Blog)](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing)
6. [Separate Compilation and Linking of CUDA Device Code (NVIDIA Blog)](https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/)
7. [Blackwell & Family-Specific Architecture Features (NVIDIA Blog)](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/)
8. [NVRTC Documentation](https://docs.nvidia.com/cuda/nvrtc/index.html)
9. [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
10. [CUDA Tips: nvcc's -code, -arch, -gencode](https://kaixih.github.io/nvcc-options/)
