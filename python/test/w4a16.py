import torch
import triton
import triton.language as tl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import time
import numpy as np
import itertools

def get_autotune_config():
    configs = []

    # sizes = [
    #     {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128},
    #     {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128},
    #     {'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 128},
    #     # {'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 128},
    #     {'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128},
    #     {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128},
    #     {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128},
    #     {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 128},
    #     # {'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 128},
    #     # {'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128},
    #     # {'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128},
    #     # {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128},
    #     # {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128},
    #     # {'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 128},
    #     # {'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128},
    #     # {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128},
    #     # {'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128},
    #     # {'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128},
    #     # {'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 128},
    # ]
    # num_warps = [2, 4, 8]
    # num_stages = [1, 2, 3]
    # num_ctas = [1]
    # kpacks = [1, 2]
    # nonkdims = [0, 16, 32]


    sizes = [
        # {'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128},
        # {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128},
        {'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128},
        # {'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128},
        # {'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 128},
    ]
    num_warps = [4]
    num_stages = [2]
    num_ctas = [1]
    kpacks = [1]
    nonkdims = [0]

    all_configs = itertools.product(sizes, num_warps, num_stages, num_ctas, kpacks, nonkdims)
    for idx, (size, warp, stage, num_cta, kpack, nonkdim) in enumerate(all_configs, 1):
        if nonkdim > 0 and (nonkdim > size["BLOCK_M"] or nonkdim > size["BLOCK_N"]):
            continue

        size["kpack"] = kpack
        size["matrix_instr_nonkdim"] = nonkdim
        configs.append(triton.Config(size, num_warps=warp, num_stages=stage, num_ctas=num_cta))

    return configs

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def dot_scale_kernel(a_ptr, stride_am, stride_ak, 
                     b_ptr, stride_bk, stride_bn,
                     bs_ptr, stride_bsn, stride_bsk,
                     c_ptr, stride_cm, stride_cn,
                     M, N, K,
                     PAD_M: tl.constexpr, PAD_N: tl.constexpr, PAD_K: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     type_a: tl.constexpr, type_b: tl.constexpr, GROUP_SIZE: tl.constexpr):
    PACK_B : tl.constexpr = 2

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    if PAD_M:
        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    else:
        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if PAD_N:
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    else:
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_ak = tl.arange(0, BLOCK_K)
    offs_bk = tl.arange(0, BLOCK_K // PACK_B)
    offs_sk = tl.arange(0, BLOCK_K // GROUP_SIZE)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    bs_ptrs = bs_ptr + (offs_bn[:, None] * stride_bsn + offs_sk[None, :] * stride_bsk)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load
        if PAD_K:
            a = tl.load(a_ptrs, mask=offs_ak[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_bk[:, None] < (K - k * BLOCK_K) / PACK_B, other=0.0)
            bs = tl.load(bs_ptrs, mask=offs_sk[None, :] < (K - k * BLOCK_K) / GROUP_SIZE, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            bs = tl.load(bs_ptrs)

        # We accumulate along the K dimension.
        accumulator += tl.dot_scaled(a, None, type_a, b, bs, type_b)

         # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // PACK_B) * stride_bk
        bs_ptrs += (BLOCK_K // GROUP_SIZE) * stride_bsk

    # Store
    c = accumulator.to(tl.float16)
    offs_cm = offs_am
    offs_cn = offs_bn
    c_ptrs = c_ptr +  offs_cm[:, None] * stride_cm +  offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c)

# @gluon.jit
# def gluon_dot_scale_kernel(a_ptr, stride_am, stride_ak, 
#                            b_ptr, stride_bk, stride_bn,
#                            bs_ptr, stride_bsn, stride_bsk,
#                            c_ptr, stride_cm, stride_cn,
#                            M, N, K,
#                            PAD_M: gl.constexpr, PAD_N: gl.constexpr, PAD_K: gl.constexpr,
#                            BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
#                            type_a: gl.constexpr, type_b: gl.constexpr, GROUP_SIZE: gl.constexpr):
#     PACK_B : gl.constexpr = 2

#     pid_m = gl.program_id(axis=0)
#     pid_n = gl.program_id(axis=1)

#     # ----------------------------------------------------------
#     # Block layout
#     a_d2r_layout: gl.constexpr = gl.BlockedLayout( # 16x128xbf16, 128x1
#         size_per_thread=[1, 8],
#         threads_per_warp=[4, 16],
#         warps_per_cta=[4, 1],
#         order=[1, 0],
#     )
#     b_d2r_layout: gl.constexpr = gl.BlockedLayout( # 64x64xu8, 1x64
#         size_per_thread=[16, 1],
#         threads_per_warp=[4, 16],
#         warps_per_cta=[1, 4],
#         order=[0, 1],
#     )
#     bs_d2r_layout: gl.constexpr = gl.BlockedLayout( # 64x4xu8, 4x1
#         size_per_thread=[1, 1],
#         threads_per_warp=[64, 1],
#         warps_per_cta=[1, 4],
#         order=[1, 0],
#     )
#     mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
#         version=3,
#         instr_shape=[16, 16, 16],
#         transposed=False,
#         warps_per_cta=[1, 4],
#     )
#     dot_a_layout: gl.constexpr = gl.DotOperandLayout(
#         operand_index=0, parent=mfma_layout, k_width=16
#     )
#     dot_b_layout: gl.constexpr = gl.DotOperandLayout(
#         operand_index=1, parent=mfma_layout, k_width=16
#     )

#     offs_am = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, a_d2r_layout))
#     offs_bn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, b_d2r_layout))
#     offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, a_d2r_layout))
#     offs_bk = gl.arange(0, BLOCK_K // PACK_B, layout=gl.SliceLayout(0, b_d2r_layout))
#     offs_sk = gl.arange(0, BLOCK_K // GROUP_SIZE, layout=gl.SliceLayout(1, bs_d2r_layout))

#     # ----------------------------------------------------------
#     # Create pointers for the first blocks of A and B.
#     if PAD_M:
#         offs_am = (pid_m * BLOCK_M + gl.arange(0, BLOCK_M)) % M
#     else:
#         offs_am = pid_m * BLOCK_M + gl.arange(0, BLOCK_M)
#     if PAD_N:
#         offs_bn = (pid_n * BLOCK_N + gl.arange(0, BLOCK_N)) % N
#     else:
#         offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
#     offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, a_d2r_layout))
#     offs_bk = gl.arange(0, BLOCK_K // PACK_B, layout=gl.SliceLayout(0, b_d2r_layout))
#     offs_sk = gl.arange(0, BLOCK_K // GROUP_SIZE, layout=gl.SliceLayout(1, bs_d2r_layout))
#     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
#     b_ptrs = b_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
#     bs_ptrs = bs_ptr + (offs_bn[:, None] * stride_bsn + offs_sk[None, :] * stride_bsk)

#     a = gl.buffer_load(a_ptr, a_offs_m * K_A + a_offs_k)

#     # -----------------------------------------------------------
#     # Iterate to compute a block of the C matrix.
#     accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32)
#     for k in range(0, tl.cdiv(K, BLOCK_K)):
#         # Load
#         if PAD_K:
#             a = gl.load(a_ptrs, mask=offs_ak[None, :] < K - k * BLOCK_K, other=0.0)
#             b = gl.load(b_ptrs, mask=offs_bk[:, None] < (K - k * BLOCK_K) / PACK_B, other=0.0)
#             bs = gl.load(bs_ptrs, mask=offs_sk[None, :] < (K - k * BLOCK_K) / GROUP_SIZE, other=0.0)
#         else:
#             a = gl.load(a_ptrs)
#             b = gl.load(b_ptrs)
#             bs = gl.load(bs_ptrs)

#         # We accumulate along the K dimension.
#         accumulator += tl.dot_scaled(a, None, type_a, b, bs, type_b)

#          # Advance the ptrs to the next K block.
#         a_ptrs += BLOCK_K * stride_ak
#         b_ptrs += (BLOCK_K // PACK_B) * stride_bk
#         bs_ptrs += (BLOCK_K // GROUP_SIZE) * stride_bsk

#     # Store
#     c = accumulator.to(tl.float16)
#     offs_cm = offs_am
#     offs_cn = offs_bn
#     c_ptrs = c_ptr +  offs_cm[:, None] * stride_cm +  offs_cn[None, :] * stride_cn
#     tl.store(c_ptrs, c)


class DotScaleKernel:
    def __init__(self, m, n, k):
        self.type_a = "bf16"
        self.type_b = "e2m1"
        self.GROUP_SIZE = 32
        self.m, self.n, self.k = m, n, k
        self.dtype = torch.bfloat16
        assert k % self.GROUP_SIZE == 0, "K must be a multiple of GROUP_SIZE"

        self.generate_test_data()

    def generate_test_data(self):
        # A operand
        self.a = torch.randn((self.m, self.k), dtype=self.dtype, device="cuda")

        # B operand
        b_mxfp4 = MXFP4Tensor(size=(self.n, self.k), device="cuda").random()
        self.b = b_mxfp4.to_packed_tensor(dim=1).T
        b_scale = torch.rand([self.n, self.k // self.GROUP_SIZE], device="cuda")
        b_scale_ref = MXScaleTensor(b_scale)
        self.b_scale = b_scale_ref.data

        # C operand
        self.c = torch.empty((self.m, self.n), dtype=self.dtype, device="cuda")

        # Ref
        # No need to pack along K since we convert each e2m1 to f32 directly for the reference matmul
        self.b_ref = b_mxfp4.to(torch.float32).T
        self.b_scale_ref = b_scale_ref.to(torch.float32).repeat_interleave(self.GROUP_SIZE, dim=1).T.contiguous()[:self.k, :self.n]

    def run(self):
        grid = lambda meta: (
            triton.cdiv(self.m, meta['BLOCK_M']),
            triton.cdiv(self.n, meta['BLOCK_N']),
        )

        # Launch kernel with proper grid and dimension parameters
        dot_scale_kernel[grid](
            self.a, *self.a.stride(),                   # a
            self.b, *self.b.stride(),                   # b
            self.b_scale, *self.b_scale.stride(),       # bs
            self.c, *self.c.stride(),                   # c
            self.m, self.n, self.k,                     # Pass dimensions
            PAD_M = self.m % 128 != 0,
            PAD_N = self.n % 128 != 0,
            PAD_K = self.k % 128 != 0,
            GROUP_SIZE = self.GROUP_SIZE,
            type_a = self.type_a,
            type_b = self.type_b,
        )

    def verify(self):
        # Ref
        ref_out = torch.matmul(self.a.to(torch.float32), self.b_ref * self.b_scale_ref).to(self.dtype)

        # Triton
        self.run()

        # Check
        torch.testing.assert_close(self.c, ref_out, atol=5e-2, rtol=5e-2)


    def mem(self):
        return self.a.numel() * self.a.element_size() + \
               self.b.numel() * self.b.element_size() + \
               self.b_scale.numel() * self.b_scale.element_size() + \
               self.c.numel() * self.c.element_size()

    def flops(self):
        return 2.0 * self.m * self.n * self.k


# # %%
# # Benchmark
# config = triton.testing.Benchmark(x_names=["M", "N", "K"],
#                                   x_vals=[(8192, 8192, 8192)],
#                                   line_arg="metric",
#                                   line_vals=["time", "tflops", "bandwidth"],
#                                   line_names=["Time(ms)", "TFLOPS", "Bandwidth(GB/s)"],
#                                   styles=[("red", "-"), ("blue", "-"), ("yellow", "-")],
#                                   ylabel="",
#                                   plot_name="w4a16 matmul performance",
#                                   args={})

# @triton.testing.perf_report([config])
# def benchmark(M, N, K, metric):
#     dot = DotScaleKernel(M, N, K)
#     quantiles = [0.5, 0.2, 0.8]
#     ms, min_ms, max_ms = triton.testing.do_bench(lambda: dot.run(), quantiles=quantiles, warmup=5, rep=20)

#     flops = dot.flops()
#     mem = dot.mem()
#     tflops = flops / (ms * 1e-3) / 1e12 # TFLOPS
#     bandwidth = mem / (ms * 1e-3) / 1e9 # GB/s

#      # Return exactly one scalar depending on which metric is active
#     if metric == "time":
#         return ms
#     elif metric == "tflops":
#         return tflops
#     elif metric == "bandwidth":
#         return bandwidth
#     else:
#         raise ValueError("Unknown metric: " + metric)

# benchmark.run(show_plots=False, print_data=True)


cases = [
    # [15, 4096, 4096],
    # [15, 4096, 14336],
    # [15, 6144, 4096],
    # [15, 8192, 8192],
    # [15, 8192, 28672],
    # [15, 10240, 8192],
    # [15, 28672, 4096],
    [15, 57344, 8192],
    # [44, 1280, 8192],
    # [44, 4096, 4096],
    # [44, 4096, 14336],
    # [44, 6144, 4096],
    # [44, 7168, 8192],
    # [44, 8192, 1024],
    # [44, 8192, 3584],
    # [44, 8192, 8192],
    # [44, 8192, 28672],
    # [44, 10240, 8192],
    # [44, 28672, 4096],
    # [44, 57344, 8192],
    # [566, 1280, 8192],
    # [566, 7168, 8192],
    # [566, 8192, 1024],
    # [566, 8192, 3584],
    # [582, 4096, 4096],
    # [582, 4096, 14336],
    # [582, 6144, 4096],
    # [582, 28672, 4096],
    # [611, 4096, 4096],
    # [611, 4096, 14336],
    # [611, 6144, 4096],
    # [611, 28672, 4096],
    # [874, 8192, 8192],
    # [874, 8192, 28672],
    # [874, 10240, 8192],
    # [874, 57344, 8192],
    # [932, 8192, 8192],
    # [932, 8192, 28672],
    # [932, 10240, 8192],
    # [932, 57344, 8192],
    # [1003, 8192, 8192],
    # [1003, 8192, 28672],
    # [1003, 10240, 8192],
    # [1003, 57344, 8192],
    # [1324, 4096, 4096],
    # [1324, 4096, 14336],
    # [1324, 6144, 4096],
    # [1324, 28672, 4096],
    # [1340, 1280, 8192],
    # [1340, 7168, 8192],
    # [1340, 8192, 1024],
    # [1340, 8192, 3584],
    # [1466, 1280, 8192],
    # [1466, 7168, 8192],
    # [1466, 8192, 1024],
    # [1466, 8192, 3584],
    # [1906, 1280, 8192],
    # [1906, 7168, 8192],
    # [1906, 8192, 1024],
    # [1906, 8192, 3584],
    # [2084, 8192, 8192],
    # [2084, 8192, 28672],
    # [2084, 10240, 8192],
    # [2084, 57344, 8192],
    # [4314, 4096, 4096],
    # [4314, 4096, 14336],
    # [4314, 6144, 4096],
    # [4314, 28672, 4096],
    # [14437, 4096, 4096],
    # [14437, 4096, 14336],
    # [14437, 6144, 4096],
    # [14437, 28672, 4096],
    # [15961, 1280, 8192],
    # [15961, 7168, 8192],
    # [15961, 8192, 1024],
    # [15961, 8192, 3584],
    # [16375, 8192, 8192],
    # [16375, 8192, 28672],
    # [16375, 10240, 8192],
    # [16375, 57344, 8192],
]

id = 0
for case in cases:
    print("test {}/{} {}...".format(id, len(cases), case))
    id += 1
    # %%
    # Benchmark
    config = triton.testing.Benchmark(x_names=["M", "N", "K"],
                                    x_vals=[case],
                                    line_arg="metric",
                                    line_vals=["time", "tflops", "bandwidth"],
                                    line_names=["Time(ms)", "TFLOPS", "Bandwidth(GB/s)"],
                                    styles=[("red", "-"), ("blue", "-"), ("yellow", "-")],
                                    ylabel="",
                                    plot_name="w4a16 matmul performance",
                                    args={})

    @triton.testing.perf_report([config])
    def benchmark(M, N, K, metric):
        dot = DotScaleKernel(M, N, K)

        # # Verify correctness
        # dot.verify()

        # Benchmark
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: dot.run(), quantiles=quantiles, warmup=2, rep=10)

        flops = dot.flops()
        mem = dot.mem()
        tflops = flops / (ms * 1e-3) / 1e12 # TFLOPS
        bandwidth = mem / (ms * 1e-3) / 1e9 # GB/s

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return ms
        elif metric == "tflops":
            return tflops
        elif metric == "bandwidth":
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    benchmark.run(show_plots=False, print_data=True)
