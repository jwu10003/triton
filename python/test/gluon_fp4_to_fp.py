import os
os.environ["HIP_VISIBLE_DEVICES"] = "4,5"

import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

GROUP_SIZE = 32
PACK_B = 2

torch.manual_seed(0)

@gluon.jit
def fp4_to_fp_kernel(
        b_ptr, c_ptr,
        N, K,
        stride_bn, stride_bk,
        stride_cn, stride_ck,
        BLOCK_SIZE_N: gl.constexpr,
        BLOCK_SIZE_K: gl.constexpr,
        PAD_N: gl.constexpr,
        PAD_K: gl.constexpr,
):
    PACK_B : gl.constexpr = 2
    GROUP_SIZE : gl.constexpr = 32

    blocked_b: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )

    if PAD_N:
        offs_bn = (gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))) % N
    else:
        offs_bn = gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))
    offs_bk = gl.arange(0, BLOCK_SIZE_K // PACK_B, layout=gl.SliceLayout(1, blocked_b))

    b_ptr_offs = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    b = gl.amd.cdna3.buffer_load(b_ptr, b_ptr_offs)
    # b_bf16 = gl.fp4_to_fp(b, gl.bfloat16, 0)
    b_bf16 = gl.fp4_to_fp(b, gl.float32, 0).to(gl.bfloat16)
    b_bf16_layout : gl.constexpr = b_bf16.type.layout

    if PAD_N:
        offs_cn = (gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, b_bf16_layout))) % N
    else:
        offs_cn = gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, b_bf16_layout))
    offs_ck = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, b_bf16_layout))
    gl.amd.cdna3.buffer_store(b_bf16, c_ptr, offs_ck[:, None] * stride_ck + offs_cn[None, :] * stride_cn)


def fp4_to_fp(a, b, bs,
          c = None,
          BLOCK_SIZE_M = 64,
          BLOCK_SIZE_N = 64,
          BLOCK_SIZE_K = 64,
          num_warps=4,
          num_stages=2,
          kpack=1,
          matrix_instr_nonkdim=0,
          a_cache="",
          b_cache="",
          bs_cache=""):
    assert a.shape[1] == b.shape[1] * PACK_B, "Incompatible dimensions"
    assert a.shape[1] == bs.shape[1] * GROUP_SIZE, "Incompatible dimensions"
    assert b.shape[0] == bs.shape[0], "Incompatible dimensions"

    assert a.is_contiguous(), "a must be contiguous"
    assert b.is_contiguous(), "b must be contiguous"
    assert bs.is_contiguous(), "bs must be contiguous"

    M, K = a.shape
    N, _ = b.shape

    c = torch.empty((N, K), device=a.device, dtype=torch.bfloat16)

    # 1D launch kernel where each block gets its own program.
    grid = (1,)
    fp4_to_fp_kernel[grid](
        b, c,
        N, K,                           #
        b.stride(0), b.stride(1),       #
        c.stride(0), c.stride(1),
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_K = BLOCK_SIZE_K,
        PAD_N = N % BLOCK_SIZE_N != 0,
        PAD_K = K % BLOCK_SIZE_K != 0
    )
    print(c[0, :32])
    return c

def generate_test_data(M, N, K):
    # A operand
    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")

    # B operand
    b_mxfp4 = MXFP4Tensor(size=(N, K), device="cuda").random()
    b = b_mxfp4.to_packed_tensor(dim=1)
    b_scale_ref = MXScaleTensor(torch.rand([N, K // GROUP_SIZE], device="cuda") + 1e-8)
    b_scale = b_scale_ref.data

    # ref
    b_scale_ref = b_scale_ref.to(torch.float32).repeat_interleave(GROUP_SIZE, dim=1).contiguous()[:N, :K]
    b_ref = b_mxfp4.to(torch.float32) * b_scale_ref

    return a, b, b_scale, b_ref

def function_test(M, N, K):
    print("test w4a16 with M={} N={} K={}".format(M, N, K))
    a, b, b_scale, b_ref = generate_test_data(M, N, K)
    triton_out = fp4_to_fp(a, b, b_scale)
    # torch_out = torch.matmul(a.to(torch.float32), b_ref.T).to(torch.bfloat16)

    # # Check
    # torch.testing.assert_close(triton_out, torch_out, atol=5e-2, rtol=5e-2)
    # print("✅ Triton and Torch match")

function_test(1, 64, 64)
