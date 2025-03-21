import paddle
import os
import numpy as np
import time

import triton
import triton.language as tl

from paddlenlp_ops import group_quant

from typing import Any, Dict, List, Optional, Tuple

import argparse

@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Collums of input
    N,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.

    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * group_size
    y_q_ptr += g_id * group_size

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr += scale_col * y_s_col_stride + scale_row

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: paddle.Tensor,
    group_size: int,
    eps: float = 1e-6,
    dtype: str = "float8_e4m3fn",
    column_major_scales: bool = False,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tenosr with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `paddle.float8_e4m3fn` is supported for now.

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    # finfo = paddle.finfo(dtype)
    fp8_max = 448.0
    fp8_min = -fp8_max

    x_q = paddle.empty_like(x, dtype=dtype)
    M = x.numel() // group_size
    N = group_size
    if column_major_scales:
        x_s = paddle.empty(
            [x.shape[-1] // group_size] + x.shape[:-1],
            dtype=paddle.float32,
        )
        x_s = x_s.transpose((-1, -2))
    else:
        x_s = paddle.empty(
            x.shape[:-1] + [x.shape[-1] // group_size],
            dtype=paddle.float32,
        )

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    if column_major_scales:
        _per_token_group_quant_fp8_colmajor[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x_s.strides[1],
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _per_token_group_quant_fp8[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            N,
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return x_q, x_s


def setup_args():
    """Setup export arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, help="range of gemm shape: m_min")
    parser.add_argument("--k", type=int, help="range of gemm shape: m_max")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = setup_args()

    M = args.m
    K = args.k

    x_tensor = paddle.randn(shape=[M, K], dtype="float32").cast(paddle.bfloat16)
    warm_up = 5
    test_time = 1

    
    for i in range(warm_up):
        x_q_ref, x_s_ref = per_token_group_quant_fp8(
            x_tensor, group_size=128, eps=0.000001, dtype="float8_e4m3fn", column_major_scales=True,
        )
    # paddle.device.synchronize()
    # start_time = time.time()
    for i in range(test_time):
        x_q_ref, x_s_ref = per_token_group_quant_fp8(
            x_tensor, group_size=128, eps=0.000001, dtype="float8_e4m3fn", column_major_scales=True,
        )
    # paddle.device.synchronize()
    # end_time = time.time()
    # avg_time = (end_time - start_time) / test_time
    # print(f'triton avg time: {avg_time}')

    # for i in range(warm_up):
    #     x_q, x_s = group_quant(
    #         x_tensor, group_size=128, transpose_scale=True, quant_max_bound=448.0, quant_min_bound=-448.0
    #     )
    
    # paddle.device.synchronize()
    # start_time = time.time()
    # for i in range(test_time):
    #     x_q, x_s = group_quant(
    #         x_tensor, group_size=128, transpose_scale=True, quant_max_bound=448.0, quant_min_bound=-448.0
    #     )
    # paddle.device.synchronize()
    # end_time = time.time()
    # avg_time = (end_time - start_time) / test_time
    # print(f'groupquant avg time: {avg_time}')

    # x_q_ref = x_q_ref.cast(paddle.float32)
    # x_q = x_q.cast(paddle.float32)
    # np.testing.assert_allclose(x_q.numpy(), x_q_ref.numpy(), rtol=1e-2, atol=1e-2)
    # np.testing.assert_allclose(x_s.numpy(), x_s_ref.numpy(), rtol=1e-2, atol=1e-2)