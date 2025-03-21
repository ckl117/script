import paddle
import os
import numpy as np
import time


from paddlenlp_ops import group_quant

import argparse


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
        x_q, x_s = group_quant(
            x_tensor, group_size=128, transpose_scale=True, quant_max_bound=448.0, quant_min_bound=-448.0
        )
    
    # paddle.device.synchronize()
    start_time = time.time()
    for i in range(test_time):
        x_q, x_s = group_quant(
            x_tensor, group_size=128, transpose_scale=True, quant_max_bound=448.0, quant_min_bound=-448.0
        )
    # paddle.device.synchronize()
    # end_time = time.time()
    # avg_time = (end_time - start_time) / test_time
    # print(f'groupquant avg time: {avg_time}')

    # x_q_ref = x_q_ref.cast(paddle.float32)
    # x_q = x_q.cast(paddle.float32)
    # np.testing.assert_allclose(x_q.numpy(), x_q_ref.numpy(), rtol=1e-2, atol=1e-2)
    # np.testing.assert_allclose(x_s.numpy(), x_s_ref.numpy(), rtol=1e-2, atol=1e-2)