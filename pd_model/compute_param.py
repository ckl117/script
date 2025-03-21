import paddle
import numpy as np
import os


if __name__ == "__main__":
    path = "/root/paddlejob/workspace/env_run/output/model/split/r1_fp8/0.pdparams"
    model_state = paddle.load(path, return_numpy=True)
    print(type(model_state))
    total_param = 0
    size = 0
    key_list = ["q_nope_k_b_proj_weight", "q_rope_proj_weight", "v_b_proj_o_weight", "k_b_proj_weight"]
    for k,v in model_state.items():
        flag = False
        for key in key_list:
            if key in k:
                flag = True
                break
        if not flag:
            continue
        count = 1
        for i in v.shape:
            count*=i
        print(f'{k}, {v.shape}, {count}, {v.dtype}')
        total_param += count
        if v.dtype == np.uint16:
            size += count*2
            # print(f'v.dtype == np.uint16 = {v.dtype == np.uint16}')
        elif v.dtype == np.float32:
            size += count*4
            # print(f'v.dtype == np.float32 = {v.dtype == np.float32}')
        else:
            size += count
            # print(f'v.dtype == np.float8')
    print(f'total_param = {total_param/1000000000} B')
    print(f'size = {size/4*161/1024/1024/1024}GB')
