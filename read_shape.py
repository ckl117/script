import paddle


if __name__ == "__main__":
    path = "/root/paddlejob/workspace/env_run/output/deepseekv3/0.pdparams"
    state_dict = paddle.load(path)
    for k, v in state_dict.items():
        # if v.dtype == paddle.float8_e4m3fn:
        print(f'{k}: {v.shape}, {v.dtype}')