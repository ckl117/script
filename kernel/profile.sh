

export CUDA_VISIBLE_DEVICES=2

export FLAGS_CUTLASS_FP8_GEMM=True
# export FLAGS_use_cutlass_device_best_config_path=/root/paddlejob/workspace/env_run/output/chenkailun/deepseek_work/script/tune/fp8_fuse_gemm_config.json

# source /root/paddlejob/workspace/env_run/output/chenkailun/env/miniconda3/bin/activate /root/paddlejob/workspace/env_run/output/chenkailun/env/miniconda3/envs/deepseek
source /root/paddlejob/workspace/env_run/output/chenkailun/env/miniconda3/bin/activate /root/paddlejob/workspace/env_run/output/chenkailun/env/serving

m_list=(1 8 16 32 128 57600)
# k_list=(7168 1536 512 1024 1152 128)
k_list=(7168 1536)

for m in "${m_list[@]}"
do
    for k in "${k_list[@]}"
    do
        ncu --replay-mode kernel --set full --details-all --cache-control=all --clock-control=base -f -o group_quant_m${m}_k${k} python /root/paddlejob/workspace/env_run/output/chenkailun/deepseek_serving/script/kernel/test_group_quant.py --m ${m} --k ${k}
        ncu --replay-mode kernel --set full --details-all --cache-control=all --clock-control=base -f -o triton_m${m}_k${k} python /root/paddlejob/workspace/env_run/output/chenkailun/deepseek_serving/script/kernel/test_triton_per_token.py --m ${m} --k ${k}
    done
done

