

# proj, N, K 非吸收版本
# q_a, 1536, 7168; 56;
# q_b, 1536, 1536; 12;
# kv_a, 576, 7168; 56;
# kv_b, 2048, 512; 4;
# q_nope_k_b, 4096, 1536
# q_rope, 512, 1536
# v_b_proj_o, 7168, 4096
# outlinear, 7168, 1024; 8
# ffn1, 2304, 7168 # M=9344 调优报错，手动指定配置; 56;
# ffn2, 7168, 1152; 9;
# shared_ffn1, 256, 7168; 56;
# shared_ffn2, 7168, 128; 1

# export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
# export PATH=/usr/local/cuda-12.3/bin:$PATH
export CUDA_VISIBLE_DEVICES=2
export FLAGS_use_cutlass_device_best_config_path=tune
source /root/paddlejob/workspace/env_run/output/chenkailun/env/miniconda3/bin/activate /root/paddlejob/workspace/env_run/output/chenkailun/env/miniconda3/envs/serving
# all 14490

# python ./tune_cutlass_fp8_block_gemm.py \
#         --m_min 9344 \
#         --m_max 9344 \
#         --n 2304 \
#         --k 7168 \
#         >  fp8_tune_log.log 2>&1 

python ./tune_cutlass_fp8_block_gemm.py \
        --m_min 9376 \
        --m_max 32768 \
        --n 1536 1536 576 2048 4096 512 7168 7168 2304 7168 256 7168 \
        --k 7168 1536 7168 512 1536 1536 4096 1024 7168 1152 7168 128 \
        >  fp8_tune_log.log 2>&1 


# export CUDA_VISIBLE_DEVICES=7
# nohup python ./tune_cutlass_fp8_block_gemm.py \
#         --m_min 32 \
#         --m_max 32768 \
#         --n 4096 \
#         --k 1536 \
#         >  tune_fp8_gemm_q_nope_k_b.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=6
# nohup python ./tune_cutlass_fp8_block_gemm.py \
#         --m_min 32 \
#         --m_max 32768 \
#         --n 512 \
#         --k 1536 \
#         >  tune_fp8_gemm_q_rope.log 2>&1 &


# export CUDA_VISIBLE_DEVICES=5
# nohup python ./tune_cutlass_fp8_block_gemm.py \
#         --m_min 32 \
#         --m_max 32768 \
#         --n 7168 \
#         --k 4096 \
#         >  tune_fp8_gemm_v_b_proj_o.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=6
# nohup python ./tune_cutlass_fp8_block_gemm.py \
#         --m_min 32 \
#         --m_max 32768 \
#         --n 1536 \
#         --k 1536 \
#         >  tune_fp8_gemm_q_b.log 2>&1 &


# export CUDA_VISIBLE_DEVICES=5
# nohup python ./tune_cutlass_fp8_block_gemm.py \
#         --m_min 32 \
#         --m_max 32768 \
#         --n 576 \
#         --k 7168 \
#         >  tune_fp8_gemm_kv_a.log 2>&1 &


# export CUDA_VISIBLE_DEVICES=4
# nohup python ./tune_cutlass_fp8_block_gemm.py \
#         --m_min 32 \
#         --m_max 32768 \
#         --n 2048 \
#         --k 512 \
#         >  tune_fp8_gemm_kv_b.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=3
# nohup python ./tune_cutlass_fp8_block_gemm.py \
#         --m_min 32 \
#         --m_max 32768 \
#         --n 7168 \
#         --k 1024 \
#         >  tune_fp8_gemm_outlinear.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# nohup python ./tune_cutlass_fp8_block_gemm.py \
#         --m_min 32 \
#         --m_max 32768 \
#         --n 2304 \
#         --k 7168 \
#         >  tune_fp8_gemm_ffn1.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# nohup python ./tune_cutlass_fp8_block_gemm.py \
#         --m_min 32 \
#         --m_max 32768 \
#         --n 7168 \
#         --k 1152 \
#         >  tune_fp8_gemm_ffn2.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python ./tune_cutlass_fp8_block_gemm.py \
#         --m_min 32 \
#         --m_max 32768 \
#         --n 256 \
#         --k 7168 \
#         >  tune_fp8_gemm_shared_ffn1.log 2>&1 &


# export CUDA_VISIBLE_DEVICES=0
# nohup python ./tune_cutlass_fp8_block_gemm.py \
#         --m_min 32 \
#         --m_max 32768 \
#         --n 7168 \
#         --k 128\
#         >  tune_fp8_gemm_shared_ffn2.log 2>&1 &