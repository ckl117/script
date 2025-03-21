rm -fr build && mkdir build
nvcc -I/root/paddlejob/workspace/env_run/output/chenkailun/bf16_batch_gemm/PaddleNLP/csrc/third_party/cutlass/include \
-gencode arch=compute_90a,code=sm_90a -O3 -DNDEBUG \
-o ./build/test test.cu