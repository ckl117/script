rm -fr build && mkdir build
git submodule update --init --recursive
nvcc -I../third_party/cutlass/include \
-gencode arch=compute_90a,code=sm_90a -O3 -DNDEBUG \
-o ./build/test test.cu