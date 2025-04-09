#!/bin/bash

CUDA_ARCH=${CUDA_ARCH:-89}

nvcc \
    -arch=sm_${CUDA_ARCH} \
    -O3 \
    --use_fast_math \
    -ptx \
    -rdc=true \
    --compiler-options -fPIC \
    src/utils/compute_forces.cu \
    -o src/utils/compute_forces.ptx

chmod 644 src/utils/compute_forces.ptx