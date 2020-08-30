#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <math.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "myCudaFun_gpu.h"

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

extern THCState *state;

void add_wrapper_fast(int n, at::Tensor tensorA, at::Tensor tensorB, at::Tensor tensorC) {
    CHECK_CUDA(tensorA)
    CHECK_CUDA(tensorB)
    CHECK_CUDA(tensorC)

    const float *data1 = tensorA.data<float>();
    const float *data2 = tensorB.data<float>();

    float *data3 = tensorC.data<float>();
    // fprintf(stderr, "data1: %f", *data1);
    // cudaStream_t stream = THCState_getCurrentStream(state);
    // for (int idx = 0; idx < n; ++idx) {
    //     fprintf(stderr, "%f %f %f\n", data1[idx], data2[idx], data3[idx]);
    // }
    add_kernerl_launcher_fast(n, data1, data2, data3);
}