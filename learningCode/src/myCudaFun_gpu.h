#ifndef _MYCUDAFUN_GPU_H
#define _MYCUDAFUN_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

void add_wrapper_fast(int n, at::Tensor tensorA, at::Tensor tensorB, at::Tensor tensorC);

void add_kernerl_launcher_fast(int n, const float *data1, const float *data2, float *data3);
#endif