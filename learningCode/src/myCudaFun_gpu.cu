#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

#include "myCudaFun_gpu.h"
#include "device_launch_parameters.h"

__global__ void add_kernel_fast(int n, const float * data1, const float *data2, float * data3) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= n) return;
    // printf("current idx: %d\n", id);
    data3[id] = data1[id] + data2[id];
    // printf("idx: %d, %f + %f = %f\n", id, data1[id], data2[id], data3[id]);
}

__global__ void run_on_gpu() {
	printf("GPU thread info X:%d Y:%d Z:%d\t block info X:%d Y:%d Z:%d\n",
		threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
}

// void add_kernerl_launcher_fast(int n, const float *data1, const float *data2, float *data3, cudaStream_t stream) {
void add_kernerl_launcher_fast(int n, const float *data1, const float *data2, float *data3) {
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK));  // blockIdx.x(col),  blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    // printf("number of tensor size: %d\n", n);
    add_kernel_fast<<<blocks, threads>>>(n, data1, data2, data3);
    // run_on_gpu<<<blocks, threads, 0>>>();
    cudaDeviceSynchronize();

    cudaError_t err;
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}