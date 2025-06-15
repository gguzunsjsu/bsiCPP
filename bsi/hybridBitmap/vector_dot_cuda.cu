#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <vector>

extern "C" __global__ void vector_dot_kernel(const long* __restrict__ vec1,
                                              const long* __restrict__ vec2,
                                              size_t n,
                                              long long* partial) {
    extern __shared__ long long sdata[];
    const int tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    long long local = 0;
    while (idx < n) {
        local += static_cast<long long>(vec1[idx]) * static_cast<long long>(vec2[idx]);
        idx += blockDim.x * gridDim.x;
    }
    sdata[tid] = local;
    __syncthreads();
    // reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

extern "C" __global__ void reduce_long_kernel(long long* partial, int count) {
    extern __shared__ long long sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long val = (idx < count) ? partial[idx] : 0;
    sdata[tid] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}
