#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

/**
 * CUDA kernel for performing AND operation between two bitmap slices and counting set bits
 * Each thread processes one or more words from the bitmap slices
 */
extern "C" __global__ void bsi_slice_and_popcount_kernel(
    const uint64_t* slice1,
    const uint64_t* slice2,
    size_t word_count,
    unsigned long long* result) 
{
    extern __shared__ unsigned long long shared_counts[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned long long count = 0;
    // Each thread processes one or more words
    while (idx < word_count) {
        uint64_t word = slice1[idx] & slice2[idx];
        count += __popcll(word); // Count bits set in the word using CUDA intrinsic
        idx += blockDim.x * gridDim.x;
    }
    
    // Store in shared memory for reduction
    shared_counts[tid] = count;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_counts[tid] += shared_counts[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        result[blockIdx.x] = shared_counts[0];
    }
}

/**
 * CUDA kernel to perform final reduction of partial results
 */
extern "C" __global__ void bsi_reduce_results_kernel(
    unsigned long long* partial_results,
    int count,
    unsigned long long* final_result)
{
    extern __shared__ unsigned long long shared_results[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load into shared memory
    unsigned long long sum = 0;
    if (idx < count) {
        sum = partial_results[idx];
    }
    shared_results[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_results[tid] += shared_results[tid + s];
        }
        __syncthreads();
    }
    
    // Write final result
    if (tid == 0) {
        final_result[blockIdx.x] = shared_results[0];
    }
}
