#include "bsi_dot_cuda_wrapper.h"
#include "hybridbitmap.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <type_traits> 
#include <algorithm>

// Global variable defined in bsi_dot_cuda.cpp to store the number of blocks from the last kernel launch
extern int g_last_kernel_num_blocks;   

// Forward declarations of CUDA kernel functions
extern "C" void bsi_slice_and_popcount_kernel(
    const uint64_t* slice1,
    const uint64_t* slice2,
    size_t word_count,
    unsigned long long* result);

extern "C" void bsi_reduce_results_kernel(
    unsigned long long* partial_results,
    int count,
    unsigned long long* final_result);

// Helper function to check CUDA errors
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
}

bool is_cuda_available() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        return false;
    }
    
    // Try to allocate a small amount of memory to verify CUDA is working
    void* testMem = nullptr;
    error = cudaMalloc(&testMem, 1024);
    
    if (error != cudaSuccess) {
        return false;
    }
    
    cudaFree(testMem);
    return true;
}

template <class uword>
long long int bsi_dot_product_cuda(
    const std::vector<HybridBitmap<uword>>& bsi1_slices, 
    const std::vector<HybridBitmap<uword>>& bsi2_slices)
{
    if constexpr (!std::is_same<uword, uint64_t>::value) {
        throw std::runtime_error("CUDA BSI dot product currently only supports uint64_t");
    }

    int bsi1_size = bsi1_slices.size();
    int bsi2_size = bsi2_slices.size();

    if (bsi1_size == 0 || bsi2_size == 0) {
        return 0; // Nothing to do
    }

    // Ensure all slices have the same buffer size and capture it
    size_t word_count = bsi1_slices[0].bufferSize();
    std::cout << "Word count:" << word_count << std::endl;
    for (const auto &s : bsi1_slices) {
        if (s.bufferSize() != word_count) {
            throw std::runtime_error("All slices in first BSI must have the same buffer size");
        }
    }
    for (const auto &s : bsi2_slices) {
        if (s.bufferSize() != word_count) {
            throw std::runtime_error("All slices in second BSI must have the same buffer size");
        }
    }
    if (word_count == 0) {
        return 0;
    }

    // Configure kernel launch (256-thread blocks, up to 65k blocks)
    const int blockSize = 140;
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // const int blockSize = prop.maxThreadsPerBlock;
    int numBlocks = static_cast<int>((word_count + blockSize - 1) / blockSize);
    numBlocks = std::min(numBlocks, 65535);
    g_last_kernel_num_blocks = numBlocks;

    // Allocate device buffers ONCE and reuse them for all slice pairs
    uint64_t *d_slice1 = nullptr, *d_slice2 = nullptr;
    unsigned long long *d_block_results = nullptr, *d_final_result = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_slice1, word_count * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_slice2, word_count * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_block_results, numBlocks * sizeof(unsigned long long)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_final_result, sizeof(unsigned long long)));

    dim3 grid(numBlocks, 1, 1);
    dim3 block(blockSize, 1, 1);
    const size_t sharedMem = blockSize * sizeof(unsigned long long);

    long long int result = 0;

    // Iterate through each pair of slices
    for (int i = 0; i < bsi1_size; ++i) {
        for (int j = 0; j < bsi2_size; ++j) {
            // Copy current slices into the pre-allocated device buffers
            CHECK_CUDA_ERROR(cudaMemcpy(d_slice1,
                                         bsi1_slices[i].getBuffer().data(),
                                         word_count * sizeof(uint64_t),
                                         cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_slice2,
                                         bsi2_slices[j].getBuffer().data(),
                                         word_count * sizeof(uint64_t),
                                         cudaMemcpyHostToDevice));

            void *kernel_args[] = { &d_slice1, &d_slice2, &word_count, &d_block_results };
            CHECK_CUDA_ERROR(cudaLaunchKernel(reinterpret_cast<void*>(bsi_slice_and_popcount_kernel),
                                              grid, block, kernel_args, sharedMem, nullptr));
            CHECK_CUDA_ERROR(cudaGetLastError());

            unsigned long long count = 0;

            if (numBlocks > 1) {
                // Launch reduction kernel when multiple blocks are used
                void *reduce_args[] = { &d_block_results, &numBlocks, &d_final_result };
                dim3 reduce_grid(1, 1, 1);
                dim3 reduce_block(256, 1, 1);
                const size_t reduce_shared = 256 * sizeof(unsigned long long);

                CHECK_CUDA_ERROR(cudaLaunchKernel(reinterpret_cast<void*>(bsi_reduce_results_kernel),
                                                  reduce_grid, reduce_block, reduce_args, reduce_shared, nullptr));
                CHECK_CUDA_ERROR(cudaGetLastError());
                CHECK_CUDA_ERROR(cudaMemcpy(&count, d_final_result,
                                            sizeof(unsigned long long),
                                            cudaMemcpyDeviceToHost));
            } else {
                // Single block result is already in d_block_results[0]
                CHECK_CUDA_ERROR(cudaMemcpy(&count, d_block_results,
                                            sizeof(unsigned long long),
                                            cudaMemcpyDeviceToHost));
            }

            // Apply power-of-two scaling based on slice indices
            if (i == 0 && j == 0) {
                result += count;
            } else {
                result += count * (2LL << (i + j - 1));
            }
        }
    }

    // Clean-up: free device memory that was allocated once
    cudaFree(d_slice1);
    cudaFree(d_slice2);
    cudaFree(d_block_results);
    cudaFree(d_final_result);

    return result;
}

// Explicit template instantiations
template long long int bsi_dot_product_cuda<uint64_t>(
    const std::vector<HybridBitmap<uint64_t>>& bsi1_slices, 
    const std::vector<HybridBitmap<uint64_t>>& bsi2_slices);
