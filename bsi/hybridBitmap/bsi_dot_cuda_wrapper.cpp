#include "bsi_dot_cuda_wrapper.h"
#include "hybridbitmap.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

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
    // Currently only supports uint64_t
    if constexpr (!std::is_same<uword, uint64_t>::value) {
        throw std::runtime_error("CUDA BSI dot product currently only supports uint64_t");
    }

    int bsi1_size = bsi1_slices.size();
    int bsi2_size = bsi2_slices.size();
    long long int result = 0;

    // Iterate through each pair of slices
    for (int i = 0; i < bsi1_size; i++) {
        for (int j = 0; j < bsi2_size; j++) {
            // Get the buffer sizes
            size_t word_count = bsi1_slices[i].bufferSize();
            if (word_count != bsi2_slices[j].bufferSize()) {
                throw std::runtime_error("BSI slices have different buffer sizes");
            }

            // Skip if either buffer is empty
            if (word_count == 0) continue;

            // Allocate device memory
            uint64_t *d_slice1, *d_slice2;
            unsigned long long *d_block_results, *d_final_result;
            
            CHECK_CUDA_ERROR(cudaMalloc(&d_slice1, word_count * sizeof(uint64_t)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_slice2, word_count * sizeof(uint64_t)));

            // Copy data to device
            CHECK_CUDA_ERROR(cudaMemcpy(d_slice1, bsi1_slices[i].getBuffer().data(), 
                                       word_count * sizeof(uint64_t), 
                                       cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_slice2, bsi2_slices[j].getBuffer().data(), 
                                       word_count * sizeof(uint64_t), 
                                       cudaMemcpyHostToDevice));

            // Configure kernel
            int blockSize = 256;
            int numBlocks = (word_count + blockSize - 1) / blockSize;
            numBlocks = std::min(numBlocks, 1024); // Cap at reasonable number
            
            // Allocate memory for partial results
            CHECK_CUDA_ERROR(cudaMalloc(&d_block_results, numBlocks * sizeof(unsigned long long)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_final_result, sizeof(unsigned long long)));
            
            // Launch kernel to count bits using cudaLaunchKernel instead of <<< >>> syntax
            void* args[] = { &d_slice1, &d_slice2, &word_count, &d_block_results };
            dim3 grid(numBlocks, 1, 1);
            dim3 block(blockSize, 1, 1);
            size_t sharedMem = blockSize * sizeof(unsigned long long);
            
            CHECK_CUDA_ERROR(cudaLaunchKernel((void*)bsi_slice_and_popcount_kernel, 
                                            grid, block, args, sharedMem, nullptr));
            
            // Check for errors
            CHECK_CUDA_ERROR(cudaGetLastError());
            
            // Reduce results if we have multiple blocks
            if (numBlocks > 1) {
                void* reduce_args[] = { &d_block_results, &numBlocks, &d_final_result };
                dim3 reduce_grid(1, 1, 1);
                dim3 reduce_block(256, 1, 1);
                size_t reduce_sharedMem = 256 * sizeof(unsigned long long);
                
                CHECK_CUDA_ERROR(cudaLaunchKernel((void*)bsi_reduce_results_kernel,
                                                reduce_grid, reduce_block, reduce_args, reduce_sharedMem, nullptr));
                
                CHECK_CUDA_ERROR(cudaGetLastError());
            }
            
            // Copy result back to host
            unsigned long long count;
            if (numBlocks > 1) {
                CHECK_CUDA_ERROR(cudaMemcpy(&count, d_final_result, 
                                          sizeof(unsigned long long), 
                                          cudaMemcpyDeviceToHost));
            } else {
                CHECK_CUDA_ERROR(cudaMemcpy(&count, d_block_results, 
                                          sizeof(unsigned long long), 
                                          cudaMemcpyDeviceToHost));
            }
            
            // Apply power-of-2 scaling factor based on slice positions
            if (i == 0 && j == 0) {
                result += count;
            } else {
                result += count * (2LL << (i + j - 1));
            }
            
            // Free device memory
            cudaFree(d_slice1);
            cudaFree(d_slice2);
            cudaFree(d_block_results);
            cudaFree(d_final_result);
        }
    }
    
    return result;
}

// Explicit template instantiations
template long long int bsi_dot_product_cuda<uint64_t>(
    const std::vector<HybridBitmap<uint64_t>>& bsi1_slices, 
    const std::vector<HybridBitmap<uint64_t>>& bsi2_slices);
