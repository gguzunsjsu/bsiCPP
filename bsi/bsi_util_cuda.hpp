#ifndef BSI_UTIL_CUDA_HPP
#define BSI_UTIL_CUDA_HPP

#include <cuda_runtime.h>
#include <iostream>
#include <string> // Required for cudaGetErrorString

// Macro to check CUDA errors
// This macro will print an error message and exit if a CUDA API call fails.
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                  << " - Code: " << err_ << " (" << cudaGetErrorString(err_) \
                  << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#endif //BSI_UTIL_CUDA_HPP
