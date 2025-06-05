#include "bsi_dot_cuda.h"
#include "bsi_dot_cuda_wrapper.h"
#include "../BsiAttribute.hpp"
#include "../BsiUnsigned.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
}
#endif

// Keep last kernel elapsed milliseconds in a static var
static float g_last_kernel_ms = 0.0f;
// Keep last kernel launched blocks in a global (extern in header)
int g_last_kernel_num_blocks = 0;

float cuda_last_kernel_time_ms() { return g_last_kernel_ms; }

void cuda_print_device_info() {
#ifdef USE_CUDA
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::cout << "No CUDA devices present.\n"; return; }
    cudaGetDeviceProperties(&prop, 0);
    int coresPerSM;
    switch (prop.major * 10 + prop.minor) {
        case 90: coresPerSM = 128; break; 
        case 86: coresPerSM = 128; break; 
        case 80: coresPerSM = 64;  break; 
        default: coresPerSM = 64;  break; }
    int totalCores = coresPerSM * prop.multiProcessorCount;
    std::cout << "CUDA Device: " << prop.name << " | SMs: " << prop.multiProcessorCount
              << " | CUDA cores: " << totalCores << " | Max threads/SM: "
              << prop.maxThreadsPerMultiProcessor << std::endl;
#else
    std::cout << "CUDA not available." << std::endl;
#endif
}

// Implementation of cuda_dot_available() - declared in bsi_dot_cuda.h
bool cuda_dot_available() {
#ifdef USE_CUDA
    return is_cuda_available();
#else
    return false;
#endif
}

template <class uword>
long long int bsi_dot_cuda(const BsiAttribute<uword>* bsi1, const BsiAttribute<uword>* bsi2) {
#ifdef USE_CUDA
    try {
        // Get slices from both BSIs
        std::vector<HybridBitmap<uword>> bsi1_slices;
        std::vector<HybridBitmap<uword>> bsi2_slices;
        
        // Collect all slices from both BSIs
        for (int i = 0; i < bsi1->getNumberOfSlices(); i++) {
            bsi1_slices.push_back(bsi1->getSlice(i));
        }
        
        for (int i = 0; i < bsi2->getNumberOfSlices(); i++) {
            bsi2_slices.push_back(bsi2->getSlice(i));
        }
        
        // Call the CUDA implementation
        // Use CUDA events to time only kernel execution
        cudaEvent_t startEvt, stopEvt; cudaEventCreate(&startEvt); cudaEventCreate(&stopEvt);
        cudaEventRecord(startEvt);
        long long int res = bsi_dot_product_cuda<uword>(bsi1_slices, bsi2_slices);
        cudaEventRecord(stopEvt); cudaEventSynchronize(stopEvt);
        float ms=0; cudaEventElapsedTime(&ms, startEvt, stopEvt);
        g_last_kernel_ms = ms;
        cudaEventDestroy(startEvt); cudaEventDestroy(stopEvt);
        return res;
    } catch (const std::exception& e) {
        std::cerr << "CUDA dot product failed: " << e.what() << std::endl;
        std::cerr << "Falling back to CPU implementation." << std::endl;
        // Fall back to CPU implementation
        // Need to cast away const since the dot method expects a non-const pointer
        return bsi1->dot(const_cast<BsiAttribute<uword>*>(bsi2));
    }
#else
    // Just use regular CPU implementation if CUDA is not enabled
    // Need to cast away const since the dot method expects a non-const pointer
    return bsi1->dot(const_cast<BsiAttribute<uword>*>(bsi2));
#endif
}

int cuda_get_last_kernel_num_blocks() {
    return g_last_kernel_num_blocks;
}

int cuda_get_sm_count() {
#ifdef USE_CUDA
    int device_count = 0;
    cudaError_t err_count = cudaGetDeviceCount(&device_count);
    if (err_count != cudaSuccess || device_count == 0) {
        // std::cerr << "Error getting CUDA device count or no devices found." << std::endl;
        return 0;
    }
    cudaDeviceProp prop;
    cudaError_t err_prop = cudaGetDeviceProperties(&prop, 0); // Assuming device 0
    if (err_prop != cudaSuccess) {
        // std::cerr << "Error getting CUDA device properties." << std::endl;
        return 0;
    }
    return prop.multiProcessorCount;
#else
    return 0;
#endif
}

int cuda_get_cores_per_sm() {
#ifdef USE_CUDA
    int dev_count = 0;
    cudaError_t err_count = cudaGetDeviceCount(&dev_count);
    if (err_count != cudaSuccess || dev_count == 0) return 0;

    cudaDeviceProp prop;
    cudaError_t err_prop = cudaGetDeviceProperties(&prop, 0); // Assuming device 0
    if (err_prop != cudaSuccess) return 0;

    int coresPerSM;
    switch (prop.major * 10 + prop.minor) {
        case 90: coresPerSM = 128; break; 
        case 86: coresPerSM = 128; break;
        case 80: coresPerSM = 64;  break; 
        default: coresPerSM = 64;  break;
    }
    return coresPerSM;
#else
    return 0;
#endif
}

// forward declare kernels for vector dot
#ifdef USE_CUDA
extern "C" __global__ void vector_dot_kernel(const long*, const long*, size_t, long long*);
extern "C" __global__ void reduce_long_kernel(long long*, int);
#endif

long long int vector_dot_cuda(const std::vector<long>& vec1, const std::vector<long>& vec2) {
#ifdef USE_CUDA
    if (vec1.size() != vec2.size()) {
        throw std::runtime_error("Vector sizes must match for CUDA dot product");
    }
    size_t n = vec1.size();

    // Device pointers
    long *d_vec1 = nullptr, *d_vec2 = nullptr;
    long long *d_partial = nullptr;

    const int blockSize = 256;
    int numBlocks = static_cast<int>((n + blockSize - 1) / blockSize);
    numBlocks = std::min(numBlocks, 65535);

    // allocate
    CHECK_CUDA_ERROR(cudaMalloc(&d_vec1, n * sizeof(long)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_vec2, n * sizeof(long)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_partial, numBlocks * sizeof(long long)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_vec1, vec1.data(), n * sizeof(long), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_vec2, vec2.data(), n * sizeof(long), cudaMemcpyHostToDevice));

    dim3 grid(numBlocks, 1, 1);
    dim3 block(blockSize, 1, 1);

    cudaEvent_t startEvt, stopEvt; cudaEventCreate(&startEvt); cudaEventCreate(&stopEvt);
    cudaEventRecord(startEvt);

    void* args[] = { &d_vec1, &d_vec2, &n, &d_partial };
    size_t sharedMem = blockSize * sizeof(long long);
    CHECK_CUDA_ERROR(cudaLaunchKernel((void*)vector_dot_kernel, grid, block, args, sharedMem, nullptr));
    CHECK_CUDA_ERROR(cudaGetLastError());

    // reduction kernel if needed
    if (numBlocks > 1) {
        int reduceBlocks = 1;
        void* redArgs[] = { &d_partial, &numBlocks };
        CHECK_CUDA_ERROR(cudaLaunchKernel((void*)reduce_long_kernel, dim3(reduceBlocks), dim3(256), redArgs, 256 * sizeof(long long), nullptr));
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    cudaEventRecord(stopEvt); cudaEventSynchronize(stopEvt);
    float ms = 0; cudaEventElapsedTime(&ms, startEvt, stopEvt);
    g_last_kernel_ms = ms; // reuse global timing var

    long long result = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&result, d_partial, sizeof(long long), cudaMemcpyDeviceToHost));

    // free
    cudaFree(d_vec1); cudaFree(d_vec2); cudaFree(d_partial);
    cudaEventDestroy(startEvt); cudaEventDestroy(stopEvt);

    g_last_kernel_num_blocks = numBlocks;
    return result;
#else
    throw std::runtime_error("CUDA not available for vector dot product");
#endif
}

// Explicit template instantiations
template long long int bsi_dot_cuda<uint64_t>(
    const BsiAttribute<uint64_t>* bsi1, 
    const BsiAttribute<uint64_t>* bsi2);
