#include "hybridbitmap_cuda_wrapper.h"
#include <cuda_runtime.h>
#include <stdexcept>

extern "C" void and_verbatim_kernel(const uint32_t*, const uint32_t*, uint32_t*, int);

void and_verbatim_cuda(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, std::vector<uint32_t>& out) {
    int n = a.size();
    out.resize(n);
    uint32_t *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, n * sizeof(uint32_t));
    cudaMalloc(&d_b, n * sizeof(uint32_t));
    cudaMalloc(&d_out, n * sizeof(uint32_t));
    cudaMemcpy(d_a, a.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    and_verbatim_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_out, n);
    cudaMemcpy(out.data(), d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}
