#include <cuda_runtime.h>
extern "C"
__global__ void and_verbatim_kernel(const uint32_t* a, const uint32_t* b, uint32_t* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] & b[idx];
    }
}
