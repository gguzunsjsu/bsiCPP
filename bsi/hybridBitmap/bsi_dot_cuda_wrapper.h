#pragma once

#include <cstdint>
#include <vector>

// Forward declaration of HybridBitmap
template <class uword> class HybridBitmap;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Performs dot product of two BSI bitmap slices on the GPU.
 * This function accelerates the BSI dot product operation by processing
 * slice pairs in parallel on the GPU.
 * 
 * @param bsi1_slices Array of bitmap slices for first BSI
 * @param bsi2_slices Array of bitmap slices for second BSI
 * @return The dot product result
 */
template <class uword>
long long int bsi_dot_product_cuda(
    const std::vector<HybridBitmap<uword>>& bsi1_slices, 
    const std::vector<HybridBitmap<uword>>& bsi2_slices);

/**
 * Helper function to check if CUDA is available and working
 * 
 * @return true if CUDA is available and working, false otherwise
 */
bool is_cuda_available();

#ifdef __cplusplus
}
#endif
