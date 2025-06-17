#ifndef HYBRIDBITMAP_CUDA_HPP
#define HYBRIDBITMAP_CUDA_HPP

#include "hybridbitmap.h" // For uword type and HybridBitmap structure
#include "../bsi_util_cuda.hpp" // For CUDA_CHECK

namespace HybridBitmapCuda {

/**
 * @brief Counts set bits in a bitmap buffer on the GPU.
 *
 * @tparam uword The word type of the bitmap.
 * @param d_buffer Device pointer to the bitmap data.
 * @param buffer_len_words Length of the buffer in words.
 * @param is_verbatim True if the bitmap is verbatim, false if RLE.
 * @param size_in_bits Actual size of the bitmap in bits (important for RLE and partial last words).
 * @return The total number of set bits.
 */
template <class uword>
unsigned long long countSetBitsDevice(
    const uword* d_buffer,
    size_t buffer_len_words,
    bool is_verbatim,
    size_t size_in_bits
);

/**
 * @brief Wrapper function to get the number of set bits in a HybridBitmap using the GPU.
 * Manages memory transfer to/from GPU.
 *
 * @tparam uword The word type of the bitmap.
 * @param hbm The HybridBitmap object (host side).
 * @return The total number of set bits.
 */
template <class uword>
unsigned long long getNumberOfOnesGpu(const HybridBitmap<uword>& hbm);

} // namespace HybridBitmapCuda

#endif // HYBRIDBITMAP_CUDA_HPP
