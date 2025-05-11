#pragma once

#include <cstdint>

// Forward declarations
template <class uword> class BsiAttribute;
template <class uword> class BsiUnsigned;

/**
 * CUDA-accelerated dot product for BSI attributes
 * This function is a wrapper that will be called from BsiUnsigned/BsiAttribute
 * 
 * @param bsi1 First BSI attribute
 * @param bsi2 Second BSI attribute
 * @return The dot product result
 */
template <class uword>
long long int bsi_dot_cuda(const BsiAttribute<uword>* bsi1, const BsiAttribute<uword>* bsi2);

/**
 * Check if CUDA acceleration is available
 * @return true if CUDA is available, false otherwise
 */
bool cuda_dot_available();
