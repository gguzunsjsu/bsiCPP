#include "bsi_dot_cuda.h"
#include "bsi_dot_cuda_wrapper.h"
#include "../BsiAttribute.hpp"
#include "../BsiUnsigned.hpp"
#include <iostream>

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
        return bsi_dot_product_cuda<uword>(bsi1_slices, bsi2_slices);
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

// Explicit template instantiations
template long long int bsi_dot_cuda<uint64_t>(
    const BsiAttribute<uint64_t>* bsi1, 
    const BsiAttribute<uint64_t>* bsi2);
