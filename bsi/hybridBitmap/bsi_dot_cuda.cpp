#include "bsi_dot_cuda.h"
#include "bsi_dot_cuda_wrapper.h"
#include "../BsiAttribute.hpp"
#include "../BsiUnsigned.hpp"
#include <iostream>

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
        const auto& bsi1_slices = bsi1->getSlice();
        const auto& bsi2_slices = bsi2->getSlice();
        
        // Call the CUDA implementation
        return bsi_dot_product_cuda<uword>(bsi1_slices, bsi2_slices);
    } catch (const std::exception& e) {
        std::cerr << "CUDA dot product failed: " << e.what() << std::endl;
        std::cerr << "Falling back to CPU implementation." << std::endl;
        // Fall back to CPU implementation
        return bsi1->dot(bsi2);
    }
    #else
    // Just use regular CPU implementation if CUDA is not enabled
    return bsi1->dot(bsi2);
    #endif
}

// Explicit template instantiations
template long long int bsi_dot_cuda<uint64_t>(
    const BsiAttribute<uint64_t>* bsi1, 
    const BsiAttribute<uint64_t>* bsi2);
