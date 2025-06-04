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

/**
 * Return the duration (in milliseconds) of the last CUDA kernel(s) executed by
 * bsi_dot_cuda. This excludes host-side memory copies – only time measured on
 * device using CUDA events.
 */
float cuda_last_kernel_time_ms();

/**
 * Return the number of blocks launched by the last CUDA kernel(s) executed by
 * bsi_dot_cuda.
 */
int cuda_get_last_kernel_num_blocks();

/**
 * Return the number of Streaming Multiprocessors (SMs) on the current CUDA device.
 */
int cuda_get_sm_count();

/**
 * Return the number of CUDA cores per SM for the current device.
 */
int cuda_get_cores_per_sm();

/** Print GPU device information (name, SMs, cores, etc.) only if CUDA is
 * available.  Safe to call multiple times; information is printed once.
 */
void cuda_print_device_info();
