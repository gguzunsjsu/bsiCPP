#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

// Only include what we need
#include "../bsi/BsiAttribute.hpp"
#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/hybridBitmap/bsi_dot_cuda.h"

int main(int argc, char* argv[]) {
    // Default parameters
    int vectorLen = 1000000;  // 1M elements
    int range = 100;          // Values in range [0,100]
    
    // Allow command-line override
    if (argc > 1) vectorLen = std::stoi(argv[1]);
    if (argc > 2) range = std::stoi(argv[2]);
    
    std::cout << "=== BSI Dot Product Benchmark ===" << std::endl;
    std::cout << "Vector length: " << vectorLen << std::endl;
    std::cout << "Value range: [0, " << range-1 << "]" << std::endl;
    
    // Check if CUDA is available
    bool cuda_available = cuda_dot_available();
    if (cuda_available) {
        std::cout << "CUDA is available and will be used." << std::endl;
    } else {
        std::cout << "CUDA is not available. Running CPU-only benchmark." << std::endl;
    }
    
    std::cout << "\n Generating random data..." << std::endl;
    
    // Generate random data
    std::vector<long> array1(vectorLen);
    std::vector<long> array2(vectorLen);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, range-1);
    
    for (int i = 0; i < vectorLen; i++) {
        array1[i] = dist(gen);
        array2[i] = dist(gen);
    }
    
    std::cout << "Building BSI structures..." << std::endl;
    
    // Create BSI structures
    BsiUnsigned<uint64_t> ubsi1;
    BsiUnsigned<uint64_t> ubsi2;
    
    // Build BSI attributes from vectors
    auto t1 = std::chrono::high_resolution_clock::now();
    BsiAttribute<uint64_t>* bsi1 = ubsi1.buildBsiAttributeFromVector(array1, 0.2);
    bsi1->setPartitionID(0);
    bsi1->setFirstSliceFlag(true);
    bsi1->setLastSliceFlag(true);
    
    BsiAttribute<uint64_t>* bsi2 = ubsi2.buildBsiAttributeFromVector(array2, 0.2);
    bsi2->setPartitionID(0);
    bsi2->setFirstSliceFlag(true);
    bsi2->setLastSliceFlag(true);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    std::cout << "BSI build time: " << build_time << " ms" << std::endl;
    std::cout << "Memory used per BSI attribute: " << bsi1->getSizeInMemory()/(1024*1024) << " MB" << std::endl;
    
    // --- Vector dot product benchmarks ---
    std::cout << "\nRunning vector dot product (CPU baseline)..." << std::endl;
    long long vector_dot_cpu = 0;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < vectorLen; i++) {
        vector_dot_cpu += array1[i] * array2[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    long long vector_cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();

    // Optional CUDA version
    long long vector_dot_gpu = vector_dot_cpu; // default to match CPU
    long long vector_gpu_time = 0;            // microseconds (kernel-only)
    if (cuda_available) {
        cuda_print_device_info();
        cout << "\n Cuda available for vector dot product\n";
        // Kernel-only timing returned via global helper
        vector_dot_gpu = vector_dot_cuda(array1, array2);
        vector_gpu_time = static_cast<long long>(cuda_last_kernel_time_ms() * 1000.0); // µs
        std::cout << "Vector GPU kernel-only time: " << cuda_last_kernel_time_ms() << " ms" << std::endl;
        int launched_blocks_v = cuda_get_last_kernel_num_blocks();
        const int kernel_block_size_v = 256;
        std::cout << "Vector GPU launched blocks: " << launched_blocks_v << " | Threads: "
                  << launched_blocks_v * kernel_block_size_v << std::endl;
    }

    // Compute dot product using BSI CPU
    t1 = std::chrono::high_resolution_clock::now();
    long long bsi_cpu_dot = bsi1->dot(bsi2);
    t2 = std::chrono::high_resolution_clock::now();
    auto bsi_cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    
    // Compute dot product using BSI GPU if available
    long long bsi_gpu_dot = 0;
    long long bsi_gpu_time = 0;
    if (cuda_available) {
        cuda_print_device_info(); // Print GPU info before benchmark

        t1 = std::chrono::high_resolution_clock::now();
        bsi_gpu_dot = bsi_dot_cuda(bsi1, bsi2);
        t2 = std::chrono::high_resolution_clock::now();
        bsi_gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
        std::cout << "BSI GPU kernel-only time: " << cuda_last_kernel_time_ms() << " ms" << std::endl;

        int launched_blocks = cuda_get_last_kernel_num_blocks();
        int total_sms = cuda_get_sm_count();
        if (total_sms > 0) { 
            std::cout << "Launched blocks for kernel: " << launched_blocks << std::endl;
            std::cout << "Total SMs on device: " << total_sms << std::endl;
            int estimated_sms_utilized = std::min(launched_blocks, total_sms);
            std::cout << "Estimated SMs utilized: " << estimated_sms_utilized 
                      << " (based on launched blocks vs available SMs)" << std::endl;
            int cores_per_sm = cuda_get_cores_per_sm();
            if (cores_per_sm > 0) {
                std::cout << "Estimated CUDA Cores utilized: " << estimated_sms_utilized * cores_per_sm << std::endl;
            }
            // Assuming block size of 256 for the main kernel, as set in bsi_dot_cuda_wrapper.cpp
            const int kernel_block_size = 256;
            std::cout << "Estimated Threads launched: " << launched_blocks * kernel_block_size << std::endl;
        }
    }
    
    // Print results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Vector CPU dot: " << vector_dot_cpu << ", time: " << vector_cpu_time << " µs" << std::endl;
    if (cuda_available) {
        std::cout << "Vector GPU dot: " << vector_dot_gpu << ", kernel time: " << vector_gpu_time << " µs" << std::endl;
    }
    std::cout << "BSI CPU dot: " << bsi_cpu_dot << ", time: " << bsi_cpu_time << " µs" << std::endl;
    if (cuda_available) {
        std::cout << "BSI GPU dot: " << bsi_gpu_dot << ", kernel time: " << bsi_gpu_time << " µs" << std::endl;
    }

    // Verify results
    if (cuda_available) {
        if (bsi_cpu_dot == bsi_gpu_dot) {
            std::cout << "\n CPU and GPU results match!" << std::endl;
        } else {
            std::cout << "\n CPU and GPU results don't match!" << std::endl;
            std::cout << "Difference: " << (bsi_cpu_dot - bsi_gpu_dot) << std::endl;
        }
    }
    
    // Calculate speedups
    std::cout << "\n=== Performance Analysis ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Vector GPU vs CPU: " << (double)vector_cpu_time / vector_gpu_time << "x" << std::endl;
    std::cout << "BSI CPU vs Vector CPU: " << (double)vector_cpu_time / bsi_cpu_time << "x" << std::endl;
    if (cuda_available) {
        std::cout << "BSI GPU vs Vector GPU: " << (double)vector_gpu_time / bsi_gpu_time << "x" << std::endl;
        std::cout << "BSI GPU vs BSI CPU: " << (double)bsi_cpu_time / bsi_gpu_time << "x" << std::endl;
    }
    
    // Clean up
    delete bsi1;
    delete bsi2;
    
    return 0;
}
