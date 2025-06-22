#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <numeric>
#include <cuda_runtime.h>

// Only include what we need
#include "../bsi/BsiVector.hpp"
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

    // Seed RNG for reproducibility (default or from argv[3])
    unsigned int seed = 12345;
    if (argc > 3) seed = static_cast<unsigned int>(std::stoul(argv[3]));
    std::mt19937 gen(seed);
    std::cout << "Using random seed: " << seed << std::endl;

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
    BsiVector<uint64_t>* bsi1 = ubsi1.buildBsiVectorFromVector(array1, 0.2);
    bsi1->setPartitionID(0);
    bsi1->setFirstSliceFlag(true);
    bsi1->setLastSliceFlag(true);
    
    BsiVector<uint64_t>* bsi2 = ubsi2.buildBsiVectorFromVector(array2, 0.2);
    bsi2->setPartitionID(0);
    bsi2->setFirstSliceFlag(true);
    bsi2->setLastSliceFlag(true);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    std::cout << "BSI build time: " << build_time << " ms" << std::endl;
    std::cout << "Memory used per BSI attribute: " << bsi1->getSizeInMemory()/(1024*1024) << " MB" << std::endl;

    if (cuda_available) {
        // Warm up GPU: create context & compile kernels (exclude one-time overhead)
        cudaFree(0);
        vector_dot_cuda(array1, array2);
        bsi_dot_cuda(bsi1, bsi2);
    }

    // Repeat each benchmark 1000 times and average
    const int runs = 1;
    std::vector<long long> vecCpuTimes(runs), vecGpuTimes(runs), bsiCpuTimes(runs), bsiGpuTimes(runs);
    long long vecCpuDot=0, vecGpuDot=0, bsiCpuDotVal=0, bsiGpuDotVal=0;

    // Vector CPU
    std::cout << "\nRunning vector CPU dot (" << runs << " runs)..." << std::endl;
    for (int j = 0; j < runs; j++) {
        long long tmp=0;
        auto s = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < vectorLen; i++) tmp += array1[i] * array2[i];
        auto e = std::chrono::high_resolution_clock::now();
        vecCpuTimes[j] = std::chrono::duration_cast<std::chrono::microseconds>(e - s).count();
        if (j == runs-1) vecCpuDot = tmp;
    }
    long long vectorCpuTime = std::accumulate(vecCpuTimes.begin(), vecCpuTimes.end(), 0LL) / runs;
    std::cout << "Vector CPU dot: " << vecCpuDot << ", avg time: " << vectorCpuTime << " µs" << std::endl;

    // Vector GPU
    long long vectorGpuTime = 0;
    if (cuda_available) {
        cuda_print_device_info();
        std::cout << "\nRunning vector GPU dot (" << runs << " runs)..." << std::endl;
        for (int j = 0; j < runs; j++) {
            vecGpuDot = vector_dot_cuda(array1, array2);
            vecGpuTimes[j] = static_cast<long long>(cuda_last_kernel_time_ms() * 1000.0);
        }
        vectorGpuTime = std::accumulate(vecGpuTimes.begin(), vecGpuTimes.end(), 0LL) / runs;
        std::cout << "Vector GPU dot: " << vecGpuDot << ", avg kernel time: " << vectorGpuTime << " µs" << std::endl;
        int launchedBlocksV = cuda_get_last_kernel_num_blocks();
        const int blockSize = 256;
        std::cout << "Vector GPU launched blocks: " << launchedBlocksV
                  << " | Threads: " << launchedBlocksV * blockSize << std::endl;
    }

    // BSI CPU
    std::cout << "\nRunning BSI CPU dot (" << runs << " runs)..." << std::endl;
    for (int j = 0; j < runs; j++) {
        auto s = std::chrono::high_resolution_clock::now();
        bsiCpuDotVal = bsi1->dot(bsi2);
        auto e = std::chrono::high_resolution_clock::now();
        bsiCpuTimes[j] = std::chrono::duration_cast<std::chrono::microseconds>(e - s).count();
    }
    long long bsiCpuTime = std::accumulate(bsiCpuTimes.begin(), bsiCpuTimes.end(), 0LL) / runs;
    std::cout << "BSI CPU dot: " << bsiCpuDotVal << ", avg time: " << bsiCpuTime << " µs" << std::endl;

    // BSI GPU
    long long bsiGpuTime = 0;
    if (cuda_available) {
        cuda_print_device_info();
        std::cout << "\nRunning BSI GPU dot (" << runs << " runs)..." << std::endl;
        for (int j = 0; j < runs; j++) {
            bsiGpuDotVal = bsi_dot_cuda(bsi1, bsi2);
            bsiGpuTimes[j] = static_cast<long long>(cuda_last_kernel_time_ms() * 1000.0);
        }
        bsiGpuTime = std::accumulate(bsiGpuTimes.begin(), bsiGpuTimes.end(), 0LL) / runs;
        std::cout << "BSI GPU dot: " << bsiGpuDotVal << ", avg kernel time: " << bsiGpuTime << " µs" << std::endl;
        int launchedBlocks = cuda_get_last_kernel_num_blocks();
        int totalSMs = cuda_get_sm_count();
        if (totalSMs > 0) {
            std::cout << "Launched blocks: " << launchedBlocks << std::endl;
            std::cout << "Total SMs: " << totalSMs << std::endl;
            int estSM = std::min(launchedBlocks, totalSMs);
            std::cout << "Estimated SMs utilized: " << estSM << std::endl;
            int cores = cuda_get_cores_per_sm();
            if (cores > 0)
                std::cout << "Estimated CUDA cores: " << estSM * cores << std::endl;
            const int bs = 256;
            std::cout << "Estimated threads: " << launchedBlocks * bs << std::endl;
        }
    }

    // Print results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Vector CPU dot: " << vecCpuDot << ", avg time: " << vectorCpuTime << " µs" << std::endl;
    if (cuda_available) {
        std::cout << "Vector GPU dot: " << vecGpuDot << ", avg kernel time: " << vectorGpuTime << " µs" << std::endl;
    }
    std::cout << "BSI CPU dot: " << bsiCpuDotVal << ", avg time: " << bsiCpuTime << " µs" << std::endl;
    if (cuda_available) {
        std::cout << "BSI GPU dot: " << bsiGpuDotVal << ", avg kernel time: " << bsiGpuTime << " µs" << std::endl;
    }

    // Verify results
    if (cuda_available) {
        if (bsiCpuDotVal == bsiGpuDotVal) {
            std::cout << "\n CPU and GPU results match!" << std::endl;
        } else {
            std::cout << "\n CPU and GPU results don't match!" << std::endl;
            std::cout << "Difference: " << (bsiCpuDotVal - bsiGpuDotVal) << std::endl;
        }
    }
    
    // Calculate speedups
    std::cout << "\n=== Performance Analysis ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Vector GPU vs CPU: " << (double)vectorCpuTime / vectorGpuTime << "x" << std::endl;
    std::cout << "BSI CPU vs Vector CPU: " << (double)vectorCpuTime / bsiCpuTime << "x" << std::endl;
    if (cuda_available) {
        std::cout << "BSI GPU vs Vector GPU: " << (double)vectorGpuTime / bsiGpuTime << "x" << std::endl;
        std::cout << "BSI GPU vs BSI CPU: " << (double)bsiCpuTime / bsiGpuTime << "x" << std::endl;
    }
    
    // Clean up
    delete bsi1;
    delete bsi2;
    
    return 0;
}
