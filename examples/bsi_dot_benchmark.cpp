#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

// Include in the correct order
#include "../bsi/BsiAttribute.hpp"
#include "../bsi/BsiSigned.hpp"
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
    
    std::cout << "\nGenerating random data..." << std::endl;
    
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
    
    // Compute dot product using vectors (baseline)
    std::cout << "\nRunning dot product benchmarks..." << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    long long vector_dot = 0;
    for (int i = 0; i < vectorLen; i++) {
        vector_dot += array1[i] * array2[i];
    }
    t2 = std::chrono::high_resolution_clock::now();
    auto vector_time = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    
    // Compute dot product using BSI CPU
    t1 = std::chrono::high_resolution_clock::now();
    long long bsi_cpu_dot = bsi1->dot(bsi2);
    t2 = std::chrono::high_resolution_clock::now();
    auto bsi_cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    
    // Compute dot product using BSI GPU if available
    long long bsi_gpu_dot = 0;
    long long bsi_gpu_time = 0;
    if (cuda_available) {
        t1 = std::chrono::high_resolution_clock::now();
        bsi_gpu_dot = bsi_dot_cuda(bsi1, bsi2);
        t2 = std::chrono::high_resolution_clock::now();
        bsi_gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    }
    
    // Print results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Vector dot product: " << vector_dot << ", time: " << vector_time << " µs" << std::endl;
    std::cout << "BSI CPU dot product: " << bsi_cpu_dot << ", time: " << bsi_cpu_time << " µs" << std::endl;
    
    if (cuda_available) {
        std::cout << "BSI GPU dot product: " << bsi_gpu_dot << ", time: " << bsi_gpu_time << " µs" << std::endl;
        
        // Verify results
        if (bsi_cpu_dot == bsi_gpu_dot) {
            std::cout << "\n✓ CPU and GPU results match!" << std::endl;
        } else {
            std::cout << "\n✗ CPU and GPU results don't match!" << std::endl;
            std::cout << "Difference: " << (bsi_cpu_dot - bsi_gpu_dot) << std::endl;
        }
    }
    
    // Calculate speedups
    std::cout << "\n=== Performance Analysis ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "BSI CPU vs Vector: " << (double)vector_time / bsi_cpu_time << "x" << std::endl;
    
    if (cuda_available) {
        std::cout << "BSI GPU vs Vector: " << (double)vector_time / bsi_gpu_time << "x" << std::endl;
        std::cout << "BSI GPU vs BSI CPU: " << (double)bsi_cpu_time / bsi_gpu_time << "x" << std::endl;
    }
    
    // Clean up
    delete bsi1;
    delete bsi2;
    
    return 0;
}
