//
// Created by poorna on 3/17/25.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>

#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiAttribute.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"
#include "../bsi/hybridBitmap/UnitTestsOfHybridBitmap.hpp"

int main() {
    int range = 50;
    int vectorLength = 10000000;

    std::vector<long> array1;
    array1.reserve(vectorLength);

//    auto t1 = std::chrono::high_resolution_clock::now();
//    for (auto i = 0; i < vectorLength; i++) {
//        array1.push_back(std::rand() % range);
//    }
//
//    long arraySum = 0;
//    for (auto i = 0; i < vectorLength; i++) {
//        arraySum += array1[i];
//    }
//    auto t2 = std::chrono::high_resolution_clock::now();
//    auto array_duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//
//    std::cout << "Traditional array:" << std::endl;
//    std::cout << "Array Sum: \t" << arraySum << std::endl;
//    std::cout << "Duration: \t" << array_duration << " microseconds" << std::endl;

    BsiUnsigned<uint64_t> ubsi;
    auto t3 = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < vectorLength; i++) {
        array1.push_back(std::rand() % range);
    }
    BsiAttribute<uint64_t>* bsi1 = ubsi.buildBsiAttributeFromVector(array1, 0.2);
    bsi1->setFirstSliceFlag(true);
    bsi1->setLastSliceFlag(true);
    bsi1->setPartitionID(0);

    long bsi1_sum = bsi1->sumOfBsi();
    auto t4 = std::chrono::high_resolution_clock::now();
    auto bsi1_duration = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    std::cout << "\nTraditional BSI creation:" << std::endl;
    std::cout << "BSI Sum: \t" << bsi1_sum << std::endl;
    std::cout << "Duration: \t" << bsi1_duration << " microseconds" << std::endl;

    // Direct BSI creation with random values
    auto t5 = std::chrono::high_resolution_clock::now();
    BsiAttribute<uint64_t>* bsi2 = bsi1->createRandomBsi(vectorLength, range, 0.2);

    long bsi2_sum = bsi2->sumOfBsi();
    auto t6 = std::chrono::high_resolution_clock::now();
    auto bsi2_duration = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();

    std::cout << "\nDirect BSI creation:" << std::endl;
    std::cout << "BSI Sum: \t" << bsi2_sum << std::endl;
    std::cout << "Duration: \t" << bsi2_duration << " microseconds" << std::endl;

    double expectedAvg = (range - 1) / 2.0;
    double expectedSum = expectedAvg * vectorLength;
    std::cout << "\nStatistical analysis:" << std::endl;
    std::cout << "Expected average value: " << expectedAvg << std::endl;
    std::cout << "Expected total sum: " << expectedSum << std::endl;

//    double errorArray = std::abs(arraySum - expectedSum) / expectedSum * 100;
    double errorBSI1 = std::abs(bsi1_sum - expectedSum) / expectedSum * 100;
    double errorBSI2 = std::abs(bsi2_sum - expectedSum) / expectedSum * 100;

    std::cout << "\nRelative errors:" << std::endl;
//    std::cout << "Array: \t\t" << errorArray << "%" << std::endl;
    std::cout << "Traditional BSI: " << errorBSI1 << "%" << std::endl;
    std::cout << "Direct BSI: \t" << errorBSI2 << "%" << std::endl;

    size_t bsi1_memory = bsi1->getSizeInMemory();
    size_t bsi2_memory = bsi2->getSizeInMemory();

    std::cout << "\nMemory usage:" << std::endl;
    std::cout << "Traditional BSI: " << bsi1_memory / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Direct BSI: \t" << bsi2_memory / (1024.0 * 1024.0) << " MB" << std::endl;

    delete bsi1;
    delete bsi2;

    return 0;
}