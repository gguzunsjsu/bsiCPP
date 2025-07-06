//
// Created by poorna on 2/6/25.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>

#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiVector.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"
#include "../bsi/hybridBitmap/UnitTestsOfHybridBitmap.hpp"

int main(){
    std::vector<long> array1;
    int range = pow(2, 16);
    int vectorLength = 10000000;

    for(auto i=0; i<vectorLength; i++){
        array1.push_back(std::rand()%range);
    }

    /*
     * Printing array
     */
//   for(auto i=0; i<vectorLength; i++){
//        std::cout << array1[i] << std::endl;
//    }

    long arraySum = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    for(auto i=0; i<vectorLength; i++){
        arraySum += array1[i];
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    auto array_duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();

    std::cout << "Array Sum is: \t" << arraySum << std::endl;
    std::cout << "Array sum duration: \t" << array_duration << std::endl;

    size_t element_size = sizeof(array1[0]);
    size_t data_size = array1.size() * element_size;
    size_t capacity_size = array1.capacity() * element_size;

    std::cout << "Element size: " << element_size << " bytes (" << element_size * 8 << " bits)" << std::endl;
    std::cout << "Data memory: " << data_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Allocated memory: " << capacity_size / (1024.0 * 1024.0) << " MB" << std::endl;

    size_t max_value = 0;
    for (const auto& val : array1) {
        max_value = std::max(max_value, static_cast<size_t>(val));
    }

    int bits_needed = 0;
    size_t temp = max_value;
    while (temp > 0) {
        bits_needed++;
        temp >>= 1;
    }

    std::cout << "Maximum value in vector: " << max_value << std::endl;
    std::cout << "Minimum bits needed: " << bits_needed << std::endl;
    std::cout << "Actual bits used per element: " << sizeof(array1[0]) * 8 << std::endl;
    std::cout << "Potential memory waste: " << 
        (sizeof(array1[0]) * 8 - bits_needed) * array1.size() / 8 / (1024.0 * 1024.0) << " MB" << std::endl;

    BsiUnsigned<uint64_t> ubsi;
    BsiVector<uint64_t>* bsi;

    std::vector<long> array1_long(array1.begin(), array1.end());
    bsi = ubsi.buildBsiVector(array1_long, 0.2);
    bsi->setFirstSliceFlag(true);
    bsi->setLastSliceFlag(true);
    bsi->setPartitionID(0);

    std::cout << "Memory used to store bsi vector: \t" << bsi->getSizeInMemory()/(pow(2, 20)) << std::endl;
//    std::cout << "Bits used by bsi: \t" << bsi->getBitsUsedBSI(pow(2, bsi->getNumberOfSlices())) << std::endl;

    auto t3 = std::chrono::high_resolution_clock::now();
    long bsi_sum = bsi->sumOfBsi();
    auto t4 = std::chrono::high_resolution_clock::now();
    auto bsi_duration = std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count();

    std::cout << "Sum using bsi is: \t" << bsi_sum << std::endl;
    std::cout << "bsi duration: \t" << bsi_duration << std::endl;




}