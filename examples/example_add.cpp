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
#include "../bsi/BsiAttribute.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"
#include "../bsi/hybridBitmap/UnitTestsOfHybridBitmap.hpp"

int main(){
    std::vector<long> array1;
    int range = 50;
    int vectorLength = 10000000;

    for(auto i=0; i<vectorLength; i++){
        array1.push_back(std::rand()%range);
    }

    /*
     * Printing array
     */
    for(auto i=0; i<vectorLength; i++){
        std::cout << array1[i] << std::endl;
    }

    long arraySum = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    for(auto i=0; i<vectorLength; i++){
        arraySum += array1[i];
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    auto array_duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();

    std::cout << "Array Sum is: \t" << arraySum << std::endl;
    std::cout << "Array sum duration: \t" << array_duration << std::endl;

    BsiUnsigned<uint64_t> ubsi;
    BsiAttribute<uint64_t>* bsi;

    bsi = ubsi.buildBsiAttributeFromVector(array1, 0.2);
    bsi->setFirstSliceFlag(true);
    bsi->setLastSliceFlag(true);
    bsi->setPartitionID(0);

    auto t3 = std::chrono::high_resolution_clock::now();
    long bsi_sum = bsi->sumOfBsi();
    auto t4 = std::chrono::high_resolution_clock::now();
    auto bsi_duration = std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count();

    std::cout << "Sum using bsi is: \t" << bsi_sum << std::endl;
    std::cout << "bsi duration: \t" << bsi_duration << std::endl;




}
