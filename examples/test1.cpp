//Sum_Vector.cpp

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

template <class uword>
std::vector<int> convertBSIToDecimal(const BsiVector<uword>& res) {
    std::vector<int> decimals(res.bsi[0].numSlices(), 0);
    for (size_t i = 0; i < res.bsi.size(); ++i) {
        int weight = 1 << (res.offset + i);  // 2^(offset + i)
        for (size_t row = 0; row < res.bsi[i].numSlices(); ++row) {
            if (res.bsi[i].test(row)) {
                decimals[row] += weight;
            }
        }
    }
    return decimals;
}

std::vector<long> hexStringToVector(const std::string& hexStr) {
    std::vector<long> result;

    // Ensure even-length hex string
    if (hexStr.length() % 2 != 0) {
        throw std::invalid_argument("Hex string must have even length");
    }

    for (size_t i = 0; i < hexStr.length(); i += 2) {
        // Extract byte pair (e.g., "00", "E5")
        std::string byteStr = hexStr.substr(i, 2);

        // Convert to long (base 16)
        long byteValue = std::stol(byteStr, nullptr, 16);
        result.push_back(byteValue);
    }

    return result;
}


int main(){
    std::vector<long> array1;
    std::vector<long> array2;
    std::vector<long> res;

    array1 = {2, 3, 3};
    array2 = {5, 2, 9};

    BsiUnsigned<uint64_t> ubsi;
    BsiVector<uint64_t>* bsi_1;
    BsiVector<uint64_t>* bsi_2;

    bsi_1 = ubsi.buildBsiVectorFromVector(array1, 0.2);
    // bsi_1 = ubsi.buildBsiVectorFromVector_without_compression(array1);
    bsi_1->setFirstSliceFlag(true);
    bsi_1->setLastSliceFlag(true);
    bsi_1->setPartitionID(0);

    std::cout << "Bsi 1: " << bsi_1->getNumberOfSlices() << std::endl;
    std::cout << "Bsi 1 numSlices: " << bsi_1->numSlices << std::endl;
    std::cout << "Bsi 1 value: " << bsi_1 << std::endl;
    std::cout <<"Number of elements in the bsi1 vector: " << bsi_1->getNumberOfRows() << std::endl;
    for (int i = 0; i < bsi_1->getNumberOfRows(); i++) {
        std::cout << i << ": " << bsi_1->getValue(i) << std::endl;

    }

    bsi_2 = ubsi.buildBsiVectorFromVector(array2, 0.2);
    // bsi_2 = ubsi.buildBsiVectorFromVector_without_compression(array2);
    bsi_2->setFirstSliceFlag(true);
    bsi_2->setLastSliceFlag(true);
    bsi_2->setPartitionID(0);


    std::cout << "Bsi 2_: " << bsi_2->getNumberOfSlices() << std::endl;
    std::cout << "Bsi 2 numSlices: " << bsi_2->numSlices << std::endl;
    std::cout << "Bsi 2 value: " << bsi_2 << std::endl;
    std::cout <<"Number of elements in the bsi2 vector: " << bsi_2->getNumberOfRows() << std::endl;
    //std::cout << bsi_2->getNumberOfRows() << std::endl;
    for (int i = 0; i < bsi_2->getNumberOfRows(); i++) {
        std::cout << i << ": " << bsi_2->getValue(i) << std::endl;

    }

    // auto t3 = std::chrono::high_resolution_clock::now();
    BsiVector<uint64_t>* resultBsi = bsi_1->SUM(bsi_2);
    // BsiVector<uint64_t>* resultBsi = bsi_1->sum_Horizontal(bsi_2);
    // auto t4 = std::chrono::high_resolution_clock::now();
    // auto bsi_duration = std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count();

    std::cout << "Sum of all values using bsi is: \t" << resultBsi->sumOfBsi() << std::endl;
    // std::cout << "bsi duration: \t" << bsi_duration << std::endl;

    // res = convertBSIToDecimal(resultBsi);

    // std::cout << "Deci: \t" << res << std::endl;
    // std::stringstream ss;
    // ss << resultBsi;
    // res = hexStringToVector(ss.str());
    // for (int i = 0 ; i <res.numSlices() ; i++) {
    // std::cout << "Deci: " << res[i] << std::endl;

    // }

    std::cout << resultBsi->getNumberOfRows() << std::endl;
    for (int i = 0; i < resultBsi->getNumberOfRows(); i++) {
        std::cout << i << ": " << resultBsi->getValue(i) << std::endl;

    }


    std::cout <<"Number of elements in the result vector: " << resultBsi->getNumberOfRows() << std::endl;


    return 0;
}