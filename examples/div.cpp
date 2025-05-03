//
// Created by panchalpc on 4/26/25.
//

// Test code for muliplication and SUM with bsi vector to array changes.

#include <iostream>
#include <cstdint>
#include <bitset>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>

#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
// #include "../bsi/BsiAttribute.hpp"
// #include "../bsi/hybridBitmap/hybridbitmap.h"
// #include "../bsi/hybridBitmap/UnitTestsOfHybridBitmap.hpp"

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    int vectorLength = 3;

    std::vector<long> dividends;
    std::vector<long> divisors;
    std::vector<long> array_two;
    std::vector<long> array_x;

    dividends = {2,6,9};
    divisors= {2,3,4};
    array_two= {2,2,2};
    array_x= {1,2,2};


    BsiSigned<uint64_t> ubsi;

    BsiAttribute<uint64_t>* dividend_bsi = ubsi.buildBsiAttributeFromVector(dividends, 0.2);
    dividend_bsi->setFirstSliceFlag(true);
    dividend_bsi->setLastSliceFlag(true);
    dividend_bsi->setPartitionID(0);

    BsiAttribute<uint64_t>* divisor_bsi = ubsi.buildBsiAttributeFromVector(divisors, 0.2);
    divisor_bsi->setFirstSliceFlag(true);
    divisor_bsi->setLastSliceFlag(true);
    divisor_bsi->setPartitionID(0);

    BsiAttribute<uint64_t>* two_bsi = ubsi.buildBsiAttributeFromVector(array_two, 0.2);
    two_bsi->setFirstSliceFlag(true);
    two_bsi->setLastSliceFlag(true);
    two_bsi->setPartitionID(0);

    BsiAttribute<uint64_t>* x_bsi = ubsi.buildBsiAttributeFromVector(array_x, 0.2);
    x_bsi->setFirstSliceFlag(true);
    x_bsi->setLastSliceFlag(true);
    x_bsi->setPartitionID(0);


    BsiAttribute<uint64_t>* resultBsi;
    BsiAttribute<uint64_t>* resultBsi2;

    auto t3 = std::chrono::high_resolution_clock::now();

    std::cout << "1: Multiplication - divisor*x_bsi" << std::endl;
    resultBsi = divisor_bsi->multiplication(x_bsi);

    std::cout << "resultBsi size: " << resultBsi << std::endl;
    std::cout << "resultBsi 0: " << resultBsi->getValue(0) << std::endl;
    std::cout << "resultBsi 1: " << resultBsi->getValue(1) << std::endl;
    std::cout << "resultBsi 2: " << resultBsi->getValue(2) << std::endl;


    std::cout << "2: SUM - dividend + result" << std::endl;
    resultBsi2 = two_bsi->SUM(resultBsi);

    std::cout << "resultBsi2 size: " << resultBsi2 << std::endl;
    std::cout << "resultBsi2 0: " << resultBsi2->getValue(0) << std::endl;
    std::cout << "resultBsi2 1: " << resultBsi2->getValue(1) << std::endl;
    std::cout << "resultBsi2 2: " << resultBsi2->getValue(2) << std::endl;

    auto t4 = std::chrono::high_resolution_clock::now();

    std::cout << "bsi div duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count() << std::endl;
    return 0;

}
