//
// Created by poorna on 2/10/25.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>

#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiAttribute.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"
#include "../bsi/hybridBitmap/UnitTestsOfHybridBitmap.hpp"

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    int dividend_range = 50;
    int divisor_range = 50;
    int vectorLength = 1000;

    std::vector<long> dividends;
    std::vector<long> divisors;

    for(int i = 0; i < vectorLength; i++) {
        dividends.push_back(1 + (gen() % dividend_range));
        divisors.push_back(1 + (gen() % divisor_range));
    }

    std::vector<long> array_quotients;
    std::vector<long> array_remainders;

    auto t1 = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < vectorLength; i++) {
        array_quotients.push_back(dividends[i] / divisors[i]);
        array_remainders.push_back(dividends[i] % divisors[i]);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto array_duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();

    long array_quotient_sum = 0;
    long array_remainder_sum = 0;

    for(int i = 0; i < vectorLength; i++) {
        array_quotient_sum += array_quotients[i];
        array_remainder_sum += array_remainders[i];
    }

    std::cout << "Array division completed." << std::endl;
    std::cout << "Sum of array quotients: " << array_quotient_sum << std::endl;
    std::cout << "Sum of array remainders: " << array_remainder_sum << std::endl;
    std::cout << "Array division duration: " << array_duration << " microseconds" << std::endl;

    /*
     * BSI division
     */
    BsiUnsigned<uint64_t> ubsi;

    BsiAttribute<uint64_t>* dividend_bsi = ubsi.buildBsiAttributeFromVector(dividends, 0.2);
    dividend_bsi->setFirstSliceFlag(true);
    dividend_bsi->setLastSliceFlag(true);
    dividend_bsi->setPartitionID(0);

    BsiAttribute<uint64_t>* divisor_bsi = ubsi.buildBsiAttributeFromVector(divisors, 0.2);
    divisor_bsi->setFirstSliceFlag(true);
    divisor_bsi->setLastSliceFlag(true);
    divisor_bsi->setPartitionID(0);

//    std::cout << "Dividend BSI slices: " << dividend_bsi->getNumberOfSlices() << std::endl;
//    std::cout << "Divisor BSI slices: " << divisor_bsi->getNumberOfSlices() << std::endl;

    BsiUnsigned<uint64_t> divider;

    // Perform division
    auto t3 = std::chrono::high_resolution_clock::now();

    std::pair<BsiAttribute<uint64_t>*, BsiAttribute<uint64_t>*> result =
            divider.divide(*dividend_bsi, *divisor_bsi);

    auto t4 = std::chrono::high_resolution_clock::now();
    auto bsi_duration = std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count();

    BsiAttribute<uint64_t>* quotient_bsi = result.first;
    BsiAttribute<uint64_t>* remainder_bsi = result.second;

//    std::cout << "Quotient BSI slices: " << quotient_bsi->getNumberOfSlices() << std::endl;
//    std::cout << "Remainder BSI slices: " << remainder_bsi->getNumberOfSlices() << std::endl;

    long bsi_quotient_sum = quotient_bsi->sumOfBsi();
    long bsi_remainder_sum = remainder_bsi->sumOfBsi();

    std::cout << "Sum of BSI quotients: " << bsi_quotient_sum << std::endl;
    std::cout << "Sum of BSI remainders: " << bsi_remainder_sum << std::endl;
    std::cout << "BSI division duration: " << bsi_duration << " microseconds" << std::endl;


//    std::cout << "\nFirst 5 rows for debugging:" << std::endl;
//    std::cout << "Row\tDividend\tDivisor\tArray Q\tArray R" << std::endl;
//    for (int i = 0; i < 5 && i < vectorLength; i++) {
//        std::cout << i << "\t" << dividends[i] << "\t\t" << divisors[i]
//                  << "\t" << array_quotients[i] << "\t" << array_remainders[i] << std::endl;
//    }
//
//    std::cout << "\nBSI results for first 5 rows:" << std::endl;
//    std::cout << "Row\tBSI Q\tBSI R" << std::endl;
//    for (int i = 0; i < 5 && i < vectorLength; i++) {
//        long bsi_q = quotient_bsi->getValue(i);
//        long bsi_r = remainder_bsi->getValue(i);
//        std::cout << i << "\t" << bsi_q << "\t" << bsi_r << std::endl;
//    }

    delete quotient_bsi;
    delete remainder_bsi;
    delete dividend_bsi;
    delete divisor_bsi;

    return 0;
}
