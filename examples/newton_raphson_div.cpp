//
// Created by parth on 6/26/25.
//

#include <iostream>
#include <cstdint>
#include <bitset>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <stdio.h>

#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiVector.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"
#include "../bsi/hybridBitmap/UnitTestsOfHybridBitmap.hpp"



int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    int one_range = 5000;
    int two_range = 5000;

    std::vector<long> one;
    std::vector<long> two;
    std::vector<double> one_dec;
    int constant = 22; //constant to multiply

    int random_vec_size = 100;

    one_dec = {-0.32};
    one = {-22};
    two = {4};

    // one = {2,6,9, 10, 50};
    // two = {2,6,9, 10, 50};

    // one = {38, 5, 12, 50, 18, 7, 3, 32, 42, 43, 15, 35, 39, 20, 39, 34, 46, 8, 13, 23};
    // two = {31, 12, 23, 48, 25, 16, 38, 6, 28, 9, 8, 46, 4, 6, 31, 8, 43, 11, 25, 16};
    //
    // one = {40, 30, 12, 8, 14, 0, 37, 33, 37, 5, 29, 49, 26, 32, 50, 27, 8, 13, 9, 19, 7,
    //                          47, 26, 34, 25, 14, 6, 11, 14, 16, 38, 42, 45, 16, 14, 3, 7, 45, 29, 45};
    // two = {39, 25, 47, 12, 5, 10, 30, 5, 1, 26, 9, 9, 2, 47, 29, 48, 28, 29, 13, 27,
    //                          30, 1, 43, 17, 46, 7, 45, 1, 27, 41, 1, 31, 43, 12, 35, 17, 26, 29, 47, 42};
    //
    //
    // one = {3845, 8983, 2462, 5143, 4161, 1682, 858, 6666, 1618, 8762, 4727, 4496, 7324, 5405, 8503, 4113, 6344, 4765,
    //                          9532, 6318, 5836, 1355, 7574, 5366, 923, 8105, 4128, 5522, 8079, 5614, 9500, 1124, 9662, 8096, 4215, 5805, 8636, 3055, 4342, 3306};
    // two = {9173, 6213, 7136, 3150, 7709, 3049, 2233, 2729, 4630, 5721, 6472, 7239, 8922, 3978, 8275, 5955, 1584, 6472, 6248, 894, 2728,
    //                          2645, 7825, 4747, 7680, 3204, 1729, 2069, 4722, 7085, 7129, 3161, 5333, 4241, 9510, 3614, 3695, 2781, 352, 3647};
    //
    // for(int i = 0; i < random_vec_size; i++) {
    //     one.push_back(1 + (gen() % one_range));
    //     two.push_back(1 + (gen() % two_range));
    // }

    // Define vector size:
    int vector_length = one.size();

    std::vector<long> normal_div(vector_length);
    std::vector<long> normal_r(vector_length);

    std::vector<long> normal_sum(vector_length);
    std::vector<long> normal_mul(vector_length);
    std::vector<long> normal_mulc(vector_length);

    std::vector<double> normal_mul_d(vector_length);
    std::vector<double> normal_sum_d(vector_length);


    auto t0 = std::chrono::high_resolution_clock::now();

    // Normal division:
    for(int i = 0; i < vector_length; ++i) {
        normal_div[i] = one[i] / two[i];
        normal_r[i] = one[i] % two[i];
        std::cout << "normal div :" << std::endl;
        std::cout << normal_div[i] << std::endl;

    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "normal div duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() << std::endl;


    // Normal sum:
    auto t2 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < vector_length; i++) {
        normal_sum[i] = one[i] + two[i];
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    std::cout << "normal sum duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count() << std::endl;


    // Normal mulitplication:
    auto tmc = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < vector_length; i++) {
        normal_mul[i] = one[i] * two[i];
    }
    auto tmc2 = std::chrono::high_resolution_clock::now();

    std::cout << "normal multiply, duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(tmc2-tmc).count() << std::endl;


    // Normal mulitply with constant:
    auto tmul1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < vector_length; i++) {
        normal_mulc[i] = one[i] * constant;
    }
    auto tmul2 = std::chrono::high_resolution_clock::now();

    std::cout << "normal multiply with const, duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(tmul2-tmul1).count() << std::endl;


    //Decimals:
    // Decimal: Normal mulitplication:
    auto tmc_d = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < vector_length; i++) {
        normal_mul_d[i] = one_dec[i] * two[i];
    }
    auto tmc2_d = std::chrono::high_resolution_clock::now();

    std::cout << "decimal: normal multiply, duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(tmc2_d-tmc_d).count() << std::endl;

    // Decimal normal sum:
    auto t2_d = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < vector_length; i++) {
        normal_sum_d[i] = one_dec[i] + two[i];
    }
    auto t3_d = std::chrono::high_resolution_clock::now();

    std::cout << "decimal normal sum duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t3_d-t2_d).count() << std::endl;


    // BSI operations:
    // BsiUnsigned<uint64_t> unbsi;
    BsiSigned<uint64_t> ubsi;


    // BsiAttribute<uint64_t>* one_bsi = ubsi.buildBsiAttributeFromVector(one, 0.2);
    BsiVector<uint64_t>* one_bsi = ubsi.buildBsiVector(one, 0.2);
    one_bsi->setFirstSliceFlag(true);
    one_bsi->setLastSliceFlag(true);
    one_bsi->setPartitionID(0);

    BsiVector<uint64_t>* two_bsi = ubsi.buildBsiVector(two, 0.2);
    two_bsi->setFirstSliceFlag(true);
    two_bsi->setLastSliceFlag(true);
    two_bsi->setPartitionID(0);

    BsiVector<uint64_t>* one_dec_bsi = ubsi.buildBsiVector(one_dec, 2 ,0.2);
    one_dec_bsi->setFirstSliceFlag(true);
    one_dec_bsi->setLastSliceFlag(true);
    one_dec_bsi->setPartitionID(0);

    BsiVector<uint64_t>* resultBsi;
    BsiVector<uint64_t>* resultBsi2;
    BsiVector<uint64_t>* resultBsi3;

    BsiVector<uint64_t>* resultBsi_d;
    BsiVector<uint64_t>* resultBsi2_d;

    std::cout << "0. Decimal value: " << one_dec_bsi->getValue_with_decimal(0) << std::endl;


    // multiplication:
    std::cout << "1: Multiplication = one x two" << std::endl;
    auto t41 = std::chrono::high_resolution_clock::now();
    resultBsi = one_bsi->multiplyBSI(two_bsi);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "bsi multiplication duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t4-t41).count() << std::endl;

    for (int j=0; j < vector_length; j++) {
        std::cout << "resultBsi " << j << ": " << resultBsi->getValue(j) << std::endl;
        if (resultBsi->getValue(j) != normal_mul[j]) {
            std::cout << "resultBsi " << j << ": " << resultBsi->getValue(j) << " - Not matched!"<< std::endl;
        }
        else continue;
    }

    // decimal multiplication:
    std::cout << "1.5: Decimal Multiplication = one_dec x two" << std::endl;
    auto t41_b = std::chrono::high_resolution_clock::now();
    resultBsi_d = one_dec_bsi->multiplyBSI(two_bsi);
    auto t4_b = std::chrono::high_resolution_clock::now();
    std::cout << "bsi multiplication duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t4_b-t41_b).count() << std::endl;

    for (int j=0; j < vector_length; j++) {
        std::cout << "resultBsi_d " << j << ": " << resultBsi_d->getValue_with_decimal(j) << std::endl;
        if (resultBsi_d->getValue_with_decimal(j) != normal_mul_d[j]) {
            std::cout << "resultBsi_d " << j << ": " << resultBsi_d->getValue_with_decimal(j) << ", " << normal_mul_d[j] << " - Not matched!"<< std::endl;
        }
        else continue;
    }

    // SUM:
    std::cout << "2: SUM = one + two" << std::endl;

    auto t5 = std::chrono::high_resolution_clock::now();
    resultBsi2 = one_bsi->SUM(two_bsi);
    auto t6 = std::chrono::high_resolution_clock::now();
    std::cout << "bsi SUM duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t6-t5).count() << std::endl;

    for (int j=0; j < vector_length; j++) {
        std::cout << "resultBsi2 " << j << ": " << resultBsi2->getValue(j) << std::endl;
        if (resultBsi2->getValue(j) != normal_sum[j]) {
            std::cout << "resultBsi2 " << j << ": " << resultBsi2->getValue(j) << ", " << normal_sum[j] << " - Not matched!"<< std::endl;
        }
        else continue;
    }

    // Decimal SUM:
    std::cout << "2.5: Decimal SUM = one_dec + two" << std::endl;
    auto t5_d = std::chrono::high_resolution_clock::now();
    resultBsi2_d = one_dec_bsi->SUM(two_bsi);
    auto t6_d = std::chrono::high_resolution_clock::now();
    std::cout << "bsi SUM duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t6_d-t5_d).count() << std::endl;

    for (int j=0; j < vector_length; j++) {
        std::cout << "resultBsi2_d " << j << ": "<< resultBsi2_d->getValue_with_decimal(j) << std::endl;
        if (resultBsi2_d->getValue_with_decimal(j) != normal_sum_d[j]) {
            std::cout << "resultBsi2_d " << j << ": " << resultBsi2_d->getValue_with_decimal(j) << ", " << normal_sum_d[j] << " - Not matched!"<< std::endl;
        }
        else continue;
    }

    //Negate:
    std::cout << "3: Negate = -1 * one" << std::endl;

    auto t7 = std::chrono::high_resolution_clock::now();
    resultBsi3 = one_bsi->negate();
    auto t8 = std::chrono::high_resolution_clock::now();
    std::cout << "bsi SUM duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t8-t7).count() << std::endl;

    for (int j=0; j < vector_length; j++) {
        std::cout << "resultBsi3 " << j << ": " << resultBsi3->getValue(j) << std::endl;
        if (resultBsi3->getValue(j) != -1 * one[j]) {
            std::cout << "resultBsi3 " << j << ": " << resultBsi3->getValue(j) << ", " << -1 * one[j] << " - Not matched!"<< std::endl;
        }
        else continue;
    }

    return 0;
}
