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
#include <vector>
#include <immintrin.h>

#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiVector.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"
#include "../bsi/hybridBitmap/UnitTestsOfHybridBitmap.hpp"
typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;



int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    int one_range = 100;
    int two_range = 100;
    int vectorLength = 100000;
    int decimalPlaces = 3;

//    std::vector<long> one = {2,6,9, 10, 50};
//    std::vector<long> two = {2,6,9, 10, 50};
//
//    std::vector<long> one = {38, 5, 12, 50, 18, 7, 3, 32, 42, 43, 15, 35, 39, 20, 39, 34, 46, 8, 13, 23};
//    std::vector<long> two = {31, 12, 23, 48, 25, 16, 38, 6, 28, 9, 8, 46, 4, 6, 31, 8, 43, 11, 25, 16};

//    std::vector<long> one = {40, 30, 12, 8, 14, 0, 37, 33, 37, 5, 29, 49, 26, 32, 50, 27, 8, 13, 9, 19, 7,
//                             47, 26, 34, 25, 14, 6, 11, 14, 16, 38, 42, 45, 16, 14, 3, 7, 45, 29, 45};
//    std::vector<long> two = {39, 25, 47, 12, 5, 10, 30, 5, 1, 26, 9, 9, 2, 47, 29, 48, 28, 29, 13, 27,
//                             30, 1, 43, 17, 46, 7, 45, 1, 27, 41, 1, 31, 43, 12, 35, 17, 26, 29, 47, 42};


//    std::vector<long> one = {3845, 8983, 2462, 5143, 4161, 1682, 858, 6666, 1618, 8762, 4727, 4496, 7324, 5405, 8503, 4113, 6344, 4765,
//                             9532, 6318, 5836, 1355, 7574, 5366, 923, 8105, 4128, 5522, 8079, 5614, 9500, 1124, 9662, 8096, 4215, 5805, 8636, 3055, 4342, 3306};
//    std::vector<long> two = {9173, 6213, 7136, 3150, 7709, 3049, 2233, 2729, 4630, 5721, 6472, 7239, 8922, 3978, 8275, 5955, 1584, 6472, 6248, 894, 2728,
//                             2645, 7825, 4747, 7680, 3204, 1729, 2069, 4722, 7085, 7129, 3161, 5333, 4241, 9510, 3614, 3695, 2781, 352, 3647};
    std::vector<double> one;
    std::vector<double> two;
    for(int i = 0; i < vectorLength; i++) {
        //one.push_back(1 + (gen() % one_range));
        one.push_back(static_cast<double>(rand()) / RAND_MAX);
        two.push_back(static_cast<double>(rand()) / RAND_MAX);


        //two.push_back(-1*two_range + rand() % (two_range + two_range + 1));
       // two.push_back(1 + (gen() % two_range));
    }


    /*
    std::vector<long> dividends;
    std::vector<long> divisors;
    std::vector<long> array_two;
    std::vector<long> array_x;

    dividends = {2,6,9};
    divisors= {2,3,4};
    array_two= {2,2,2};
    array_x= {1,2,2};


    BsiSigned<uint64_t> ubsi;

    BsiVector<uint64_t>* dividend_bsi = ubsi.buildBsiAttributeFromVector(dividends, 0.2);
    dividend_bsi->setFirstSliceFlag(true);
    dividend_bsi->setLastSliceFlag(true);
    dividend_bsi->setPartitionID(0);

    BsiVector<uint64_t>* divisor_bsi = ubsi.buildBsiAttributeFromVector(divisors, 0.2);
    divisor_bsi->setFirstSliceFlag(true);
    divisor_bsi->setLastSliceFlag(true);
    divisor_bsi->setPartitionID(0);

    BsiVector<uint64_t>* two_bsi = ubsi.buildBsiAttributeFromVector(array_two, 0.2);
    two_bsi->setFirstSliceFlag(true);
    two_bsi->setLastSliceFlag(true);
    two_bsi->setPartitionID(0);

    BsiVector<uint64_t>* x_bsi = ubsi.buildBsiVectorFromVector(array_x, 0.2);
    x_bsi->setFirstSliceFlag(true);
    x_bsi->setLastSliceFlag(true);
    x_bsi->setPartitionID(0);


    BsiVector<uint64_t>* resultBsi;
    BsiVector<uint64_t>* resultBsi2;

    auto t3 = std::chrono::high_resolution_clock::now();

    // std::cout << "1: Multiplication - divisor*x_bsi" << std::endl;
    // resultBsi = divisor_bsi->multiplication(x_bsi);
    //
    // std::cout << "resultBsi numSlices: " << resultBsi << std::endl;
    // std::cout << "resultBsi 0: " << resultBsi->getValue(0) << std::endl;
    // std::cout << "resultBsi 1: " << resultBsi->getValue(1) << std::endl;
    // std::cout << "resultBsi 2: " << resultBsi->getValue(2) << std::endl;

    std::cout << "2: SUM - two + result" << std::endl;
    resultBsi2 = two_bsi->SUM(x_bsi);

    std::cout << "resultBsi2 numSlices: " << resultBsi2 << std::endl;
    std::cout << "resultBsi2 0: " << resultBsi2->getValue(0) << std::endl;
    std::cout << "resultBsi2 1: " << resultBsi2->getValue(1) << std::endl;
    std::cout << "resultBsi2 2: " << resultBsi2->getValue(2) << std::endl;

    auto t4 = std::chrono::high_resolution_clock::now();

    */

//    BsiSigned<uint64_t> ubsi;
    BsiUnsigned<uint128_t> ubsi;


    std::cout << "Building bit vectors..." << std::endl;
    auto t9 = std::chrono::high_resolution_clock::now();


    BsiVector<uint128_t>* one_bsi = ubsi.buildBsiVector(one, decimalPlaces, 0.2);
    one_bsi->setFirstSliceFlag(true);
    one_bsi->setLastSliceFlag(true);
    one_bsi->setPartitionID(0);

    BsiVector<uint128_t>* two_bsi = ubsi.buildBsiVector(two, decimalPlaces,0.2);
    two_bsi->setFirstSliceFlag(true);
    two_bsi->setLastSliceFlag(true);
    two_bsi->setPartitionID(0);

    auto t10 = std::chrono::high_resolution_clock::now();
    std::cout << "Time to build bitVectors: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t10-t9).count() << std::endl;



    BsiVector<uint128_t>* resultBsi;
    BsiVector<uint128_t>* resultBsi2;
    BsiVector<uint128_t>* resultBsi3;
    BsiVector<uint128_t>* resultBsi_4;
    BsiVector<uint128_t>* resultBsi5;
    std::cout << "Slices in bsi One: " << one_bsi->numSlices << std::endl;
    std::cout << "Slices in bsi Two: " << two_bsi->numSlices << std::endl;

    std::cout << "Second vector: " << std::endl;

    std::cout << "vector two bsi 1: " << two_bsi->getValue(0) << std::endl;
    std::cout << "vector two bsi 2: " << two_bsi->getValue(1) << std::endl;
    std::cout << "vector two bsi 3: " << two_bsi->getValue(2) << std::endl;
    std::cout << "vector two bsi 4: " << two_bsi->getValue(3) << std::endl;
    std::cout << "vector two 1: " << two[0] << std::endl;
    std::cout << "vector two 2: " << two[1] << std::endl;
    std::cout << "vector two 3: " << two[2] << std::endl;
    std::cout << "vector two 4: " << two[3] << std::endl;


    //multiplication:
    std::cout << "Bsi Multiplication: " << std::endl;

    auto t3 = std::chrono::high_resolution_clock::now();
    resultBsi = one_bsi->multiplication(two_bsi);
    auto t4 = std::chrono::high_resolution_clock::now();

    std::cout << "resultBsi numSlices: " << resultBsi << std::endl;
    std::cout << "resultBsi 0: " << resultBsi->getValue(one.size()-1) << std::endl;
    std::cout << "resultBsi 1: " << resultBsi->getValue(one.size()-2) << std::endl;
    std::cout << "resultBsi 2: " << resultBsi->getValue(one.size()-3) << std::endl;
    std::cout << "resultBsi 3: " << resultBsi->getValue(one.size()-4) << std::endl;
    std::cout << "resultBsi 4: " << resultBsi->getValue(one.size()-5) << std::endl;
    std::cout << "result 0: " << one[one.size()-1]*two[one.size()-1] << std::endl;
    std::cout << "result 1: " << one[one.size()-2]*two[one.size()-2] << std::endl;
    std::cout << "result 2: " << one[one.size()-3]*two[one.size()-3] << std::endl;
    std::cout << "result 3: " << one[one.size()-4]*two[one.size()-4] << std::endl;
    std::cout << "result 4: " << one[one.size()-5]*two[one.size()-5] << std::endl;

    std::cout << "bsi multiplication duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count() << std::endl;

    // SUM:
    std::cout << "2: SUM = one + two" << std::endl;

    auto t5 = std::chrono::high_resolution_clock::now();
    resultBsi2 = one_bsi->SUM(two_bsi);
    auto t6 = std::chrono::high_resolution_clock::now();

    std::cout << "resultBsi2 numSlices: " << resultBsi2 << std::endl;
    std::cout << "resultBsi 0: " << resultBsi2->getValue(one.size()-1) << std::endl;
    std::cout << "resultBsi 1: " << resultBsi2->getValue(one.size()-2) << std::endl;
    std::cout << "resultBsi 2: " << resultBsi2->getValue(one.size()-3) << std::endl;
    std::cout << "resultBsi 3: " << resultBsi2->getValue(one.size()-4) << std::endl;
    std::cout << "resultBsi 4: " << resultBsi2->getValue(one.size()-5) << std::endl;
    std::cout << "result 0: " << one[one.size()-1]+two[one.size()-1] << std::endl;
    std::cout << "result 1: " << one[one.size()-2]+two[one.size()-2] << std::endl;
    std::cout << "result 2: " << one[one.size()-3]+two[one.size()-3] << std::endl;
    std::cout << "result 3: " << one[one.size()-4]+two[one.size()-4] << std::endl;
    std::cout << "result 4: " << one[one.size()-5]+two[one.size()-5] << std::endl;

    std::cout << "bsi SUM duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t6-t5).count() << std::endl;


    //multiplyByConstant:
    std::cout << "3: multiplyByConstant = one * const" << std::endl;

    auto t7 = std::chrono::high_resolution_clock::now();
    int constant = 10;

    resultBsi3 = one_bsi->multiplyByConstant(constant);

    std::cout << "resultBsi3 numSlices: " << resultBsi3 << std::endl;
    std::cout << "resultBsi 0: " << resultBsi3->getValue(one.size()-1) << std::endl;
    std::cout << "resultBsi 1: " << resultBsi3->getValue(one.size()-2) << std::endl;
    std::cout << "resultBsi 2: " << resultBsi3->getValue(one.size()-3) << std::endl;
    std::cout << "resultBsi 3: " << resultBsi3->getValue(one.size()-4) << std::endl;
    std::cout << "resultBsi 4: " << resultBsi3->getValue(one.size()-5) << std::endl;
    std::cout << "result 0: " << one[one.size()-1]*constant << std::endl;
    std::cout << "result 1: " << one[one.size()-2]*constant << std::endl;
    std::cout << "result 2: " << one[one.size()-3]*constant << std::endl;
    std::cout << "result 3: " << one[one.size()-4]*constant << std::endl;
    std::cout << "result 4: " << one[one.size()-5]*constant << std::endl;


//    std::cout << "resultBsi3 numSlices: " << one_bsi << std::endl;
//    std::cout << "resultBsi3 0: " << one_bsi->getValue(0) << std::endl;
//    std::cout << "resultBsi3 1: " << one_bsi->getValue(1) << std::endl;
//    std::cout << "resultBsi3 2: " << one_bsi->getValue(2) << std::endl;
//    std::cout << "resultBsi3 3: " << one_bsi->getValue(3) << std::endl;
//    std::cout << "resultBsi3 4: " << one_bsi->getValue(4) << std::endl;

    auto t8 = std::chrono::high_resolution_clock::now();
    std::cout << "bsi multiplyByConstant duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t8-t7).count() << std::endl;



    //multiplyBSI:
  //  std::cout << "4: multiplyBSI = one * two" << std::endl;
//    auto t9 = std::chrono::high_resolution_clock::now();


//    BsiVector<uint64_t>* resultBsi3 = one_bsi->multiplyByConstant(constant);
//    resultBsi_4 = one_bsi->multiplyBSI(two_bsi);

 ///   auto t10 = std::chrono::high_resolution_clock::now();
    //
    // std::cout << "resultBsi_4 numSlices: " << resultBsi_4 << std::endl;
    // std::cout << "resultBsi 0: " << resultBsi_4->getValue(one.size()-1) << std::endl;
    // std::cout << "resultBsi 1: " << resultBsi_4->getValue(one.size()-2) << std::endl;
    // std::cout << "resultBsi 2: " << resultBsi_4->getValue(one.size()-3) << std::endl;
    // std::cout << "resultBsi 3: " << resultBsi_4->getValue(one.size()-4) << std::endl;
    // std::cout << "resultBsi 4: " << resultBsi_4->getValue(one.size()-5) << std::endl;
    // std::cout << "result 0: " << one[one.size()-1]*constant << std::endl;
    // std::cout << "result 1: " << one[one.size()-2]*constant << std::endl;
    // std::cout << "result 2: " << one[one.size()-3]*constant << std::endl;
    // std::cout << "result 3: " << one[one.size()-4]*constant << std::endl;
    // std::cout << "result 4: " << one[one.size()-5]*constant << std::endl;
    //
    //std::cout << "bsi mltiply horizontal duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t10-t9).count() << std::endl;



// MultiplyWithBsi:
    std::cout << "Multiplication horizontal precision" << std::endl;

    auto t11 = std::chrono::high_resolution_clock::now();
    resultBsi5 = one_bsi->multiplyWithBsiHorizontal(two_bsi, 5);
    auto t12 = std::chrono::high_resolution_clock::now();

    std::cout << "resultBsi2 numSlices: " << resultBsi5 << std::endl;
    std::cout << "resultBsi 0: " << resultBsi5->getValue(one.size()-1) << std::endl;
    std::cout << "resultBsi 1: " << resultBsi5->getValue(one.size()-2) << std::endl;
    std::cout << "resultBsi 2: " << resultBsi5->getValue(one.size()-3) << std::endl;
    std::cout << "resultBsi 3: " << resultBsi5->getValue(one.size()-4) << std::endl;
    std::cout << "resultBsi 4: " << resultBsi5->getValue(one.size()-5) << std::endl;
    std::cout << "result 0: " << one[one.size()-1]*two[one.size()-1] << std::endl;
    std::cout << "result 1: " << one[one.size()-2]*two[one.size()-2] << std::endl;
    std::cout << "result 2: " << one[one.size()-3]*two[one.size()-3] << std::endl;
    std::cout << "result 3: " << one[one.size()-4]*two[one.size()-4] << std::endl;
    std::cout << "result 4: " << one[one.size()-5]*two[one.size()-5] << std::endl;

    std::cout << "bsi Multiply duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t12-t11).count() << std::endl;


    // Dot product:
    std::cout << "Dot product" << std::endl;

    auto t13 = std::chrono::high_resolution_clock::now();
    double dotres = one_bsi->dotProduct(two_bsi);
    auto t14 = std::chrono::high_resolution_clock::now();
    std::cout << "Dot product: " << dotres << std::endl;
    std::cout << "Dot Product time: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t14-t13).count() << std::endl;


    // Dot product:
    std::cout << "Dot " << std::endl;

    auto t15 = std::chrono::high_resolution_clock::now();
    dotres = one_bsi->dot(two_bsi);
    auto t16 = std::chrono::high_resolution_clock::now();
    std::cout << "Dot: " << dotres/(double)pow(10,2*decimalPlaces) << std::endl;
    std::cout << "Dot time: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t16-t15).count() << std::endl;

    // Dot product:
    std::cout << "Dot vector " << std::endl;

    auto t17 = std::chrono::high_resolution_clock::now();
    dotres = 0;
    for (int i=0; i<one.size(); i++) {
        dotres += one[i]*two[i];
    }
    auto t18 = std::chrono::high_resolution_clock::now();
    std::cout << "Dot vector: " << dotres << std::endl;
    std::cout << "Dot vector time: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t18-t17).count() << std::endl;


    return 0;
}