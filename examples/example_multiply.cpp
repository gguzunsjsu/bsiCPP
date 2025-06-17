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
    int range = 50;
    int vectorLength = 10;
    std::vector<long> array1;

    for(int i=1;i<vectorLength+1;i++){
        array1.push_back(i);
    }

//    for(std::vector<long>::iterator it = array1.begin(); it!=array1.end(); ++it){
//        std::cout<<*it<<"\t";
//    }
//    std::cout<<std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    long vector_mul = 1;
    for(int i=1;i<vectorLength+1;i++){
        vector_mul*=i;
    }
    std::cout<<"Vector multiply sum: \t"<<vector_mul<<std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "vector array multiply duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << std::endl;

    /*
     * bsi multiply
     */
    BsiUnsigned<uint64_t> ubsi;
    BsiVector<uint64_t>* bsi;
    bsi = ubsi.buildBsiVectorFromVector(array1, 0.2);
    bsi->setPartitionID(0);
    bsi->setFirstSliceFlag(true);
    bsi->setLastSliceFlag(true);

    BsiVector<uint64_t>* bsi_res;

    /*
     * print values of bsi
     */
//    for(auto i=0; i<vectorLength; i++){
//        std::cout<<"bsi value "<<i<<" "<<bsi->getValue(i)<<std::endl;
//    }

    auto t3 = std::chrono::high_resolution_clock::now();
    uint64_t product = 1;
    for(size_t i=0; i<bsi->getNumberOfRows(); i++){
        product *= bsi->getValue(i);
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    std::cout << "bsi multiply duration: \t" << std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count() << std::endl;

    std::cout<<"bsi product is: \t"<<product<<std::endl;


}