//
// Created by poorna on 2/10/25.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cstdlib>
#include <vector>

#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiAttribute.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"
#include "../bsi/hybridBitmap/UnitTestsOfHybridBitmap.hpp"

int main(){
    std::vector<long> dividendArray;
    int range = 50;
    int vectorLength = 1000000;  // one million elements

    // Create dividend array with random numbers.
    for(int i=0; i<vectorLength; i++){
        dividendArray.push_back(std::rand() % range);
    }

    // ---------------- Normal Division Test ----------------
    // Use a constant divisor.
    long divisorConstant = 7;
    std::vector<long> normalQuotient(vectorLength);
    std::vector<long> normalRemainder(vectorLength);

    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<vectorLength; i++){
        normalQuotient[i] = dividendArray[i] / divisorConstant;
        normalRemainder[i] = dividendArray[i] % divisorConstant;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto normal_duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "Normal division duration: " << normal_duration << " microseconds" << std::endl;

    // ---------------- BSI-Based Division Test ----------------
    // Build the BSI attribute for the dividend.
    BsiUnsigned<uint64_t> ubsi;
    BsiAttribute<uint64_t>* dividendBSI = ubsi.buildBsiAttributeFromVector(dividendArray, 0.2);
    dividendBSI->setFirstSliceFlag(true);
    dividendBSI->setLastSliceFlag(true);
    dividendBSI->setPartitionID(0);

    // Build the divisor BSI (constant value applied to all rows).
    BsiAttribute<uint64_t>* divisorBSI = ubsi.buildQueryAttribute(divisorConstant, vectorLength, 0);

    // Cast them to BsiUnsigned.
    BsiUnsigned<uint64_t>* uDividend = dynamic_cast<BsiUnsigned<uint64_t>*>(dividendBSI);
    BsiUnsigned<uint64_t>* uDivisor  = dynamic_cast<BsiUnsigned<uint64_t>*>(divisorBSI);
    if(!uDividend || !uDivisor){
        std::cerr << "Error: Could not cast BsiAttribute to BsiUnsigned." << std::endl;
        return 1;
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    // Perform division. (Only the divisor is passed, since uDividend is the dividend.)
    auto divResult = uDividend->divide(*uDivisor);
    auto t4 = std::chrono::high_resolution_clock::now();
    auto bsi_divide_duration = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    // Retrieve quotient and remainder.
    BsiAttribute<uint64_t>* quotientBSI = divResult.first;
    BsiAttribute<uint64_t>* remainderBSI = divResult.second;

    // For testing, we use sumOfBsi() as a simple scalar summary.
    long quotientSum = quotientBSI->sumOfBsi();
    long remainderSum = remainderBSI->sumOfBsi();

    std::cout << "BSI division results:" << std::endl;
    std::cout << "Quotient (sumOfBsi): " << quotientSum << std::endl;
    std::cout << "Remainder (sumOfBsi): " << remainderSum << std::endl;
    std::cout << "BSI division duration: " << bsi_divide_duration << " microseconds" << std::endl;

    // Cleanup
    delete dividendBSI;
    delete divisorBSI;
    delete quotientBSI;
    delete remainderBSI;

    return 0;
}