#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <memory>
#include <ctime>

#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiVector.hpp"

int main() {
    const size_t vectorLength = 10000000;
    std::vector<uint16_t> vector1;
    std::vector<uint16_t> vector2;

    int range = 1000;

    for(size_t i = 0; i < vectorLength; i++) {
        vector1.push_back(std::rand() % range);
        vector2.push_back(std::rand() % range);
    }

    /*
     * Normal multiplication
     */
    std::vector<long> mul_res(vectorLength);
    auto t3 = std::chrono::high_resolution_clock::now();
    for(auto i=0; i<vectorLength; i++){
        mul_res[i] = vector1[i] * vector2[i];
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    auto normalMul_duration = std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count();
    long normalMul_sum = 0;
    for(auto i=0; i<vectorLength; i++){
        normalMul_sum += mul_res[i];
    }

    std::cout << "Normal multiplication duration: " << normalMul_duration << std::endl;
    std::cout << "Normal multiplication result sum: " << normalMul_sum << std::endl;

    size_t element_size = sizeof(vector1[0]);
    size_t data_size = vector1.size() * element_size;
    size_t capacity_size = vector1.capacity() * element_size;

    std::cout << "Element size: " << element_size << " bytes (" << element_size * 8 << " bits)" << std::endl;
    std::cout << "Data memory: " << data_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Allocated memory: " << capacity_size / (1024.0 * 1024.0) << " MB" << std::endl;

    size_t max_value = 0;
    for (const auto& val : vector1) {
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
    std::cout << "Actual bits used per element: " << sizeof(vector1[0]) * 8 << std::endl;
    std::cout << "Potential memory waste: " <<
              (sizeof(vector1[0]) * 8 - bits_needed) * vector1.size() / 8 / (1024.0 * 1024.0) << " MB" << std::endl;

    /*
     * bsi multiplication
     */
    try {
        BsiUnsigned<uint64_t> ubsi;
        double compressionThreshold = 0.2;

        std::vector<long> array1_long(vector1.begin(), vector1.end());
        std::vector<long> array2_long(vector2.begin(), vector2.end());

        BsiVector<uint64_t>* bsi1 = ubsi.buildBsiVector(array1_long, compressionThreshold);
        BsiVector<uint64_t>* bsi2 = ubsi.buildBsiVector(array2_long, compressionThreshold);

        std::cout << "Memory used to store bsi1 attribute: \t" << bsi1->getSizeInMemory()/(pow(2, 20)) << std::endl;
        std::cout << "Memory used to store bsi2 attribute: \t" << bsi2->getSizeInMemory()/(pow(2, 20)) << std::endl;



        if (!bsi1 || !bsi2) {
            std::cerr << "Failed to create BSI structures" << std::endl;
            delete bsi1;
            delete bsi2;
            return 1;
        }

        std::cout << "Performing BSI multiplication..." << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        BsiVector<uint64_t>* resultBsi = bsi1->multiplyWithBsiHorizontal(bsi2);
        auto t2 = std::chrono::high_resolution_clock::now();

        if (!resultBsi) {
            std::cerr << "Multiplication failed" << std::endl;
            delete bsi1;
            delete bsi2;
            return 1;
        }

        std::cout << "BSI multiplication duration: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()
                  << " microseconds" << std::endl;


//        long bsiMul_sum = 0;
//        for(size_t i = 0; i < vectorLength; i++) {
//            long bsiValue = resultBsi->getValue(i);
//            long expectedValue = vector1[i] * vector2[i];
//            bsiMul_sum += bsiValue;
//
//            std::cout << i << ": " << vector1[i] << " * " << vector2[i]
//                      << " = " << bsiValue << " (Expected: "
//                      << expectedValue << ")" << std::endl;
//        }
//
//        std::cout << "BSI multiplication manual sum: " << bsiMul_sum << std::endl;
        std::cout << "BSI sumOfBsi() method result: " << resultBsi->sumOfBsi() << std::endl;

        // Clean up
        delete bsi1;
        delete bsi2;
        delete resultBsi;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return 1;
    }

    return 0;
}