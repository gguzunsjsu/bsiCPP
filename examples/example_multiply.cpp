#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <memory>
#include <ctime>

#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"

int main() {
    const size_t vectorLength = 10000000;
    std::vector<long> vector1;
    std::vector<long> vector2;

    int range = 50;

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

    /*
     * bsi multiplication
     */
    try {
        BsiUnsigned<uint64_t> ubsi;
        double compressionThreshold = 0.2;

        BsiAttribute<uint64_t>* bsi1 = ubsi.buildBsiAttributeFromVector(vector1, compressionThreshold);
        BsiAttribute<uint64_t>* bsi2 = ubsi.buildBsiAttributeFromVector(vector2, compressionThreshold);

        if (!bsi1 || !bsi2) {
            std::cerr << "Failed to create BSI structures" << std::endl;
            delete bsi1;
            delete bsi2;
            return 1;
        }

        std::cout << "Performing BSI multiplication..." << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        BsiAttribute<uint64_t>* resultBsi = bsi1->multiplyWithBsiHorizontal(bsi2);
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