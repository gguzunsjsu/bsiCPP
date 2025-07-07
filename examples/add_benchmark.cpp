#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cstdint>
using uint128_t = unsigned __int128;

#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiVector.hpp"

struct RangeInfo {
    std::size_t rangeMax;      
    int         expectedBits;  
    const char *label;
};

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    constexpr std::size_t vectorLength = 10'000'000;   // 10 M rows
    std::vector<RangeInfo> ranges = {
        {101,       7,  "0-100"},
        {1001,      10, "0-1000"},
        {1u << 16,  16, "0-65535"}
    };

    auto MB = [](double bytes) { return bytes / (1024.0 * 1024.0); };

    for (const auto &cfg : ranges) {
        std::cout << "\n================ Benchmark for range "
                  << cfg.label << " ================\n";

        std::vector<long> data(vectorLength);
        for (std::size_t i = 0; i < vectorLength; ++i)
            data[i] = std::rand() % cfg.rangeMax;

        long plainSum = 0;
        auto tAdd0 = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < vectorLength; ++i)
            plainSum += data[i];
        auto tAdd1 = std::chrono::high_resolution_clock::now();

        auto durPlain = std::chrono::duration_cast<
                            std::chrono::microseconds>(tAdd1 - tAdd0).count();

        std::size_t element_size  = sizeof(data[0]);          // 8 bytes
        std::size_t data_size     = data.size()     * element_size;
        std::size_t capacity_size = data.capacity() * element_size;

        std::size_t maxVal = *std::max_element(data.begin(), data.end());
        int bitsNeeded = 0;
        for (auto tmp = maxVal; tmp; tmp >>= 1) ++bitsNeeded;

        BsiUnsigned<uint128_t> ubsi;
        auto tBuild0 = std::chrono::high_resolution_clock::now();
        auto *bsi = ubsi.buildBsiVector(data, 0.2);
        auto tBuild1 = std::chrono::high_resolution_clock::now();
        bsi->setPartitionID(0);
        bsi->setFirstSliceFlag(true);
        bsi->setLastSliceFlag(true);    

        auto tSum0 = std::chrono::high_resolution_clock::now();
        long bsiSum = bsi->sumOfBsi();
        auto tSum1 = std::chrono::high_resolution_clock::now();

        auto durBuild  = std::chrono::duration_cast<
                            std::chrono::microseconds>(tBuild1 - tBuild0).count();
        auto durBsiSum = std::chrono::duration_cast<
                            std::chrono::microseconds>(tSum1 - tSum0).count();

        std::cout << "Element size: " << element_size
                  << " bytes (" << element_size * 8 << " bits)\n";

        std::cout << "Vector sum = " << plainSum
                  << " | time = " << durPlain << " µs"
                  << " | data mem ≈ " << std::fixed << std::setprecision(2)
                  << MB(data_size) << " MB"
                  << " | alloc mem ≈ " << MB(capacity_size) << " MB\n";

        std::cout << "BSI   sum = " << bsiSum
                  << " | build = " << durBuild << " µs"
                  << " | sum = " << durBsiSum << " µs"
                  << " | mem ≈ " << MB(bsi->getSizeInMemory()) << " MB\n";

        std::cout << "Expected bits: " << cfg.expectedBits
                  << " | bits needed by data: " << bitsNeeded
                  << " | BSI slices: " << bsi->getNumberOfSlices() << "\n";

        if (plainSum != bsiSum)
            std::cerr << "[WARNING] mismatch between plain and BSI sums!\n";

        delete bsi;
    }
    return 0;
}


/*
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cstdint>
using uint128_t = unsigned __int128;

#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiVector.hpp"

struct RangeInfo {
    std::size_t rangeMax;      
    int         expectedBits; 
    const char *label;
};

int main() {
    constexpr std::size_t vectorLength = 10'000'000;              
    std::vector<RangeInfo> ranges = {
        {101,       7,  "0-100"},
        {1001,      10, "0-1000"},
        {1u << 16,  16, "0-65535"}
    };

    auto MB = [](double bytes) { return bytes / (1024.0 * 1024.0); };

    for (const auto &cfg : ranges) {
        std::cout << "\n================ Benchmark for range "
                  << cfg.label << " ================\n";

        std::size_t plainMemBytes = 0;          
        long        plainSum      = 0;          
        std::size_t maxVal        = 0;          

        std::vector<long> dataForBsi;           
        dataForBsi.reserve(vectorLength);

        if (cfg.rangeMax <= 0x101) {
            std::vector<std::uint8_t> data8(vectorLength);
            for (std::size_t i = 0; i < vectorLength; ++i)
                data8[i] = std::rand() % cfg.rangeMax;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (std::size_t i = 0; i < vectorLength; ++i)
                plainSum += data8[i];
            auto t1 = std::chrono::high_resolution_clock::now();
            auto durPlain = std::chrono::duration_cast<
                                std::chrono::microseconds>(t1 - t0).count();

            dataForBsi.assign(data8.begin(), data8.end());
            maxVal = *std::max_element(data8.begin(), data8.end());
            plainMemBytes = vectorLength * sizeof(std::uint8_t);

            BsiUnsigned<uint128_t> ubsi;
            auto tBuild0 = std::chrono::high_resolution_clock::now();
            auto *bsi = ubsi.buildBsiVector(dataForBsi, 0.0);
            auto tBuild1 = std::chrono::high_resolution_clock::now();

            auto tSum0 = std::chrono::high_resolution_clock::now();
            long bsiSum = bsi->sumOfBsi();
            auto tSum1 = std::chrono::high_resolution_clock::now();

            auto durBuild  = std::chrono::duration_cast<
                                std::chrono::microseconds>(tBuild1 - tBuild0).count();
            auto durBsiSum = std::chrono::duration_cast<
                                std::chrono::microseconds>(tSum1 - tSum0).count();

            int bitsNeeded = 0;
            for (auto tmp = maxVal; tmp; tmp >>= 1) ++bitsNeeded;

            std::cout << "Vector sum = " << plainSum
                      << " | time = " << durPlain << " µs"
                      << " | mem ≈ " << std::fixed << std::setprecision(2)
                      << MB(plainMemBytes) << " MB\n";

            std::cout << "BSI   sum = " << bsiSum
                      << " | build = " << durBuild << " µs"
                      << " | sum = " << durBsiSum << " µs"
                      << " | mem ≈ " << MB(bsi->getSizeInMemory()) << " MB\n";

            std::cout << "Expected bits: " << cfg.expectedBits
                      << " | bits needed by data: " << bitsNeeded
                      << " | BSI slices: " << bsi->getNumberOfSlices() << "\n";

            if (plainSum != bsiSum)
                std::cerr << "Mismatch between plain and BSI sums!\n";

            delete bsi;                                  
        }
        else {
            std::vector<std::uint16_t> data16(vectorLength);
            for (std::size_t i = 0; i < vectorLength; ++i)
                data16[i] = std::rand() % cfg.rangeMax;

            auto t0 = std::chrono::high_resolution_clock::now();
            for (std::size_t i = 0; i < vectorLength; ++i)
                plainSum += data16[i];
            auto t1 = std::chrono::high_resolution_clock::now();
            auto durPlain = std::chrono::duration_cast<
                                std::chrono::microseconds>(t1 - t0).count();

            dataForBsi.assign(data16.begin(), data16.end());
            maxVal = *std::max_element(data16.begin(), data16.end());
            plainMemBytes = vectorLength * sizeof(std::uint16_t);

            BsiUnsigned<uint128_t> ubsi;
            auto tBuild0 = std::chrono::high_resolution_clock::now();
            auto *bsi = ubsi.buildBsiVector(dataForBsi, 0.0);
            auto tBuild1 = std::chrono::high_resolution_clock::now();

            bsi->setPartitionID(0);
            bsi->setFirstSliceFlag(true);
            bsi->setLastSliceFlag(true);

            auto tSum0 = std::chrono::high_resolution_clock::now();
            long bsiSum = bsi->sumOfBsi();
            auto tSum1 = std::chrono::high_resolution_clock::now();

            auto durBuild  = std::chrono::duration_cast<
                                std::chrono::microseconds>(tBuild1 - tBuild0).count();
            auto durBsiSum = std::chrono::duration_cast<
                                std::chrono::microseconds>(tSum1 - tSum0).count();

            int bitsNeeded = 0;
            for (auto tmp = maxVal; tmp; tmp >>= 1) ++bitsNeeded;

            std::cout << "Vector sum = " << plainSum
                      << " | time = " << durPlain << " µs"
                      << " | mem ≈ " << std::fixed << std::setprecision(2)
                      << MB(plainMemBytes) << " MB\n";

            std::cout << "BSI   sum = " << bsiSum
                      << " | build = " << durBuild << " µs"
                      << " | sum = " << durBsiSum << " µs"
                      << " | mem ≈ " << MB(bsi->getSizeInMemory()) << " MB\n";

            std::cout << "Expected bits: " << cfg.expectedBits
                      << " | bits needed by data: " << bitsNeeded
                      << " | BSI slices: " << bsi->getNumberOfSlices() << "\n";

            if (plainSum != bsiSum)
                std::cerr << "Mismatch between plain and BSI sums!\n";

            delete bsi;
        }
    }
    return 0;
}
*/