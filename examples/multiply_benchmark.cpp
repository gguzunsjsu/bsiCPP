#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cstdint>
using uint128_t = unsigned __int128;

#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiVector.hpp"

struct RangeInfo {
    std::size_t rangeMax;
    int         expectedBits;
    const char* label;
};

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    constexpr std::size_t N = 10'000'000;
    std::vector<RangeInfo> ranges = {
        {101,      7,  "0-100"},
        {1001,     10, "0-1000"},
        {1u<<16,   16, "0-65535"}
    };

    auto MB = [](double bytes){ return bytes / (1024.0*1024.0); };

    for (auto& cfg : ranges) {
        std::cout << "\n=== Benchmark range " << cfg.label << " ===\n";

        std::vector<long> A(N), B(N);
        for (std::size_t i = 0; i < N; ++i) {
            A[i] = std::rand() % cfg.rangeMax;
            B[i] = std::rand() % cfg.rangeMax;
        }

        std::vector<long> R(N);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < N; ++i)
            R[i] = A[i] * B[i];
        auto t1 = std::chrono::high_resolution_clock::now();
        auto durPlain = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        long plainSum = 0;
        for (auto v : R)
            plainSum += v;
        size_t eSz = sizeof(A[0]);
        size_t usedA = A.size()*eSz, allocA = A.capacity()*eSz;
        size_t usedB = B.size()*eSz, allocB = B.capacity()*eSz;
        size_t usedR = R.size()*sizeof(R[0]), allocR = R.capacity()*sizeof(R[0]);

        BsiUnsigned<uint128_t> ubsi;
        auto tb0 = std::chrono::high_resolution_clock::now();
        auto *bA = ubsi.buildBsiVector(A, 0.2);
        auto *bB = ubsi.buildBsiVector(B, 0.2);
        bA->setPartitionID(0); bA->setFirstSliceFlag(true); bA->setLastSliceFlag(true);
        bB->setPartitionID(0); bB->setFirstSliceFlag(true); bB->setLastSliceFlag(true);
        auto tb1 = std::chrono::high_resolution_clock::now();
        auto durBuild = std::chrono::duration_cast<std::chrono::microseconds>(tb1 - tb0).count();

        auto tm0 = std::chrono::high_resolution_clock::now();
        auto *bR = bA->multiplyWithBsiHorizontal(bB);
        auto tm1 = std::chrono::high_resolution_clock::now();
        auto durBSI = std::chrono::duration_cast<std::chrono::microseconds>(tm1 - tm0).count();

        long bsiSum = bR->sumOfBsi();

        long maxA = *std::max_element(A.begin(), A.end());
        int bitsNeeded = 0; for (auto v = maxA; v; v >>= 1) ++bitsNeeded;

        std::cout << "Plain mul sum=" << plainSum
                  << " | time=" << durPlain << " µs\n"
                  << "  A mem used/alloc=" << std::fixed<<std::setprecision(2)
                  << MB(usedA) << "/" << MB(allocA) << " MB\n"
                  << "  B mem used/alloc=" << MB(usedB) << "/" << MB(allocB) << " MB\n"
                  << "  R mem used/alloc=" << MB(usedR) << "/" << MB(allocR) << " MB\n";

        std::cout << "BSI mul | build=" << durBuild << " µs"
                  << " | mul=" << durBSI << " µs"
                  << " | sum=" << bsiSum << "\n"
                  << "  BSI A mem=" << MB(bA->getSizeInMemory()) << " MB"
                  << " | BSI B mem=" << MB(bB->getSizeInMemory()) << " MB"
                  << " | BSI R mem=" << MB(bR->getSizeInMemory()) << " MB\n";

        std::cout << "Expected bits=" << cfg.expectedBits
                  << " | bitsNeeded=" << bitsNeeded
                  << " | slices A=" << bA->getNumberOfSlices()
                  << " | slices B=" << bB->getNumberOfSlices()
                  << " | slices R=" << bR->getNumberOfSlices() << "\n";

        if (plainSum != bsiSum)
            std::cerr << "[WARN] mismatch plain vs BSI mul\n";

        delete bA; delete bB; delete bR;
    }

    return 0;
}