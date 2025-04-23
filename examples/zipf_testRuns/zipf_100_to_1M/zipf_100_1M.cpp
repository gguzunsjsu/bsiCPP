//
// Created by poorna on 4/23/25.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

#include "../../../bsi/BsiUnsigned.hpp"
#include "../../../bsi/BsiSigned.hpp"
#include "../../../bsi/BsiAttribute.hpp"
#include "../../../bsi/hybridBitmap/hybridbitmap.h"
#include "../../../bsi/hybridBitmap/UnitTestsOfHybridBitmap.hpp"

using Clock = std::chrono::high_resolution_clock;
using us    = std::chrono::microseconds;

double parse_skew(const std::string &fn) {
    auto p = fn.find("skew");
    if (p == std::string::npos) return 0.0;
    p += 4;
    auto q = fn.find('_', p);
    std::string tok = fn.substr(p, q - p);
    if      (tok == "05")  return 0.5;
    else if (tok == "15")  return 1.5;
    else                   return std::stod(tok);
}

int main(){
    // list of files
    std::vector<std::string> files = {
            "rows100_skew1_card16.txt",
            "rows1000_skew1_card16.txt",
            "rows10k_skew1_card16.txt",
            "rows100k_skew1_card16.txt",
            "rows1M_skew1_card16.txt",
            "rows10M_skew1_card16.txt"
    };

    const int RUNS = 5;
    const double COMPRESSION_THRESHOLD = 0.5;
//    const int    MAX_SLICES           = 16;

    std::ofstream out("results_100_1M.csv");
    out << "skew,bsi_avg_us,vec_avg_us\n";

    for (auto &fn : files) {
        // --- load A,B from file ---
        std::vector<long> A, B;
        A.reserve(10000);
        B.reserve(10000);

        std::ifstream fin(fn);
        if (!fin) {
            std::cerr << "ERROR: cannot open " << fn << "\n";
            continue;
        }
        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            long a,b;
            char comma;
            if (!(iss >> a >> comma >> b)) continue;
            A.push_back(a);
            B.push_back(b);
        }
        fin.close();

        size_t N = A.size();
        double skew = parse_skew(fn);
        std::cout << "File: " << fn
                  << "  (skew="<<skew<<", N="<<N<<")\n";

        // --- prepare BSI builder once ---
        BsiUnsigned<uint64_t> ubsi;
//        ubsi.setCompressionThreshold(COMPRESSION_THRESHOLD);
//        ubsi.setMaxNumBitSlices(MAX_SLICES);

        long total_vec_us = 0;
        long total_bsi_us = 0;

        for (int run = 0; run < RUNS; run++) {
            // vector dot
            auto t0 = Clock::now();
            long vec_dot = 0;
            for (size_t i = 0; i < N; i++)
                vec_dot += A[i] * B[i];
            auto t1 = Clock::now();
            long vec_us = std::chrono::duration_cast<us>(t1 - t0).count();
            total_vec_us += vec_us;

            // build BSI attributes
            std::vector<long> A_long(A.begin(), A.end());
            std::vector<long> B_long(B.begin(), B.end());
            auto *bsiA = ubsi.buildBsiAttributeFromVector(A_long, COMPRESSION_THRESHOLD);
            auto *bsiB = ubsi.buildBsiAttributeFromVector(B_long, COMPRESSION_THRESHOLD);
            bsiA->setFirstSliceFlag(true);
            bsiA->setLastSliceFlag(true);
            bsiB->setFirstSliceFlag(true);
            bsiB->setLastSliceFlag(true);

//            cout << "Number of slices in bsiA: \t" << bsiA->getNumberOfSlices() << endl;
//            cout << "Number of slices in bsiB: \t" << bsiB->getNumberOfSlices() << endl;
            // time for BSI dot
            auto t2 = Clock::now();
            long bsi_dot = bsiA->dot(bsiB);
            auto t3 = Clock::now();
            long bsi_us = std::chrono::duration_cast<us>(t3 - t2).count();
            total_bsi_us += bsi_us;

            delete bsiA;
            delete bsiB;
        }

        double avg_vec = total_vec_us / double(RUNS);
        double avg_bsi = total_bsi_us / double(RUNS);

        // print & CSV
        std::cout << std::fixed << std::setprecision(2)
                  << "   avg vector: " << avg_vec << " µs"
                  << "   avg BSI: "   << avg_bsi << " µs\n\n";

        out << skew << ","
            << avg_bsi << ","
            << avg_vec << "\n";
    }

    out.close();
    std::cout << "Wrote results.csv (averaged over " << RUNS << " runs)\n";
    return 0;
}
