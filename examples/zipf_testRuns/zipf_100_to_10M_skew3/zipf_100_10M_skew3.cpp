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
    if      (tok == "05") return 0.5;
    else if (tok == "15") return 1.5;
    else                  return std::stod(tok);
}

size_t parse_size(const std::string &fn) {
    auto p = fn.find("rows");
    auto q = fn.find("_skew", p);
    std::string tok = fn.substr(p + 4, q - (p + 4));  // e.g. "10k", "1M", "1000"

    if (!tok.empty()) {
        char suf = tok.back();
        if (suf=='k' || suf=='K') {
            // kilo
            size_t v = std::stoul( tok.substr(0, tok.size()-1) );
            return v * 1'000;
        }
        else if (suf=='m' || suf=='M') {
            // mega
            size_t v = std::stoul( tok.substr(0, tok.size()-1) );
            return v * 1'000'000;
        }
    }

    return std::stoul(tok);
}


int main(){
    std::vector<std::string> files = {
            "rows100_skew3_card16.txt",
            "rows1000_skew3_card16.txt",
            "rows10k_skew3_card16.txt",
            "rows100k_skew3_card16.txt",
            "rows1M_skew3_card16.txt",
            "rows10M_skew3_card16.txt"
    };

    const int RUNS = 5;
    const double COMPRESSION_THRESHOLD = 0.5;

    std::ofstream out("results_100_1M_with_size.csv");
    out << "skew,size,bsi_avg_us,vec_avg_us\n";

    for (auto &fn : files) {
        // load A,B
        std::vector<long> A, B;
        std::ifstream fin(fn);
        if (!fin) {
            std::cerr << "ERROR: cannot open " << fn << "\n";
            continue;
        }
        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            long a,b; char c;
            if (iss >> a >> c >> b) {
                A.push_back(a);
                B.push_back(b);
            }
        }
        fin.close();

        size_t N    = A.size();
        double skew = parse_skew(fn);

        // for CSV
        size_t size_val = parse_size(fn);

        BsiUnsigned<uint64_t> ubsi;
        long total_vec_us = 0, total_bsi_us = 0;

        for (int run = 0; run < RUNS; run++) {
            // vector dot
            auto t0 = Clock::now();
            long vec_dot = 0;
            for (auto i = 0u; i < N; i++)
                vec_dot += A[i] * B[i];
            auto t1 = Clock::now();
            total_vec_us += std::chrono::duration_cast<us>(t1 - t0).count();

            // BSI dot
            auto *bsiA = ubsi.buildBsiAttributeFromVector(A, COMPRESSION_THRESHOLD);
            auto *bsiB = ubsi.buildBsiAttributeFromVector(B, COMPRESSION_THRESHOLD);
            bsiA->setFirstSliceFlag(true);
            bsiA->setLastSliceFlag(true);
            bsiB->setFirstSliceFlag(true);
            bsiB->setLastSliceFlag(true);

            auto t2 = Clock::now();
            long bsi_dot = bsiA->dot(bsiB);
            auto t3 = Clock::now();
            total_bsi_us += std::chrono::duration_cast<us>(t3 - t2).count();

            delete bsiA;
            delete bsiB;
        }

        double avg_vec = total_vec_us / double(RUNS);
        double avg_bsi = total_bsi_us / double(RUNS);

        out << skew << ","
            << size_val << ","
            << std::fixed << std::setprecision(2)
            << avg_bsi << ","
            << avg_vec << "\n";
    }

    out.close();
    std::cout << "Wrote results_100_1M_with_size.csv\n";
    return 0;
}

