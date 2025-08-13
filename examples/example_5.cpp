//
// Created by parth on 8/1/25.
//

#include <iostream>
#include <cstdint>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>

#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiVector.hpp"

// typedef __int128 int128_t;
// typedef unsigned __int128 uint128_t;

class ZipfDistribution {
public:
    ZipfDistribution(int N, double alpha) : N_(N), alpha_(alpha) {
        calculate_probabilities();
    }
    int operator()(std::mt19937& rng) {
        std::uniform_real_distribution<> dist(0.0, 1.0);
        double U = dist(rng);
        auto it = std::lower_bound(cumulative_probabilities_.begin(), cumulative_probabilities_.end(), U);
        return std::distance(cumulative_probabilities_.begin(), it) + 1;
    }
private:
    int N_;
    double alpha_;
    std::vector<double> probabilities_;
    std::vector<double> cumulative_probabilities_;
    void calculate_probabilities() {
        probabilities_.resize(N_ + 1); cumulative_probabilities_.resize(N_ + 1);
        double sum_inv_k_alpha = 0.0;
        for (int k = 1; k <= N_; ++k) sum_inv_k_alpha += 1.0 / std::pow((double)k, alpha_);
        double C = 1.0 / sum_inv_k_alpha;
        cumulative_probabilities_[0] = 0.0;
        for (int k = 1; k <= N_; ++k) {
            probabilities_[k] = C / std::pow((double)k, alpha_);
            cumulative_probabilities_[k] = cumulative_probabilities_[k - 1] + probabilities_[k];
        }
    }
};

template<typename T>
void SUM_test(BsiVector<T>* bsi_one, BsiVector<T>* bsi_two,
                       const std::vector<long>& normal_sum, int vector_length, const std::string& label = "") {
    std::cout << "\n[BSI " << label << " sum test]\n";
    auto t_sum = std::chrono::high_resolution_clock::now();
    BsiVector<T>* bsi_sum = bsi_one->SUM(bsi_two);
    auto t_sum2 = std::chrono::high_resolution_clock::now();
    std::cout << "BSI sum time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_sum2 - t_sum).count() << "us\n";
    for (int i = 0; i < vector_length; ++i)
        if (bsi_sum->getValue(i) != normal_sum[i])
            std::cout << "Mismatch sum [" << i << "]: BSI=" << bsi_sum->getValue(i) << " vs Normal=" << normal_sum[i] << "\n";
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    int range = 10000;
    int vector_length = 100; // keep small for verbose checks (set higher for performance tests)
    double alpha = 4.0;

    std::vector<long> one(vector_length), two(vector_length), one_signed(vector_length), two_signed(vector_length);
    std::vector<double> one_dec(vector_length), two_dec(vector_length);
    int constant = 22;

    // Generate Zipf-distributed integers and decimals
    ZipfDistribution zipf_dist(range, alpha);
    for (int i = 0; i < vector_length; ++i) {
        one[i] = zipf_dist(gen) - 2;
        two[i] = zipf_dist(gen) - 2;

        one_signed[i] = zipf_dist(gen) - (range / 2);
        two_signed[i] = zipf_dist(gen) - (range / 2);

        one_dec[i] = ((zipf_dist(gen) - 2) % 500) / 100.0; // Range: up to ~5.0
        two_dec[i] = ((zipf_dist(gen) - 2) % 500) / 100.0;
    }

    // Reference vectors for operation checks
    std::vector<double> normal_div(vector_length), normal_div_d(vector_length);
    std::vector<long> normal_sum(vector_length), normal_mul(vector_length), normal_mulc(vector_length), normal_neg(vector_length),
                        signed_sum(vector_length), signed_mul(vector_length), signed_neg(vector_length),
                        mixed_sum_signed_plus_unsigned(vector_length), mixed_sum_unsigned_plus_signed(vector_length),
                        mixed_mul_signed_unsigned(vector_length), mixed_mul_unsigned_signed(vector_length);
    std::vector<double> normal_sum_d(vector_length), normal_mul_d(vector_length), normal_neg_d(vector_length);

    // Plain arithmetic results
    for (int i = 0; i < vector_length; ++i) {
        normal_sum[i] = one[i] + two[i];
        normal_mul[i] = one[i] * two[i];
        normal_mulc[i] = one[i] * constant;
        normal_neg[i] = -one[i];
        normal_div[i] = (two[i] != 0) ? (double)one[i] / two[i] : 0;
        normal_sum_d[i] = one_dec[i] + two[i];
        normal_mul_d[i] = one_dec[i] * two[i];
        normal_neg_d[i] = -one_dec[i];
        normal_div_d[i] = (two[i] != 0) ? one_dec[i] / two[i] : 0;

        signed_sum[i] = one_signed[i] + two_signed[i];
        signed_mul[i] = one_signed[i] * two_signed[i];
        signed_neg[i] = -one_signed[i];
        mixed_sum_signed_plus_unsigned[i] = one_signed[i] + two[i];
        mixed_sum_unsigned_plus_signed[i] = one[i] * two_signed[i];
        mixed_mul_signed_unsigned[i] = one_signed[i] * two[i];
        mixed_mul_unsigned_signed[i] = one[i] * two_signed[i];
    }

    // === BSI part ===
    BsiUnsigned<uint64_t> bsi;
    BsiSigned<uint64_t> s_bsi;

    auto t_bsi_build = std::chrono::high_resolution_clock::now();
    BsiVector<uint64_t>* bsi_one = bsi.buildBsiVector(one, 0.2);
    bsi_one->setFirstSliceFlag(true);
    bsi_one->setLastSliceFlag(true);
    bsi_one->setPartitionID(0);

    BsiVector<uint64_t>* bsi_two = bsi.buildBsiVector(two, 0.2);
    bsi_two->setFirstSliceFlag(true);
    bsi_two->setLastSliceFlag(true);
    bsi_two->setPartitionID(0);

    BsiVector<uint64_t>* bsi_one_dec = bsi.buildBsiVector(one_dec, 2, 0.2);
    bsi_one_dec->setFirstSliceFlag(true);
    bsi_one_dec->setLastSliceFlag(true);
    bsi_one_dec->setPartitionID(0);

    BsiVector<uint64_t>* bsi_one_signed = s_bsi.buildBsiVector(one_signed, 0.2);
    bsi_one_signed->setFirstSliceFlag(true);
    bsi_one_signed->setLastSliceFlag(true);
    bsi_one_signed->setPartitionID(0);

    BsiVector<uint64_t>* bsi_two_signed = s_bsi.buildBsiVector(two_signed, 0.2);
    bsi_two_signed->setFirstSliceFlag(true);
    bsi_two_signed->setLastSliceFlag(true);
    bsi_two_signed->setPartitionID(0);

    auto t_bsi_build2 = std::chrono::high_resolution_clock::now();
    std::cout << "Time to build BSI vectors: " << std::chrono::duration_cast<std::chrono::microseconds>(t_bsi_build2 - t_bsi_build).count() << "us\n";



    // ==== Multiplication ====
    // std::cout << "\n[BSI int multiply test]\n";
    // auto t_mul = std::chrono::high_resolution_clock::now();
    // BsiVector<uint64_t>* bsi_mul = bsi_one->multiply_bsi(bsi_two);
    // auto t_mul2 = std::chrono::high_resolution_clock::now();
    // std::cout << "BSI multiplication time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_mul2 - t_mul).count() << "us\n";
    // for (int i = 0; i < vector_length; ++i) {
    //     long val = bsi_mul->getValue(i);
    //     if (val != normal_mul[i])
    //         std::cout << "Mismatch mul [" << i << "]: BSI=" << val << " vs Normal=" << normal_mul[i] << "\n";
    // }
    //
    // std::cout << "\n[BSI decimal multiply test]\n";
    // auto t_mul_d = std::chrono::high_resolution_clock::now();
    // BsiVector<uint64_t>* bsi_mul_d = bsi_one_dec->multiply_bsi(bsi_two);
    // auto t_mul2_d = std::chrono::high_resolution_clock::now();
    // std::cout << "BSI decimal multiplication time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_mul2_d - t_mul_d).count() << "us\n";
    // for (int i = 0; i < vector_length; ++i) {
    //     double val = bsi_mul_d->getValue_with_decimal(i);
    //     if (val != normal_mul_d[i])
    //         std::cout << std::setprecision(10) << "Mismatch mul_d [" << i << "]: BSI=" << val << " vs Normal=" << normal_mul_d[i] << "\n";
    // }

    // ==== Sum ====
    std::cout << "\n[BSI int sum test]\n";
    auto t_sum = std::chrono::high_resolution_clock::now();
    BsiVector<uint64_t>* bsi_sum = bsi_one->SUM(bsi_two);
    auto t_sum2 = std::chrono::high_resolution_clock::now();
    std::cout << "BSI sum time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_sum2 - t_sum).count() << "us\n";
    for (int i = 0; i < vector_length; ++i)
        if (bsi_sum->getValue(i) != normal_sum[i])
            std::cout << "Mismatch sum [" << i << "]: BSI=" << bsi_sum->getValue(i) << " vs Normal=" << normal_sum[i] << "\n";
    // SUM_test(bsi_one, bsi_two, normal_sum, vector_length, "int sum: unsigned + unsigned");

    std::cout << "\n[BSI mixed sum test: signed + unsigned]\n";
    auto t_mix_sum_su_start = std::chrono::high_resolution_clock::now();
    BsiVector<uint64_t>* bsi_mixed_sum_signed_plus_unsigned = bsi_one_signed->SUM(bsi_two);
    auto t_mix_sum_su_end = std::chrono::high_resolution_clock::now();
    std::cout << "Simulated BSI mixed signed+unsigned sum time: " <<
        std::chrono::duration_cast<std::chrono::microseconds>(t_mix_sum_su_end - t_mix_sum_su_start).count() << "us\n";

    for (int i = 0; i < vector_length; ++i) {
        if (bsi_mixed_sum_signed_plus_unsigned->getValue(i) != mixed_sum_signed_plus_unsigned[i])
            std::cout << "Mismatch mixed signed+unsigned sum [" << i << "]: BSI(sim)=" << bsi_mixed_sum_signed_plus_unsigned->getValue(i) << " vs Normal=" << mixed_sum_signed_plus_unsigned[i] << "\n";
    }

    std::cout << "\n[BSI mixed sum test: unsigned + signed]\n";
    auto t_mix_sum_us_start = std::chrono::high_resolution_clock::now();
    BsiVector<uint64_t>* bsi_mixed_sum_unsigned_plus_signed = bsi_one->SUM(bsi_two_signed);
    auto t_mix_sum_us_end = std::chrono::high_resolution_clock::now();
    std::cout << "Simulated BSI mixed unsigned+signed sum time: " <<
        std::chrono::duration_cast<std::chrono::microseconds>(t_mix_sum_us_end - t_mix_sum_us_start).count() << "us\n";

    for (int i = 0; i < vector_length; ++i) {
        if (bsi_mixed_sum_unsigned_plus_signed->getValue(i) != mixed_sum_unsigned_plus_signed[i])
            std::cout << "Mismatch mixed unsigned+signed sum [" << i << "]: BSI(sim)=" << bsi_mixed_sum_unsigned_plus_signed->getValue(i)
                      << " vs Normal=" << mixed_sum_unsigned_plus_signed[i] << "\n";
    }

    std::cout << "\n[BSI signed int sum test]\n";
    auto t_sum_signed = std::chrono::high_resolution_clock::now();
    BsiVector<uint64_t>* bsi_sum_signed = bsi_one_signed->SUM(bsi_two_signed);
    auto t_sum_signed2 = std::chrono::high_resolution_clock::now();
    std::cout << "BSI signed sum time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_sum_signed2 - t_sum_signed).count() << "us\n";

    for (int i = 0; i < vector_length; ++i)
        if (bsi_sum_signed->getValue(i) != signed_sum[i])
            std::cout << "Mismatch signed sum [" << i << "]: BSI=" << bsi_sum_signed->getValue(i) << " vs Normal=" << signed_sum[i] << "\n";

    // std::cout << "\n[BSI decimal sum test]\n";
    // auto t_sum_d = std::chrono::high_resolution_clock::now();
    // BsiVector<uint64_t>* bsi_sum_d = bsi_one_dec->SUM(bsi_two);
    // auto t_sum2_d = std::chrono::high_resolution_clock::now();
    // std::cout << "BSI decimal sum time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_sum2_d - t_sum_d).count() << "us\n";
    // for (int i = 0; i < vector_length; ++i)
    //     if (bsi_sum_d->getValue_with_decimal(i) != normal_sum_d[i])
    //         std::cout << std::setprecision(10) << "Mismatch sum_d [" << i << "]: BSI=" << bsi_sum_d->getValue_with_decimal(i) << " vs Normal=" << normal_sum_d[i] << "\n";

    // ==== Negation ====
    std::cout << "\n[BSI int negation test]\n";
    auto t_neg = std::chrono::high_resolution_clock::now();
    BsiVector<uint64_t>* bsi_neg = bsi_one->negate();
    auto t_neg2 = std::chrono::high_resolution_clock::now();
    std::cout << "BSI negation time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_neg2 - t_neg).count() << "us\n";
    for (int i = 0; i < vector_length; ++i)
        if (bsi_neg->getValue(i) != normal_neg[i])
            std::cout << "Mismatch neg [" << i << "]: BSI=" << bsi_neg->getValue(i) << " vs Normal=" << normal_neg[i] << "\n";

    // std::cout << "\n[BSI decimal negation test]\n";
    // auto t_neg_d = std::chrono::high_resolution_clock::now();
    // BsiVector<uint64_t>* bsi_neg_d = bsi_one_dec->negate();
    // auto t_neg2_d = std::chrono::high_resolution_clock::now();
    // std::cout << "BSI decimal negation time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_neg2_d - t_neg_d).count() << "us\n";
    // for (int i = 0; i < vector_length; ++i)
    //     if (bsi_neg_d->getValue_with_decimal(i) != normal_neg_d[i])
    //         std::cout << std::setprecision(10) << "Mismatch neg_d [" << i << "]: BSI=" << bsi_neg_d->getValue_with_decimal(i) << " vs Normal=" << normal_neg_d[i] << "\n";


    return 0;
}
