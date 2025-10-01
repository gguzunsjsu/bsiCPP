#include <iostream>
#include <vector>
#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiVector.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"
#include <vector>
#include <cmath>
#include <random>
#include <string>

namespace {
struct Settings {
    bool sign = true;
    bool useRandom = false;
    std::size_t length = 2;
    double range = 1.0;
    int decimalPoints = 5;
    double compressThreshold = 0.4;
    unsigned int seed = std::random_device{}();
};

[[noreturn]] void printUsageAndExit(const char* binary) {
    std::cerr << "Usage: " << binary
              << " [--sign true|false] [--random] [--len <N>] [--range <R>]"
              << " [--seed <S>] [--decimals <D>]" << std::endl;
    std::exit(EXIT_FAILURE);
}

Settings parseArguments(int argc, char** argv) {
    Settings settings;
    auto requireValue = [&](int& index, const char* flag) -> std::string {
        if (index + 1 >= argc) {
            std::cerr << "Missing value for " << flag << std::endl;
            printUsageAndExit(argv[0]);
        }
        ++index;
        return argv[index];
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--sign") {
            std::string value = requireValue(i, "--sign");
            if (value == "true" || value == "1") {
                settings.sign = true;
            } else if (value == "false" || value == "0") {
                settings.sign = false;
            } else {
                std::cerr << "Invalid value for --sign: " << value << std::endl;
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--random") {
            settings.useRandom = true;
        } else if (arg == "--len") {
            std::string value = requireValue(i, "--len");
            try {
                settings.length = static_cast<std::size_t>(std::stoul(value));
            } catch (const std::exception&) {
                std::cerr << "Invalid length: " << value << std::endl;
                printUsageAndExit(argv[0]);
            }
            settings.useRandom = true;
        } else if (arg == "--range") {
            std::string value = requireValue(i, "--range");
            try {
                settings.range = std::stod(value);
            } catch (const std::exception&) {
                std::cerr << "Invalid range: " << value << std::endl;
                printUsageAndExit(argv[0]);
            }
            settings.useRandom = true;
        } else if (arg == "--seed") {
            std::string value = requireValue(i, "--seed");
            try {
                settings.seed = static_cast<unsigned int>(std::stoul(value));
            } catch (const std::exception&) {
                std::cerr << "Invalid seed: " << value << std::endl;
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--decimals") {
            std::string value = requireValue(i, "--decimals");
            try {
                settings.decimalPoints = std::stoi(value);
            } catch (const std::exception&) {
                std::cerr << "Invalid decimals: " << value << std::endl;
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--help") {
            printUsageAndExit(argv[0]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsageAndExit(argv[0]);
        }
    }

    if (settings.length == 0 && settings.useRandom) {
        std::cerr << "Vector length must be greater than zero." << std::endl;
        printUsageAndExit(argv[0]);
    }

    if (settings.range <= 0.0 && settings.useRandom) {
        std::cerr << "Range must be positive." << std::endl;
        printUsageAndExit(argv[0]);
    }

    if (settings.decimalPoints < 0) {
        std::cerr << "Decimal points must be non-negative." << std::endl;
        printUsageAndExit(argv[0]);
    }

    return settings;
}

std::vector<double> generateRandomVector(std::size_t length, double minValue,
                                         double maxValue, std::mt19937& rng) {
    std::uniform_real_distribution<double> distribution(minValue, maxValue);
    std::vector<double> values(length);
    for (double& value : values) {
        value = distribution(rng);
    }
    return values;
}

}

int main(int argc, char** argv){


    Settings settings = parseArguments(argc, argv);
    bool sign = settings.sign;
    std::vector<double> vec1;
    std::vector<double> vec2;
    if(settings.useRandom){
        std::mt19937 rng(settings.seed);
        if(sign){
            vec1 = generateRandomVector(settings.length, -settings.range, settings.range, rng);
            vec2 = generateRandomVector(settings.length, -settings.range, settings.range, rng);
        } else {
            vec1 = generateRandomVector(settings.length, 0.0, settings.range, rng);
            vec2 = generateRandomVector(settings.length, 0.0, settings.range, rng);
        }
        std::cout << "Generated random vectors with length=" << settings.length
                  << ", range=" << settings.range
                  << ", seed=" << settings.seed << std::endl;
    } else {
        if(sign){
            vec1 = {0.02, -0.04, -0.08, 0.16};
            vec2 = {-0.505343345, -0.12364533, 0.2162351, -0.512};
            // vec1 = {2, -17, -7, 34, -81, -99, 23, 56, -45, 67, 89, -90, 123, -145, 167, -189};
            // vec2 = {-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16};
        } else {
            vec1 = {-2.02, 4.04, -8.08, 16.16};
            vec2 = {-1.2, -2.5, 3.6, 4.1};
        }
    }

    int decimalPoints = settings.decimalPoints;
    double dot_res=0;
    for(std::size_t i=0; i<vec1.size(); i++){
        dot_res += vec1[i]*vec2[i];
    }
    //Printing vectors
    // std::cout << "Vector 1: ";
    // for(auto v: vec1){ std::cout << v << " "; }
    // std::cout << std::endl;
    // std::cout << "Vector 2: ";
    // for(auto v: vec2){ std::cout << v << " "; }
    // std::cout << std::endl;

    //doing the same with bsi but by using buildBsiVector with decimal points
    BsiSigned<uint64_t> bsi;
    BsiVector<uint64_t>* bsi_1;
    BsiVector<uint64_t>* bsi_2;
    bsi_1 = bsi.buildBsiVector(vec1, decimalPoints, settings.compressThreshold);
    bsi_1->setPartitionID(0); bsi_1->setFirstSliceFlag(true); bsi_1->setLastSliceFlag(true);
    bsi_2 = bsi.buildBsiVector(vec2, decimalPoints, settings.compressThreshold);
    bsi_2->setPartitionID(0); bsi_2->setFirstSliceFlag(true); bsi_2->setLastSliceFlag(true);

    //going through the slices
    int numberOfSlices = bsi_1->getNumberOfSlices();
    std::cout << "Number of slices in BSI1: " << numberOfSlices << std::endl;
    // for(int i=0; i<numberOfSlices; i++){
    //     HybridBitmap<uint64_t> slice = bsi_1->getSlice(i);
    //     std::cout << "Slice " << i << " : ";
    //     for(auto word: slice.buffer){ std::cout << std::bitset<64>(word) << " "; }
    //     std::cout << std::endl;

    //     auto sliceValue = bsi_1->getValue(i);
    //     std::cout << "Value at position " << i << " : " << sliceValue << std::endl;
    // }

    auto start = std::chrono::high_resolution_clock::now();
    double bsi_dot_res =
    static_cast<double>(bsi_1->dot(bsi_2)) / std::pow(10.0, 2 * decimalPoints);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "BSI dot product computation took " << duration.count() << " ms" << std::endl;
    std::cout << "BSI Dot product after descaling: " << bsi_dot_res << std::endl;

    //printing results
    std::cout << "Expected dot product: " << dot_res << std::endl;
    std::cout << "BSI dot product after descaling: " << bsi_dot_res << std::endl;

    //Error percentage
    double error_percentage = 0.0;
    if (std::abs(dot_res) > 1e-9) {
        error_percentage = std::abs((dot_res - bsi_dot_res) / dot_res) * 100.0;
        std::cout << "Error percentage: " << error_percentage << "%" << std::endl;
    } else {
        double absolute_error = std::abs(dot_res - bsi_dot_res);
        std::cout << "Absolute error (expected zero dot product): " << absolute_error << std::endl;
    }

    //comparing results
    if (abs(dot_res - bsi_dot_res) < 0.01){
        std::cout << "Success" << std::endl;
    } else {
        std::cout << "Failure" << std::endl;
    }
    //final print  


    std::cout << "Vector dot product: " << dot_res << std::endl;
}
