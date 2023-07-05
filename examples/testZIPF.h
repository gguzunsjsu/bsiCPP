#ifndef BSIHYBRID_TESTZIPF_H
#define BSIHYBRID_TESTZIPF_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

std::pair<std::vector<long>, std::vector<long>>  readFile() {
    std::ifstream inputFile("/Users/akankshajoshi/Documents/RA/new/bsiCPP/examples/generated_data/rows10k_skew_card16/rows10k_skew1_card16.txt");
    std::vector<long> array1;
    std::vector<long> array2;

    std::string line;
    while (std::getline(inputFile, line)) {
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, ',')) {
            // Convert the token to an integer and add it to the respective array
            if (!token.empty()) {
                int num = std::stoi(token);
                array1.push_back(num);

                // Read the next token for the second array
                if (std::getline(ss, token, ',')) {
                    int num2 = std::stoi(token);
                    array2.push_back(num2);
                }
            }
        }
    }

//    std::cout << "Array 1: ";
//    for (const auto& num : array1) {
//        std::cout << num << " ";
//    }
//    std::cout << std::endl;
//
//    std::cout << "Array 2: ";
//    for (const auto& num : array2) {
//        std::cout << num << " ";
//    }
//    std::cout << std::endl;

    return std::make_pair(array1, array2);
}

#endif
