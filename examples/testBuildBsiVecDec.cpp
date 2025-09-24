#include <iostream>
#include <vector>
#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiVector.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"
#include <vector>
#include <cmath>

int main(){
    std::vector<double> vec1 = {0.02, -0.04, -0.08, 0.16};
    std::vector<double> vec2 = {-0.505343345, -0.12364533, 0.2162351, -0.512};
    int decimalPoints = 2;
    double dot_res=0;
    for(auto i=0; i<vec1.size(); i++){
        dot_res += vec1[i]*vec2[i];
    }
    //Printing vectors
    std::cout << "Vector 1: ";
    for(auto v: vec1){ std::cout << v << " "; }
    std::cout << std::endl;
    std::cout << "Vector 2: ";
    for(auto v: vec2){ std::cout << v << " "; }
    std::cout << std::endl;

    //doing the same with bsi but by using buildBsiVector with decimal points
    BsiSigned<uint64_t> bsi;
    BsiVector<uint64_t>* bsi_1;
    BsiVector<uint64_t>* bsi_2;
    bsi_1 = bsi.buildBsiVector(vec1, decimalPoints, 0.4f);
    bsi_1->setPartitionID(0); bsi_1->setFirstSliceFlag(true); bsi_1->setLastSliceFlag(true);
    bsi_2 = bsi.buildBsiVector(vec2, decimalPoints, 0.4f);
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


    double bsi_dot_res =
    static_cast<double>(bsi_1->dot(bsi_2)) / std::pow(10.0, 2 * decimalPoints);
    std::cout << "BSI Dot product after descaling: " << bsi_dot_res << std::endl;

    //printing results
    std::cout << "Expected dot product: " << dot_res << std::endl;
    std::cout << "BSI dot product after descaling: " << bsi_dot_res << std::endl;

    //Error percentage
    double error_percentage = std::abs((dot_res - bsi_dot_res) / dot_res) * 100.0;
    std::cout << "Error percentage: " << error_percentage << "%" << std::endl;

    //comparing results
    if (abs(dot_res - bsi_dot_res) < 0.01){
        std::cout << "Success" << std::endl;
    } else {
        std::cout << "Failure" << std::endl;
    }
    //final print  


    std::cout << "Vector dot product: " << dot_res << std::endl;
}