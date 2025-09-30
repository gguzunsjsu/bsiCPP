#include <iostream>
#include "../bsi/BsiVector.hpp"
#include <cmath>    
#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
#include <vector>


int main(){
    std::vector<double> vec1 = {-2, -4, -8, -16};
    std::vector<double> vec2 = {-1, -2, -3, -4};
    // std::vector<double> vec2 = {-0.505343345, -0.12364533, 0.2162351, -0.512};
    long long vector_dot=0;

    for(int i=0; i<vec1.size(); i++){
        vector_dot += vec1[i]*vec2[i];
    }
    std::cout << "Vector dot product: " << vector_dot << std::endl;

    BsiSigned<uint64_t> bsi;
    BsiVector<uint64_t>* bsi_1;
    BsiVector<uint64_t>* bsi_2;
    bsi_1 = bsi.buildBsiVector(vec1, 0, 0.4f);
    bsi_2 = bsi.buildBsiVector(vec2, 0, 0.4f);
    bsi_1->setPartitionID(0); bsi_1->setFirstSliceFlag(true); bsi_1->setLastSliceFlag(true);
    bsi_2->setPartitionID(0); bsi_2->setFirstSliceFlag(true); bsi_2->setLastSliceFlag(true);

    long long dot_res = bsi_1->dot(bsi_2);
    std::cout << "BSI Dot product: " << dot_res << std::endl;

    int numberOfSlices = bsi_1->getNumberOfSlices();
    std::cout << "Number of slices in BSI1: " << numberOfSlices
                << std::endl;

    for(int i=0; i<numberOfSlices; i++){
        auto sliceValue = bsi_1->getValue(i);
        std::cout << "Value at position " << i << " : " << sliceValue << std::endl;
    }
}
