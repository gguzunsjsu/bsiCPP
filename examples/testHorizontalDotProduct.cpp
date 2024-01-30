//
// Created by meera on 1/29/24.
//


#include <iostream>
#include <chrono>

#include "BsiUnsigned.hpp"
#include "BsiSigned.hpp"
#include "BsiAttribute.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"
#include "../bsi/hybridBitmap/UnitTestsOfHybridBitmap.hpp"

#include "testBSIAttributeBuilding.h"
#include "testBSI.hpp"

int main(){
    std::cout<<"Testing Horizontal Dot Product\n";
    //Build a bsi attribute
    testBSI<uint64_t>* testDotProductHorizontal = new testBSI<uint64_t>(true);
    testDotProductHorizontal->buildBSIAttribute();
    //Check if all the slices are verbatim
    for(int i =0;i<testDotProductHorizontal->bsi_attribute->bsi.size();i++){
        if(testDotProductHorizontal->bsi_attribute->bsi[i].verbatim== false)
            return 0;
    }
    //The BSI Vector has only verbatim slices, proceed
    std::cout<<"\nProceding with the horizontal verbatim dot product";
    std:: cout<<"\n"<<testDotProductHorizontal->horizontalDotProductTesting();
    return 0;
}
