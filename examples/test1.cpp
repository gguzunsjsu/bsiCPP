//
// Created by gheor on 5/14/2022.
//


#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>

#include "../bsi/BsiUnsigned.hpp"

#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiAttribute.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"
#include "../bsi/hybridBitmap/UnitTestsOfHybridBitmap.hpp"




int main()
{
    std::cout << "Hello" << std::endl;
    /*
    *  std::cout << bsiu.getValue(0) << std::endl;
    BsiSigned<ulong> bsis;

    BsiAttribute<long> *res = bsiu.SUM(&bsis);

    std::cout << res->getNumberOfSlices() << std::endl;
    */
    BsiUnsigned<uint64_t> bsiu;
   
}
