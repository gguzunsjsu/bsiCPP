#include <vector>
#include <chrono>


#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiVector.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"


#pragma once
bool validateBuild(std::vector<long>, double);
bool validateMultiplicationByAConstant(std::vector<long>, BsiVector<uint64_t>*, int);
bool validateBSIWithArray(std::vector<long>, BsiVector<uint64_t>*);