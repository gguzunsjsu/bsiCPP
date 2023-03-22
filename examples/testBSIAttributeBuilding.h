#include <vector>
#include <chrono>


#include "BsiUnsigned.hpp"
#include "BsiSigned.hpp"
#include "BsiAttribute.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"

#pragma once
bool validateBuild(std::vector<long>, double);
bool validateMultiplicationByAConstant(std::vector<long>, BsiAttribute<uint64_t>*, int);
bool validateBSIWithArray(std::vector<long>, BsiAttribute<uint64_t>*);
