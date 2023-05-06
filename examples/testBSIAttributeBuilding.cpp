#include <iostream>
#include <array>
#include <chrono>
#if __has_include(<format>)
#include <format>
#endif

#include "BsiUnsigned.hpp"
#include "BsiSigned.hpp"
#include "BsiAttribute.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"

#include "testBSIAttributeBuilding.h"
using namespace std;

bool validateBuild(vector<long> array, double compressThreshold) {
	//Build the BSI representation of the array
	BsiAttribute<uint64_t>* bsi_1;
	BsiUnsigned<uint64_t> ubsi;
	bsi_1 = ubsi.buildBsiAttributeFromVector(array, compressThreshold);
	//Check if the BSI is stored properly
	//Retrieve the bsi representation at each index and compare with the input array elements
	bool result = true;
	for (int i = 0; i < array.size(); i++) {
		if (array[i] != bsi_1->getValue(i)) {
			cout << "Element at " << i + 1 << " position does not match the BSI representation \n";
			cout << "BSI representation: " << bsi_1->getValue(i) << "\n";
			cout << "Value in the array:  " << array[i] << "\n";
			result = false;
		}					
	}
	return result;
}
bool validateBSIWithArray(vector<long> array, BsiAttribute<uint64_t>* bsi){
	bool result = true;
	for (int i = 0; i < array.size(); i++) {
		if (array[i] != bsi->getValue(i)) {
			cout << "Element at " << i + 1 << " position does not match the BSI representation \n";
			cout << "BSI representation: " << bsi->getValue(i) << "\n";
			cout << "Value in the array:  " << array[i] << "\n";
			result = false;
		}
	}
	return result;
}

bool validateMultiplicationByAConstant(std::vector<long> array, BsiAttribute<uint64_t>* bsi, int multiplier) {

	//Build the BSI representation of the array
	bool result = true;
	for (int i = 0; i < array.size(); i++) {
		if (array[i]*multiplier != bsi->getValue(i)) {
			cout << "Element at " << i + 1 << " position does not match the BSI representation \n";
			cout << "BSI representation: " << bsi->getValue(i) << "\n";
			cout << "Value in the array:  " << array[i] * multiplier << "\n";
			result = false;
		}
	}
	return result;


}

bool validateBSIWithArray(BsiAttribute<uint64_t>* bsi, std::vector<long> array ) {
	bool result = true;
	for (int i = 0; i < array.size(); i++) {
		if (array[i] != bsi->getValue(i)) {
			cout << "Element at " << i + 1 << " position does not match the BSI representation \n";
			cout << "BSI representation: " << bsi->getValue(i) << "\n";
			cout << "Value in the array:  " << array[i] << "\n";
			result = false;
		}
	}
	return result;
}