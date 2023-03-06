
#include "testBSIAttributeBuilding.h"

#include <iostream>
using namespace std;

#include "BsiUnsigned.hpp"
#include "BsiSigned.hpp"
#include "BsiAttribute.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"
#include "../bsi/hybridBitmap/UnitTestsOfHybridBitmap.hpp"

int main() {
    vector<long> array1;
    int numberOfElementsInTheArray;
    cout << "Enter the number of elements in the array: ";
    cin >> numberOfElementsInTheArray;
    for (int i = 0; i < numberOfElementsInTheArray; i++) {
        array1.push_back(i);
    }
    double compressionThreshold;
    cout << "Enter the compression threshold: ";
    cin >> compressionThreshold;
    cout << "The number of elements in array1: " << array1.size() << "\n";
    cout << "The BSI representation is correct ?  " << (validateBuild(array1, compressionThreshold) == true ? "Yes" : "No");
    cout << "\nThank you";
    return 0;
}