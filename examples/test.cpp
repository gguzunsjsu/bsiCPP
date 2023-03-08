
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
    char ch;
    do {
        int numberOfElementsInTheArray;
        cout << "Enter the number of elements in the array: ";
        cin >> numberOfElementsInTheArray;

        int randomChoice;
        cout << "Do you want to initialize the array with \n 1. random numbers \n 2. preset numbers ? ";
        cin >> randomChoice;
        if (randomChoice == 2) {
            for (int i = 0; i < numberOfElementsInTheArray; i++) {
                array1.push_back(i);
            }
        }
        else {
            cout << "\nInitializing the array with random numbers\n";
            int range1 = 10000;
            //Fill in random numbers in the array        
            for (int i = 0; i < numberOfElementsInTheArray; i++) {
                array1.push_back(std::rand() % range1);
            }
        }
        
        double compressionThreshold;
        cout << "Enter the compression threshold: ";
        cin >> compressionThreshold;
        cout << "The number of elements in array1: " << array1.size() << "\n";
        cout << "The BSI representation is correct ?  " << (validateBuild(array1, compressionThreshold) == true ? "Yes" : "No");
        

        BsiSigned<uint64_t> bsi;
        

        BsiAttribute<uint64_t>* bsi_1 = bsi.buildBsiAttributeFromVector(array1, compressionThreshold);
        int multiplier;
        cout << "\n\nEnter the number to multiply with ? ";
        cin >> multiplier;
        BsiAttribute<uint64_t>* bsi_2 = bsi_1->multiplyByConstant(multiplier);
        cout << "\n The BSI Multiplication with constant " << multiplier << " result :";
        cout << (validateMultiplicationByAConstant(array1, bsi_2, multiplier) == true ? "Yes" : "No");
        cout << "\nDo you want to continue? (n to stop) ";
        cin >> ch;

    } while (ch != 'n' && ch != 'N');
    

    cout << "\n\nThank you";
    return 0;
}