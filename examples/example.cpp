
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>

#include "BsiUnsigned.hpp"
#include "BsiSigned.hpp"
#include "BsiAttribute.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"
#include "../bsi/hybridBitmap/UnitTestsOfHybridBitmap.hpp"


int main(){
    
    char choice;
    do {
        vector<long> array1;
        vector<long> array2;
        int range1 = 10000;
        int range2 = 10000;
        int vectorLen = 1000;
        int numberOfElementsInTheArray;
        cout << "Enter the number of elements in the array: ";
        cin >> numberOfElementsInTheArray;
        for (int i = 0; i < numberOfElementsInTheArray; i++) {
            array1.push_back(i+1);
            array2.push_back(1);
        }

        //Fill in random numbers in the array
        /*
        * for (int i = 0; i < vectorLen; i++) {
            array1.push_back(std::rand() % range1);
            array2.push_back(std::rand() % range2);
        }
        */

        cout << "The number of elements in array1: " << array1.size() << "\n";
        //cout << "The number of elements in arraay2: " << array2.size() << "\n";
        //Build BSIAttribute from the vector
        BsiUnsigned<uint64_t> ubsi;
        BsiAttribute<uint64_t>* bsi_1;
        BsiAttribute<uint64_t>* bsi_2;
        bsi_1 = ubsi.buildBsiAttributeFromVector(array1, 0);
        bsi_1->setPartitionID(0);
        bsi_1->setFirstSliceFlag(true);
        bsi_1->setLastSliceFlag(true);
        /*
        * bsi_2 = ubsi.buildBsiAttributeFromVector(array2, 0);
        bsi_2->setPartitionID(0);
        bsi_2->setFirstSliceFlag(true);
        bsi_2->setLastSliceFlag(true);
        */
        

        //Print some attributes of the BSI thus built
        cout << "NUmber of slices in the first BSI attribute: " << bsi_1->getNumberOfSlices() << "\n";
        cout << "NUmber of rows in the first BSI attribute: " << bsi_1->getNumberOfSlices() << "\n";
        //cout << "NUmber of slices in the second BSI attribute: " << bsi_2->getNumberOfSlices() << "\n";
        //cout << "NUmber of rows in the second BSI attribute: " << bsi_2->getNumberOfSlices() << "\n";

        //Checking sum of elements
        long array1Sum = 0;
        for (long element : array1) {
            array1Sum += element;
        }
        cout << "Sum of elements in array1: " << array1Sum << "\n";
        cout << "Sum of elements in the First BSI Attribute: " << bsi_1->sumOfBsi() << "\n";
        cout << "Print the elements if the First BSI Attribute" << "\n";
        for (int i = 0; i < array1.size(); i++) {
            cout << "Element "<<i+1<<": "<<bsi_1->getValue(i) << "\n";
        }
        cout << "Do you want to check again? ";
        cin >> choice;
    } while (choice == 'y');
    
    

    //Checking multiplication by a constant
    /*
    int number = 2;
    vector<long> doubleArray1;
    for (long element : array1) {
        doubleArray1.push_back(element * number);
    }
   
    long doubleArraySum = 0;
    for (long element : doubleArray1) {
        //cout << element << ", ";
        doubleArraySum += element;
    }
    cout << "The multiplied doubleArray sum: "<<doubleArraySum<<"\n";
    BsiAttribute<uint64_t>* result = bsi_1->multiplyByConstant(number);
    cout << "The multiplied BSI Attribute sum" << result->sumOfBsi()<<"\n";
    */
    

    cout << "Thank you !";



    return 0;
}
