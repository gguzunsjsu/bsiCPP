#ifndef testBSI_H
#define testBSI_H
#include "BsiUnsigned.hpp"
#include "BsiSigned.hpp"
#include "BsiAttribute.hpp"
#include "testBSIAttributeBuilding.h"
#include <iostream>
#include <chrono>
#include <numeric>


using namespace std;
/*
* this test class is for testing various operations on BSI
* Each instance of the class has a range of elements it can hold
*/
template <class uword = uint64_t>
class testBSI {
public:
    int range;
    double compressionThreshold;
    vector<long> array;
    BsiSigned<uword> signed_bsi;
    BsiAttribute<uword>* bsi_attribute;
    int numberOfElementsInTheArray;
    

    //Constructors

    
    testBSI()
    {
    }
    
    testBSI(int range) {
        this->range = range;
    }

    //Member functions
    void buildBSIAttribute() {


        cout << "Enter the number of elements in the array: ";
        cin >> this->numberOfElementsInTheArray;
        int randomChoice;
        cout << "Do you want to initialize the array with \n 1. random numbers \n 2. preset numbers \n 3. user input? ";
        cin >> randomChoice;
        if (randomChoice == 2) {
            for (int i = 0; i < this->numberOfElementsInTheArray; i++) {
                this->array.push_back((i+1)% this->range);
            }
        }
        else if (randomChoice == 3) {
            long number;
            cout << "\nEnter the numbers : \n";
            for (int i = 0; i < this->numberOfElementsInTheArray; i++) {
                cin >> number;
                this->array.push_back(number % this->range);
            }
        }
        else {
            cout << "\nInitializing the array with random numbers\n";
            //Fill in random numbers in the array        
            for (int i = 0; i < this->numberOfElementsInTheArray; i++) {
                this->array.push_back(std::rand() % this->range);
            }
        }
        cout << "Enter the compression threshold: ";
        cin >> this->compressionThreshold;
        cout << "The number of elements in array: " << this->array.size() << "\n";
        auto start = chrono::high_resolution_clock::now();
        this->bsi_attribute = this->signed_bsi.buildBsiAttributeFromVector(this->array, this->compressionThreshold);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "\nTime to build the BSI Attribute: " << duration.count() << endl<<endl;
        this->bsi_attribute->setPartitionID(0);
        this->bsi_attribute->setFirstSliceFlag(true);
        this->bsi_attribute->setLastSliceFlag(true);
        

    }
    void multiplyByConstant() {
        int multiplier;
        cout << "\n\nEnter the number to multiply with ? ";
        cin >> multiplier;
        auto start = chrono::high_resolution_clock::now();
        BsiAttribute<uint64_t>* result = this->bsi_attribute->multiplyByConstantNew(multiplier);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "\nTime for Multiplication By Constant for the BSI Attribute: " << duration.count() << endl;
        cout << "Multiplication by constant correct ? " << validateMultiplicationByAConstant(this->array, result, multiplier) <<"\n";

    }
    void sumOfTwoBSIVectors() {
        //For simple testing, lets test the addition of the BSI Attribute with each other
        BsiAttribute<uint64_t>* bsi2 = this->signed_bsi.buildBsiAttributeFromVector(this->array, this->compressionThreshold);
        bsi2->setPartitionID(0);
        bsi2->setFirstSliceFlag(true);
        bsi2->setLastSliceFlag(true);

        BsiAttribute<uint64_t>* result = this->bsi_attribute->SUM(bsi2);
        vector<long> sumArray;
        for (int i = 0; i < this->array.size(); i++) {
            sumArray.push_back(this->array[i] * 2);
        }
        cout << "Sum of two BSI instances is correct ? " << validateBSIWithArray(sumArray, result)<<"\n";
    }
    void sumOfBSIVectorElements() {
        //Time to add elements in the BSI
        auto start = chrono::high_resolution_clock::now();
        auto sum = this->bsi_attribute->sumOfBsi();
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "\nTime for addition of elements for the BSI Attribute: " << duration.count() << endl;
        //Time to add elements in the array
        start = chrono::high_resolution_clock::now();
        long sum_of_elems = std::accumulate(this->array.begin(), this->array.end(), 0);
        stop = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "Time to add elements in the C++ standard vector: " << duration.count() << endl;


    }
    void vectorMultiplicationOfBSI() {
        BsiAttribute<uint64_t>* bsi2 = this->signed_bsi.buildBsiAttributeFromVector(this->array, this->compressionThreshold);
        bsi2->setPartitionID(0);
        bsi2->setFirstSliceFlag(true);
        bsi2->setLastSliceFlag(true);
        //Try vector multiplication with C++ - For now square of numbers
        vector<long> v;
        for (long i = 0; i < this->array.size(); ++i) {
            v.push_back(this->array[i] * this->array[i]);
        }
        BsiAttribute<uint64_t>* bsi3 = this->bsi_attribute->multiplication(bsi2);
        cout << "Vector multiplication  correct ? " << validateBSIWithArray(v,bsi3 ) << "\n";
        cout << "Validated successfully vector multiplication" << endl;

    }
    void vectorMultiplicationOfBSIWithUserInput() {
        vector<long> array1;
        long number;
        int randomChoice;
        cout << "Do you want to initialize the array with \n 1. random numbers \n 2. input numbers ? ";
        cin >> randomChoice;
        
        if (randomChoice == 2) {
            cout << "Enter the numbers : \n";
            for (int i = 0; i < this->numberOfElementsInTheArray; i++) {
                cin >> number;
                array1.push_back(number % this->range);
            }
        }
        else {
            cout << "\nInitializing the array with random numbers\n";
            //Fill in random numbers in the array        
            for (int i = 0; i < this->numberOfElementsInTheArray; i++) {
                array1.push_back(std::rand() % this->range);
            }
        }
        BsiAttribute<uint64_t>* bsi2 = this->signed_bsi.buildBsiAttributeFromVector(array1, this->compressionThreshold);
        bsi2->setPartitionID(0);
        bsi2->setFirstSliceFlag(true);
        bsi2->setLastSliceFlag(true);
        //Try vector multiplication with C++ - For now square of numbers
        vector<long> v;
        for (long i = 0; i < this->array.size(); ++i) {
            v.push_back(this->array[i] * array1[i]);
        }
        BsiAttribute<uint64_t>* bsi3 = this->bsi_attribute->multiplication(bsi2);

        cout << "Vector multiplication  correct ? " << validateBSIWithArray(v, bsi3) << "\n";
        cout << "Validated successfully vector multiplication" << endl;

    }

};

#endif
