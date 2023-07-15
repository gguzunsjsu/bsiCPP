
#include "testBSIAttributeBuilding.h"
#include "testBSI.hpp"
//#include "benchmarking.h"

#include <iostream>
#include <chrono>

#include "BsiUnsigned.hpp"
#include "BsiSigned.hpp"
#include "BsiAttribute.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"
#include "../bsi/hybridBitmap/UnitTestsOfHybridBitmap.hpp"


//void MultiplyVectorByScalar(vector<long>& v, int k) {
//    transform(v.begin(), v.end(), v.begin(), [k](long& c) { return c * k; });
//}

int main() {
    //Create an instance of the testBSI class for numbers in the range  
    /*
    * testBSI<uint64_t>* test = new testBSI<uint64_t>(100000);
    test->buildBSIAttribute();
    cout << "BSI Attribute building valid ? " << validateBSIWithArray(test->array, test->bsi_attribute);
    test->multiplyByConstant();
    test->sumOfBSIVectorElements();  

    //Testing vector multiplication
    testBSI<uint64_t>* testVectorMultiplication = new testBSI<uint64_t>(100000);
    testVectorMultiplication->buildBSIAttribute();
    testVectorMultiplication->vectorMultiplicationOfBSI();
    */
    
    
    

    //Testing sumOf BSI
    /*
    *  testBSI<uint64_t>* testVectorMultiplication = new testBSI<uint64_t>(100);
    testVectorMultiplication->buildBSIAttribute();
    testVectorMultiplication->sumOfBSIVectorElements();
    */


   
    testBSI<uint64_t>* testVectorMultiplication = new testBSI<uint64_t>(1000);
    testVectorMultiplication->buildBSIAttribute();
    testVectorMultiplication->dotProductForTestingNew();
    //testVectorMultiplication->vectorMultiplicationOfBSIWithUserInput();


//    benchmarking<uint64_t>* testVectorMultiplication = new benchmarking<uint64_t>(100);
//    testVectorMultiplication->buildBSIAttribute();
//    testVectorMultiplication->vectorMultiplicationOfBSIWithUserInput();

    cout << "\n\nThank you";
    return 0;

    /*
   
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
        //Working with unsigned BSI for now

        BsiSigned<uint64_t> bsi;
        
        auto start = chrono::high_resolution_clock::now();
        BsiAttribute<uint64_t>* bsi_1 = bsi.buildBsiAttributeFromVector(array1, compressionThreshold);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "\nTime to build the BSI Attribute: " << duration.count() << endl;
        bsi_1->setPartitionID(0);
        bsi_1->setFirstSliceFlag(true);
        bsi_1->setLastSliceFlag(true);
        
        BsiAttribute<uint64_t>* bsi_2 = bsi.buildBsiAttributeFromVector(array1, compressionThreshold);
        bsi_2->setPartitionID(0);
        bsi_2->setFirstSliceFlag(true);
        bsi_2->setLastSliceFlag(true);
        BsiAttribute<uint64_t>* bsi_3 = bsi_1->SUM(bsi_2);
        vector<long> sumArray;
        for (int i = 0; i < array1.size(); i++) {
            sumArray.push_back(array1[i] * 2);
        }
        cout << "Sum of two BSI instances is valid ? " << validateBSIWithArray(sumArray, bsi_3);
        
       
        
        int multiplier;
        cout << "\n\nEnter the number to multiply with ? ";
        cin >> multiplier;
        //Time calculation
        start = chrono::high_resolution_clock::now();
        BsiAttribute<uint64_t>* bsi_2 = bsi_1->multiplyByConstantNew(multiplier);
        stop = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "Time to Multiply by constant the BSI Attribute: " << duration.count() << endl;

        cout << "\n The BSI Multiplication with constant " << multiplier << " result :";
        cout << (validateMultiplicationByAConstant(array1, bsi_2, multiplier) == true ? "Yes" : "No");
        
        start = chrono::high_resolution_clock::now();
        MultiplyVectorByScalar(array1, multiplier);
        stop = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "\nTime to Multiply by constant the Vector: " << duration.count() << endl;

        array1.clear();
        numberOfElementsInTheArray = 0;
        cout << "\nDo you want to continue? (n to stop) ";
        cin >> ch;

    } while (ch != 'n' && ch != 'N');
    *
    */




  
}

