#ifndef testBSI_H
#define testBSI_H
#include "BsiUnsigned.hpp"
#include "BsiSigned.hpp"
#include "BsiAttribute.hpp"
#include "testBSIAttributeBuilding.h"
#include <iostream>
#include <chrono>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include "zipf.h"
#include <torch/torch.h>

//----- Constants -----------------------------------------------------------

#define  FALSE          0       // Boolean false
#define  TRUE           1       // Boolean true

using namespace std;
/*
 * this test class is for testing various operations on BSI
 * Each instance of the class has a range of elements it can hold
 */
template <class uword = uint64_t>
class testBSI
{
public:
    int range;
    double compressionThreshold;
    vector<long> array;
    BsiSigned<uword> signed_bsi;
    BsiAttribute<uword> *bsi_attribute;
    int numberOfElementsInTheArray;
    int maxValue;
    torch::Tensor tensor;

    // Constructors

    testBSI()
    {
    }

    testBSI(int range)
    {
        this->range = range;
    }

    // Member functions
    void buildBSIAttribute()
    {

        cout << "Enter the number of elements in the array: ";
        cin >> this->numberOfElementsInTheArray;
        int randomChoice;
        cout << "Enter the Max value: ";
        cin >> this->maxValue;
        this->range=this->maxValue;

        cout << "Do you want to initialize the array with \n 1. random numbers \n 2. preset numbers \n 3. user input? ";
        cin >> randomChoice;
        if (randomChoice == 2)
        {
            for (int i = 0; i < this->numberOfElementsInTheArray; i++)
            {
                this->array.push_back((i + 1) % this->range);
            }
        }
        else if (randomChoice == 3)
        {
            long number;
            cout << "\nEnter the numbers : \n";
            for (int i = 0; i < this->numberOfElementsInTheArray; i++)
            {
                cin >> number;
                this->array.push_back(number % this->range);
            }
        }
        else
        {
            cout << "\nInitializing the array with random numbers\n";
            int distribution;
            cout << "Choose your distribution \n 1. Uniform \n 2. Skewed zipf distribution ";
            cin >> distribution;
            if(distribution==1){
                cout << "\nInitializing the array with random UNIFORM numbers\n";
                for (int i = 0; i < this->numberOfElementsInTheArray; i++)
                {
                    this->array.push_back(std::rand() % this->range);
                }
            }
            else{
                double s =1.0;
                cout << "\nZIPF: choose a real number between 0 and 5 for the zipf skew: ";
                cin >> s;
                cout << "\nInitializing the array with random ZIPF SKEWED numbers\n";
                std::random_device rd;
                std::mt19937 gen(rd());
                zipf_distribution<> zipf(this->range, s);

                for (int i = 0; i < this->numberOfElementsInTheArray; i++)
                {
                    this->array.push_back(zipf(gen)-1);
                }

            }

            // Fill in random numbers in the array


        }
        cout << "Enter the compression threshold: ";
        cin >> this->compressionThreshold;
        cout << "The number of elements in array: " << this->array.size() << "\n";
        auto start = chrono::high_resolution_clock::now();
//        this->bsi_attribute = this->signed_bsi.buildBsiAttributeFromVector(this->array, this->compressionThreshold);
        this->bsi_attribute = this->signed_bsi.buildBsiAttributeFromVectorSigned(this->array, this->compressionThreshold);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "\nTime to build the BSI Attribute: " << duration.count() << endl;
        this->bsi_attribute->setPartitionID(0);
        this->bsi_attribute->setFirstSliceFlag(true);
        this->bsi_attribute->setLastSliceFlag(true);
    }

    // Member functions
    void buildBSIAttributefromTensor()
    {
        cout << "Enter the number of elements in the array: ";
        cin >> this->numberOfElementsInTheArray;

        cout << "Enter the Max value: ";
        cin >> this->maxValue;
        range = maxValue;

        int randomChoice;
        cout << "Do you want to initialize the array with \n 1. random numbers \n 2. preset numbers \n 3. user input? ";
        cin >> randomChoice;

        if (randomChoice == 2) {
            std::vector<long> presetValues;
            for (int i = 0; i < numberOfElementsInTheArray; i++) {
                presetValues.push_back((i + 1) % range);
            }
            // Create a PyTorch tensor from the vector
            this->tensor = torch::from_blob(presetValues.data(), {numberOfElementsInTheArray}, torch::kLong);
        }
        else if (randomChoice == 3) {
            vector<long> userValues;
            long number;
            cout << "\nEnter the numbers : \n";
            for (int i = 0; i < numberOfElementsInTheArray; i++) {
                cin >> number;
                userValues.push_back(number % range);
            }
            // Create a PyTorch tensor from the vector
            this->tensor = torch::from_blob(userValues.data(), {numberOfElementsInTheArray}, torch::kLong);
        }
        else {
            cout << "\nInitializing the tensor with random numbers\n";
            int distribution;
            cout << "Choose your distribution \n 1. Uniform \n 2. Skewed zipf distribution ";
            cin >> distribution;
            if (distribution == 1) {
                cout << "\nInitializing the tensor with random UNIFORM numbers\n";
                vector<long> randomValues;
                for (int i = 0; i < this->numberOfElementsInTheArray; i++) {
                    randomValues.push_back(rand() % range);
                }
                // Create a PyTorch tensor from the vector
                this->tensor = torch::from_blob(randomValues.data(), {numberOfElementsInTheArray}, torch::kLong);
            }
            else {
                double s = 1.0;
                cout << "\nZIPF: choose a real number between 0 and 5 for the zipf skew: ";
                cin >> s;
                cout << "\nInitializing the tensor with random ZIPF SKEWED numbers\n";
                random_device rd;
                mt19937 gen(rd());
                vector<long> zipfValues;
                zipf_distribution<> zipf(range, s);

                for (int i = 0; i < this->numberOfElementsInTheArray; i++) {
                    zipfValues.push_back(zipf(gen) - 1);
                }
                // Create a PyTorch tensor from the vector
                tensor = torch::from_blob(zipfValues.data(), {this->numberOfElementsInTheArray}, torch::kLong);
            }
        }

        cout << "Enter the compression threshold: ";
        cin >> this->compressionThreshold;
        cout << "The number of elements in array: " << this->numberOfElementsInTheArray<< "\n";
        auto start = chrono::high_resolution_clock::now();
        this->bsi_attribute = this->signed_bsi.buildBsiAttributeFromTensor(this->tensor, this->compressionThreshold);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "\nTime to build the BSI Attribute: " << duration.count() << endl;
        this->bsi_attribute->setPartitionID(0);
        this->bsi_attribute->setFirstSliceFlag(true);
        this->bsi_attribute->setLastSliceFlag(true);
    }

    void multiplyByConstant()
    {
        int multiplier;
        cout << "\n\nEnter the number to multiply with ? ";
        cin >> multiplier;
        auto start = chrono::high_resolution_clock::now();
        BsiAttribute<uint64_t> *result = this->bsi_attribute->multiplyByConstantNew(multiplier);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "\nTime for Multiplication By Constant for the BSI Attribute: " << duration.count() << endl;
        cout << "Multiplication by constant correct ? " << validateMultiplicationByAConstant(this->array, result, multiplier) << "\n";
    }
    void sumOfTwoBSIVectors()
    {
        // For simple testing, lets test the addition of the BSI Attribute with each other
        BsiAttribute<uint64_t> *bsi2 = this->signed_bsi.buildBsiAttributeFromVector(this->array, this->compressionThreshold);
        bsi2->setPartitionID(0);
        bsi2->setFirstSliceFlag(true);
        bsi2->setLastSliceFlag(true);

        BsiAttribute<uint64_t> *result = this->bsi_attribute->SUM(bsi2);
        vector<long> sumArray;
        for (int i = 0; i < this->array.size(); i++)
        {
            sumArray.push_back(this->array[i] * 2);
        }
        cout << "Sum of two BSI instances is correct ? " << validateBSIWithArray(sumArray, result) << "\n";
    }
    void sumOfBSIVectorElements()
    {
        // Time to add elements in the BSI
        auto start = chrono::high_resolution_clock::now();
        auto sum = this->bsi_attribute->sumOfBsi();
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "\nTime for addition of elements for the BSI Attribute: " << duration.count() << endl;
        // Time to add elements in the array
        start = chrono::high_resolution_clock::now();
        long sum_of_elems = std::accumulate(this->array.begin(), this->array.end(), 0);
        stop = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "Time to add elements in the C++ standard vector: " << duration.count() << endl;
    }
    void vectorMultiplicationOfBSI()
    {
        BsiAttribute<uint64_t> *bsi2 = this->signed_bsi.buildBsiAttributeFromVector(this->array, this->compressionThreshold);
        bsi2->setPartitionID(0);
        bsi2->setFirstSliceFlag(true);
        bsi2->setLastSliceFlag(true);
        // Try vector multiplication with C++ - For now square of numbers
        vector<long> v;
        for (long i = 0; i < this->array.size(); ++i)
        {
            v.push_back(this->array[i] * this->array[i]);
        }
        BsiAttribute<uint64_t> *bsi3 = this->bsi_attribute->multiplication(bsi2);
        cout << "Vector multiplication  correct ? " << validateBSIWithArray(v, bsi3) << "\n";
        cout << "Done with multiplication" << endl;
    }
    void vectorMultiplicationOfBSIWithUserInput()
    {
        vector<long> array1;
        long number;
        int randomChoice;
        cout << "Do you want to initialize the array with \n 1. random numbers \n 2. input numbers ? ";
        cin >> randomChoice;

        if (randomChoice == 2)
        {
            cout << "Enter the numbers : \n";
            for (int i = 0; i < this->numberOfElementsInTheArray; i++)
            {
                cin >> number;
                array1.push_back(number % this->range);
            }
        }
        else
        {
            cout << "\nInitializing the array with random numbers\n";
            // Fill in random numbers in the array
            for (int i = 0; i < this->numberOfElementsInTheArray; i++)
            {
                array1.push_back(std::rand() % this->range);
            }
        }
        BsiAttribute<uint64_t> *bsi2 = this->signed_bsi.buildBsiAttributeFromVector(array1, this->compressionThreshold);
        bsi2->setPartitionID(0);
        bsi2->setFirstSliceFlag(true);
        bsi2->setLastSliceFlag(true);
        // Try vector multiplication with C++ - For now square of numbers
        vector<long> v;
        for (long i = 0; i < this->array.size(); ++i)
        {
            v.push_back(this->array[i] * array1[i]);
        }
        BsiAttribute<uint64_t> *bsi3 = this->bsi_attribute->multiplication(bsi2);

        cout << "Vector multiplication  correct ? " << validateBSIWithArray(v, bsi3) << "\n";
        cout << "Done with multiplication" << endl;
    }
    long dotProductForTestingNew() {
        vector<long> array1;
        long number;
        int randomChoice;
        cout << "We are in the method to test dot product of two vectors represented as bsis\n";
        cout << "Enter the numbers in the new vector: \n";
        cout << "Do you want to initialize the array with \n 1. random numbers \n 2. input numbers ? ";
        cin >> randomChoice;

        if (randomChoice == 2)
        {
            cout << "Enter the numbers : \n";
            for (int i = 0; i < this->numberOfElementsInTheArray; i++)
            {
                cin >> number;
                array1.push_back(number % this->range);
            }
        }
        else {
            cout << "\nInitializing the array with random numbers\n";
            int distribution;
            cout << "Choose your distribution \n 1. Uniform \n 2. Skewed zipf distribution ";
            cin >> distribution;
            if (distribution == 1) {
                cout << "\nInitializing the array with random UNIFORM numbers\n";
                for (int i = 0; i < this->numberOfElementsInTheArray; i++) {
                    array1.push_back(std::rand() % this->range);
                }
            } else {
                double s = 1.0;
                cout << "\nZIPF: choose a real number between 0 and 5 for the zipf skew: ";
                cin >> s;
                cout << "\nInitializing the array with random ZIPF SKEWED numbers\n";

                std::random_device rd;
                std::mt19937 gen(rd());
                zipf_distribution<> zipf(this->range, s);

                for (int i = 0; i < this->numberOfElementsInTheArray; i++) {
                    array1.push_back(zipf(gen) - 1);
                }
            }
        }
        cout << "\nDONE!\n";
        // Build the BSI attribute for this
//        BsiAttribute<uint64_t>* bsi2 = this->signed_bsi.buildBsiAttributeFromVector(array1, this->compressionThreshold);
        BsiAttribute<uint64_t>* bsi2 = this->signed_bsi.buildBsiAttributeFromVectorSigned(array1, this->compressionThreshold);
        bsi2->setPartitionID(0);
        bsi2->setFirstSliceFlag(true);
        bsi2->setLastSliceFlag(true);
        // Validate the build of the BSI attribute
        //cout << "BSI Attribute building is correct ? " << validateBSIWithArray(array1, bsi2) << "\n";
        cout << "Let's try to do dot product\n";

        auto start = chrono::high_resolution_clock::now();
        long resultFromVectors = vectorDotProduct(this->array, array1);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "\nTime for dot product for the vectors: " << duration.count() << endl;
        cout << "Result from vectors : " << resultFromVectors << endl;


//        start = chrono::high_resolution_clock::now();
//        long resultbsi2 =  this->bsi_attribute->dotProduct(bsi2);
//        stop = chrono::high_resolution_clock::now();
//        duration = chrono::duration_cast<chrono::microseconds>(stop - start);
//        cout << "\nTime for dot product via multiplication, horizontal split and no compression for the BSI Attribute: " << duration.count() << endl;
//        cout << "Result: " << resultbsi2 << endl;

        // Take average time of 5 runs
        long total = 0;
        long result;
        for(int i =0; i<5;i++) {
            auto start = chrono::high_resolution_clock::now();
            result = this->bsi_attribute->dot(bsi2);
            auto stop = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            total = total + duration.count();
        }
        double average = total/5;
        cout << "\nAverage Time for dot product for the BSI Attribute: " << average << endl;
        cout << "Result: " << result << endl;

        return result;
    }

    long vectorDotProduct(vector<long> vector_a, vector<long> vector_b) {
        long product = 0;
        for (int i = 0; i < this->numberOfElementsInTheArray; i++)
            product = product + vector_a[i] * vector_b[i];
        return product;
    }


    long dotProductForTesting()
    {
        vector<long> array1;
        long number;
        cout << "We are in the method to test dot product of two vectors represented as bsis";
        cout << "Enter the numbers in the new vector: \n";
        for (int i = 0; i < this->numberOfElementsInTheArray; i++)
        {
            cin >> number;
            array1.push_back(number % this->range);
        }
        // Build the BSI attribute for this
        BsiAttribute<uint64_t> *bsi2 = this->signed_bsi.buildBsiAttributeFromVector(array1, this->compressionThreshold);
        bsi2->setPartitionID(0);
        bsi2->setFirstSliceFlag(true);
        bsi2->setLastSliceFlag(true);
        // Validate the build of the BSI attribute
        cout << "BSI Attribute building is correct ? " << validateBSIWithArray(array1, bsi2) << "\n";
        // Multiple the two BSI Attributes
        /*
        * BsiAttribute<uint64_t> *bsi3 = this->bsi_attribute->multiplication(bsi2);
        // Validate multiplication
        vector<long> v;
        for (long i = 0; i < this->array.size(); ++i)
        {
            v.push_back(this->array[i] * array1[i]);
        }
        cout << "Vector multiplication  correct ? " << validateBSIWithArray(v, bsi3) << "\n";
        */
        
        cout << "Let's try to do dot product\n";
        // Initialize the necessary vectors
        // res = new BsiUnsigned<uint64_t>();
        HybridBitmap<uint64_t> hybridBitmap;
        hybridBitmap.reset();
        hybridBitmap.verbatim = true;
        /*
        for (int j = 0; j < this->bsi_attribute->size + bsi3->size; j++)
        {
            res->addSlice(hybridBitmap);
        }
        */

        int size_a = this->bsi_attribute->size;
        int size_b = bsi2->size;
        std::vector<uint64_t> a(size_a);
        std::vector<uint64_t> b(size_b);
        std::vector<uint64_t> answer(size_a + size_b);
        long dotProductSum = 0;
        int ansSize = a.size();
        // For each word in the BSI buffer
        for (int bu = 0; bu < this->bsi_attribute->bsi[0].bufferSize(); bu++)
        {
            // For each slice, get the decimal representations added into an array
            for (int j = 0; j < this->bsi_attribute->size; j++)
            {
                a[j] = this->bsi_attribute->bsi[j].getWord(bu); // fetching one word
            }
            for (int j = 0; j < bsi2->size; j++)
            {
                b[j] = bsi2->bsi[j].getWord(bu);
            }
            // Multiply the two vectors and take their running sum
            // For {1,2,3,4,5} => {21, 6, 24}}
            for (int i = 0; i < a.size(); i++)
            {
                answer[i] = a[i] & b[0];
            }
            for (int i = a.size(); i < b.size() + a.size(); i++)
            {
                answer[i] = 0;
            }
            //{21,4,16,0,0,0}
            uint64_t S, C, FS;
            int k = 1;
            // For each value in the second vector
            for (int it = 1; it < b.size(); it++)
            {
                S = answer[k] ^ a[0];
                C = answer[k] & a[0];
                FS = S & b[it];
                answer[k] = (~b[it] & answer[k]) | (b[it] & FS);
                //{21,0,16,0,0,0}
                // Second iteration of loop via b
                //{21,0,2,4,0,0}
                for (int i = 1; i < a.size(); i++)
                {
                    // What happens here
                    if ((i + k) < ansSize)
                    {
                        S = answer[i + k] ^ a[i] ^ C;
                        C = (answer[i + k] & a[i]) | (a[i] & C) | (answer[i + k] & C);
                    }
                    else
                    {
                        S = a[i] ^ C;
                        C = a[i] & C;
                        FS = S & b[it];
                        ansSize++;
                        answer[ansSize - 1] = FS;
                    }
                    FS = b[it] & S;
                    answer[i + k] = (~b[it] & answer[i + k]) | (b[it] & FS);
                    //{21,0,18,0,0,0}
                    //{21,0,18,4,0,0}
                    // Second iteration of loop via b
                    //{21,0,2,20,0,0}
                    //{21,0,2,20,24,0}
                }
                // When does this even become a usecase
                // When there is extra to be calculated?
                for (int i = a.size() + k; i < ansSize; i++)
                {
                    S = answer[i] ^ C;
                    C = answer[i] & C;
                    FS = b[it] & S;
                    answer[k] = (~b[it] & answer[k]) | (b[it] & FS);
                    ;
                }
                if (C > 0)
                {
                    ansSize++;
                    answer[ansSize - 1] = b[it] & C;
                }
                k++;
            } // End of loop via size of b
            // So we have the answer for one buffer in ans
            // Add it as a slice to our result
            /*
            for (int j = 0; j < ans.size(); j++)
            {
                res->bsi[j].addVerbatim(answer[j]);
            }
            */
            // Get the number of ones in each element of the answer vector and sum it
            for (auto n = 0; n < answer.size(); n++)
            {
                long temp = countOnes(answer[n]) * (1<<n);
                dotProductSum += temp;
            }
        }
        // Finally we have the result in res
        //{21,0,2,20,24,0}
        return dotProductSum;
    }
    int countOnes(int n)
    {
        int count = 0;
        while (n != 0)
        {
            n = n & (n - 1); // remove the rightmost 1-bit from n
            count++;
        }
        return count;
    }
};

#endif
