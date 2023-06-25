#include "testBsiSigned.h"
#include "BsiAttribute.hpp"
#include "BsiSigned.hpp"
#include "BsiUnsigned.hpp"

using namespace std;

int main() {
    BsiSigned<uint64_t> build;
    BsiAttribute<uint64_t>* bsi;
    int len = 1000;
    int range = 10000;
    int k = 2;

    //--- preset array ---
    vector<long> array{-1,1};
    //--- randomize array ---
    /*vector<long> array;
    srand(time(0));
    for (int i=0; i<len; i++) {
        array.push_back(std::rand()%range-range/2);
    }*/

    //--- buildBSI ---
    sort(array.begin(),array.end());
    bsi = build.buildBsiAttributeFromVectorSigned(array,0.5);
    for (int i=0; i<array.size(); i++) {
        cout << bsi->getValue(i) << ", ";
    }
    cout << endl;

    //--- topKMax ---
    HybridBitmap<uint64_t> topkmax = bsi->topKMax(k);
    vector<long> topkmax_vector;
    cout << "topkmax number of ones: " << topkmax.numberOfOnes() << "\n";
    for (int i=0; i<topkmax.sizeInBits(); i++) {
        if (topkmax.get(i)) {
            topkmax_vector.push_back(bsi->getValue(i));
            cout << bsi->getValue(i) << " ";
        }
    }
    cout << "array length: " << topkmax_vector.size() << "\n";
    sort(topkmax_vector.begin(),topkmax_vector.end(),greater<long>());

    //--- verify accuracy ---
    int j = 0;
    bool correct = true;
    while (j<topkmax_vector.size()) {
        if (topkmax_vector[j] != array[array.size()-j-1]) {
            cout << "\n" << "incorrect" << "\n";
            correct = false;
            break;
        }
        j++;
    }
    if (correct) {
        cout << "\n" << "correct" << "\n";
    }

    //--- topKMin ---
    HybridBitmap<uint64_t> topkmin = bsi->topKMin(k);
    vector<long> topkmin_vector;
    cout << "topkmin number of ones: " << topkmin.numberOfOnes() << "\n";
    for (int i=0; i<topkmin.sizeInBits(); i++) {
        if (topkmin.get(i)) {
            topkmin_vector.push_back(bsi->getValue(i));
            cout << bsi->getValue(i) << " ";
        }
    }
    cout << "array length: " << topkmin_vector.size() << "\n";
    sort(topkmin_vector.begin(),topkmin_vector.end());

    //--- verify accuracy ---
    int i = 0;
    correct = true;
    while (i<topkmin_vector.size()) {
        if (topkmin_vector[i] != array[i]) {
            cout << "\n" << "incorrect" << "\n";
            correct = false;
            break;
        }
        i++;
    }
    if (correct) {
        cout << "\n" << "correct" << "\n";
    }
    array.clear();
    return 0;
}
