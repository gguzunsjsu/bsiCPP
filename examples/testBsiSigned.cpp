//
// Created by Cindy Zhang on 6/3/23.
//

#include "testBsiSigned.h"
#include "BsiAttribute.hpp"
#include "BsiSigned.hpp"
#include "BsiUnsigned.hpp"

using namespace std;

int main() {
    BsiSigned<uint64_t> build;
    BsiAttribute<uint64_t>* bsi;
    vector<long> array{1,1};

    bsi = build.buildBsiAttributeFromVector(array,0.2);
    //bsi->TwosToSignMagnitude();
    for (int i=0; i<array.size(); i++) {
        cout << bsi->getValue(i) << " ";
    }
    cout << endl;
    cout << bsi->size << "\n";
    HybridBitmap<uint64_t> topkmax = bsi->topKMax(2);
    cout << topkmax.numberOfOnes();

    return 0;
}
