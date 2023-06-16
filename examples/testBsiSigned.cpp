#include "testBsiSigned.h"
#include "BsiAttribute.hpp"
#include "BsiSigned.hpp"
#include "BsiUnsigned.hpp"

using namespace std;

int main() {
    BsiUnsigned<uint64_t> build;
    BsiAttribute<uint64_t>* bsi;
    vector<long> array{4,4,4};

    //--- buildBSI ---
    bsi = build.buildBsiAttributeFromVectorSigned(array,0.5);
    //bsi->TwosToSignMagnitude();
    //cout << "number of slices: " << bsi->getNumberOfSlices() << "\n";
    for (int i=0; i<array.size(); i++) {
        cout << bsi->getValue(i) << " ";
    }
    cout << endl;
    //cout << "bsi buffer size: " << bsi->bsi.size() << "\n";
    //cout << "bsi existence bitmap size in bits: " << bsi->existenceBitmap.sizeInBits() << "\n";

    //--- topKMax ---
    HybridBitmap<uint64_t> topkmax = bsi->topKMax(2);
    //cout << "topk size in bits: " << topkmax.sizeInBits() << "\n";
    cout << "topkmax number of ones: " << topkmax.numberOfOnes() << "\n";
    for (int i=0; i<topkmax.sizeInBits(); i++) {
        cout << topkmax.get(i) << " ";
    }
    cout << endl;

    //--- topKMin ---
    HybridBitmap<uint64_t> topkmin = bsi->topKMin(2);
    //cout << "topk size in bits: " << topkmax.sizeInBits() << "\n";
    cout << "topkmin number of ones: " << topkmin.numberOfOnes() << "\n";
    for (int i=0; i<topkmin.sizeInBits(); i++) {
        cout << topkmin.get(i) << " ";
    }
    return 0;
}
