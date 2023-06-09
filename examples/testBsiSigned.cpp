#include "testBsiSigned.h"
#include "BsiAttribute.hpp"
#include "BsiSigned.hpp"
#include "BsiUnsigned.hpp"

using namespace std;

int main() {
    BsiUnsigned<uint64_t> build;
    BsiAttribute<uint64_t>* bsi;
    vector<long> array{4};

    bsi = build.buildBsiAttributeFromVectorSigned(array,0.2);
    //bsi->TwosToSignMagnitude();
    cout << "number of slices: " << bsi->getNumberOfSlices() << "\n";
    for (int i=0; i<array.size(); i++) {
        cout << bsi->getValue(i) << " ";
    }
    cout << endl;
    cout << "bsi size in bits: " << bsi->existenceBitmap.sizeInBits() << "\n";
    HybridBitmap<uint64_t> topkmax = bsi->topKMax(2);
    cout << "topk size in bits: " << topkmax.sizeInBits() << "\n";
    cout << topkmax.numberOfOnes() << "\n";
    for (int i=0; i<topkmax.sizeInBits(); i++) {
        cout << topkmax.get(i) << " ";
    }
    return 0;
}
