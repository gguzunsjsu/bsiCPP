#include <fstream>
#include "testBsiSigned.h"
#include "BsiAttribute.hpp"
#include "BsiSigned.hpp"
#include "BsiUnsigned.hpp"

using namespace std;
int compare(const void* a, const void* b);
void runQuickSort(vector<long> array,int k);
void runTopKMax(int k, BsiAttribute<uint64_t>* bsi, vector<long> array);
void runTopKMin(int k, BsiAttribute<uint64_t>* bsi, vector<long> array);
int main() {
    BsiSigned<uint64_t> build;
    BsiAttribute<uint64_t>* bsi;
    int k = 500;

    //--- read zipf generated array ---
    vector<long> array;
    string line;
    ifstream file("/Users/zhang/CLionProjects/bsiCPP/examples/rows10k_skew0_card16_neg");
    while (getline(file, line)) {
        array.push_back(stoi(line.substr(0,line.size()-1)));
    }
    file.close();

    //--- buildBSI ---
    bsi = build.buildBsiAttributeFromVectorSigned(array,0.5);

    //--- test runtimes ---
    sort(array.begin(),array.end());
    runTopKMax(k,bsi,array);
    runTopKMin(k,bsi,array);

    array.clear();
    return 0;
}
void runQuickSort(vector<long> array, int k) {
    //--- quick sort ---
    long* a = &array[0];
    int len = array.size();
    auto start = chrono::high_resolution_clock::now();
    std::qsort(a,len,sizeof(long),compare);
    /*vector<long> res;
    for (int i=1; i<=k; i++) {
        res.push_back(a[len-i]);
        //cout << a[len-i] << " ";
    }*/
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "Time for quick sort: " << duration.count() << "\n";
}
int compare(const void* a, const void* b)
{
    const int* x = (int*) a;
    const int* y = (int*) b;

    if (*x > *y)
        return 1;
    else if (*x < *y)
        return -1;

    return 0;
}

vector<long> presetArray() {
    return vector<long>{};
}
vector<long> randomizeArray(int len, int range) {
    vector<long> array;
    srand(time(0));
    for (int i=0; i<len; i++) {
        array.push_back(std::rand()%range-range/2);
    }
    return array;
}
void runTopKMax(int k, BsiAttribute<uint64_t>* bsi, vector<long> array) {
    //--- topKMax ---
    auto start = chrono::high_resolution_clock::now();
    HybridBitmap<uint64_t> topkmax = bsi->topKMax(k);
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "Time for topKMax: " << duration.count() << "\n";

    vector<long> topkmax_vector;
    //cout << "topkmax number of ones: " << topkmax.numberOfOnes() << "\n";
    for (int i=0; i<topkmax.sizeInBits(); i++) {
        if (topkmax.get(i)) {
            topkmax_vector.push_back(bsi->getValue(i));
            //cout << bsi->getValue(i) << " ";
        }
    }

    cout << "array length: " << topkmax_vector.size() << "\n";
    sort(topkmax_vector.begin(),topkmax_vector.end(),greater<long>());

    //--- verify accuracy ---
    int j = 0;
    bool correct = true;
    while (j<topkmax_vector.size()) {
        cout << topkmax_vector[j] << " " << array[array.size()-j-1] << "\n";
        if (topkmax_vector[j] != array[array.size()-j-1]) {
            correct = false;
            cout << j << "\n";
            //break;
        }
        j++;
    }
    if (correct && topkmax_vector.size() >= k) {
        cout << "\n" << "correct" << "\n";
    } else {
        cout << "\n" << "incorrect" << "\n";
    }
}
void runTopKMin(int k, BsiAttribute<uint64_t>* bsi, vector<long> array) {
    //--- topKMin ---
    auto start1 = chrono::high_resolution_clock::now();
    HybridBitmap<uint64_t> topkmin = bsi->topKMin(k);
    auto stop1 = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::microseconds>(stop1 - start1);
    cout << "Time for topKMin: " << duration1.count() << "\n";
    vector<long> topkmin_vector;
    cout << "topkmin number of ones: " << topkmin.numberOfOnes() << "\n";
    for (int i=0; i<topkmin.sizeInBits(); i++) {
        if (topkmin.get(i)) {
            topkmin_vector.push_back(bsi->getValue(i));
            //cout << bsi->getValue(i) << " ";
        }
    }
    cout << "array length: " << topkmin_vector.size() << "\n";
    sort(topkmin_vector.begin(),topkmin_vector.end());

    //--- verify accuracy ---
    int i = 0;
    bool correct = true;
    while (i<topkmin_vector.size()) {
        if (topkmin_vector[i] != array[i]) {
            correct = false;
            break;
        }
        i++;
    }
    if (correct && topkmin_vector.size() >= k) {
        cout << "\n" << "correct" << "\n";
    } else {
        cout << "\n" << "incorrect" << "\n";
    }
}