#include <fstream>
#include "testBsiSigned.h"
#include "BsiAttribute.hpp"
#include "BsiSigned.hpp"
#include "BsiUnsigned.hpp"

using namespace std;
int compare(const void* a, const void* b);
void runQuickSort(vector<long> array);
void runTopKMax(int k, BsiAttribute<uint64_t>* bsi, vector<long> array);
void runTopKMin(int k, BsiAttribute<uint64_t>* bsi, vector<long> array);
void processAndRun(string filename);
int main() {
    string filenames[] = {"rows1k_skew1_card16_neg",
                          //"rows1M_skew1_card16_neg",
                          "rows10k_skew0.5_card16_neg",
                          "rows10k_skew0_card16_neg",
                          "rows10k_skew1.5_card16_neg",
                          "rows10k_skew1_card4_neg",
                          "rows10k_skew1_card8_neg",
                          "rows10k_skew1_card16_neg",
                          "rows10k_skew2_card16_neg",
                          "rows100_skew1_card16_neg",
                          "rows100k_skew1_card16_neg"};
    //ifstream file("/Users/zhang/CLionProjects/bsiCPP/examples/rows10k_skew0_card16_neg");
    for (string filename: filenames) {
        processAndRun(filename);
    }

    return 0;
}
void processAndRun(string filename) {
    cout << filename << "\n";
    BsiSigned<uint64_t> build;
    BsiAttribute<uint64_t>* bsi;
    int k = 5000;
    vector<long> array;

    //--- read file ---
    string line;
    ifstream file("/Users/zhang/CLionProjects/bsiCPP/examples/"+filename);
    while (getline(file, line)) {
        array.push_back(stol(line.substr(0,line.size()-1)));
    }
    file.close();
    if (array.size() < k) {
        return;
    }

    //--- buildBSI ---
    bsi = build.buildBsiAttributeFromVectorSigned(array,0.5);

    //--- test runtimes ---
    //runQuickSort(array);
    sort(array.begin(),array.end());
    runTopKMax(k,bsi,array);
    runTopKMin(k,bsi,array);

    array.clear();
}
void runQuickSort(vector<long> array) {
    //--- quick sort ---
    double time = 0;
    int len = array.size();
    for (int i=0; i<5; i++) {
        long* a = new long[len];
        for (int j=0; j<len; j++) {
            a[j] = array[j];
        }
        auto start = chrono::high_resolution_clock::now();
        std::qsort(a,len,sizeof(long),compare);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        time += duration.count();
        delete []a;
    }

    cout << "Average time for quick sort: " << time/5 << "\n";
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
    HybridBitmap<uint64_t> topkmax;
    double time = 0;
    for (int i=0; i<5; i++) {
        auto start = chrono::high_resolution_clock::now();
        topkmax = bsi->topKMax(k);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        time += duration.count();
    }
    cout << "Time for topKMax: " << time/5 << "\n";

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
        //cout << topkmax_vector[j] << " " << array[array.size()-j-1] << "\n";
        if (topkmax_vector[j] != array[array.size()-j-1]) {
            correct = false;
            cout << j << "\n";
            break;
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
    HybridBitmap<uint64_t> topkmin;
    double time = 0;
    for (int i=0; i<5; i++) {
        auto start1 = chrono::high_resolution_clock::now();
        topkmin = bsi->topKMin(k);
        auto stop1 = chrono::high_resolution_clock::now();
        auto duration1 = chrono::duration_cast<chrono::microseconds>(stop1 - start1);
        time += duration1.count();
    }
    cout << "Time for topKMin: " << time/5 << "\n";
    vector<long> topkmin_vector;
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