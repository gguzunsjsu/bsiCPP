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
void runRangeBetween(int range_begin, int range_end, BsiAttribute<uint64_t>* bsi, vector<long> array);
void runReLU(int threshold, BsiAttribute<uint64_t>* bsi, vector<long> array);
int main() {
    string filenames[] = {"rows100_skew1_card16_neg",
                          "rows1k_skew1_card16_neg",
                          "rows10k_skew1_card16_neg",
                          "rows100k_skew1_card16_neg",
                          "rows1M_skew1_card16_neg",
                          "rows10k_skew0_card16_neg",
                          "rows10k_skew0.5_card16_neg",
                          "rows10k_skew1.5_card16_neg",
                          "rows10k_skew2_card16_neg",
                          "rows10k_skew1_card4_neg",
                          "rows10k_skew1_card8_neg"};

    /*for (string filename: filenames) {
        processAndRun(filename);
    }*/
    processAndRun("signed_testcase");
    //processAndRun("rows100_skew1_card16_neg");

    return 0;
}
void processAndRun(string filename) {
    cout << filename << "\n";
    BsiSigned<uint64_t> build;
    BsiAttribute<uint64_t>* bsi;
    //int k = 500000;
    //int range_begin = -100;
    //int range_end = 100;
    int threshold = 100;
    vector<long> array;

    //--- read file ---
    string line;
    ifstream file("/Users/zhang/CLionProjects/bsiCPP/examples/"+filename);
    while (getline(file, line)) {
        long el = stol(line.substr(0, line.size() - 1));
        array.push_back(el);
    }
    file.close();
    /*if (array.size() < k) {
        return;
    }*/

    //--- buildBSI ---
    bsi = build.buildBsiAttributeFromVectorSigned(array,0.5);

    //--- test runtimes ---
    //runQuickSort(array);
    //runTopKMax(k,bsi,array);
    //runTopKMin(k,bsi,array);
    //runRangeBetween(range_begin,range_end,bsi,array);
    runReLU(threshold,bsi,array);

    array.clear();
}
void runReLU(int threshold, BsiAttribute<uint64_t>* bsi, vector<long> array) {
    //--- reLU ---
    HybridBitmap<uint64_t> relu;
    bsi->signMagnitudeToTwosInPlace(bsi->bits+1);
    for (int i=0; i<bsi->rows; i++) {
        cout << bsi->getValue(i) << " ";
    }
    cout << "\n";

    double time = 0;
    for (int i=0; i<5; i++) {
        auto start = chrono::high_resolution_clock::now();
        relu = bsi->reLU(threshold);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        time += duration.count();
    }
    cout << "Time for bsi reLU: " << time/5 << "\n";

    vector<long> ans;
    double time1 = 0;
    for (int i=0; i<5; i++) {
        ans.clear();
        auto start = chrono::high_resolution_clock::now();
        for (int j=0; j<array.size(); j++) {
            int num = array[j];
            if (num >= threshold) {
                ans.push_back(j);
            }
        }
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        time1 += duration.count();
    }
    cout << "Time for linear reLU: " << time1/5 << "\n";
    cout << ans.size() << " " << relu.numberOfOnes() << "\n";
    // check for accuracy
    if (ans.size() != relu.numberOfOnes()) {
        cout << "incorrect\n";
        vector<int> relu_pos = relu.positionsToVector();
        sort(relu_pos.begin(),relu_pos.end(),less<long>());
        sort(ans.begin(),ans.end(),less<long>());
        for (int i=0; i<max(ans.size(),relu_pos.size()); i++) {
            if (i < relu_pos.size()) {cout << relu_pos[i] << " ";} else {cout << -1 << " ";}
            if (i < ans.size()) {cout << ans[i] << "\n";} else {cout << -1 << "\n";}

        }
    } else {
        vector<int> relu_pos = relu.positionsToVector();
        sort(relu_pos.begin(),relu_pos.end(),greater<long>());
        sort(ans.begin(),ans.end(),greater<long>());
        bool res = true;
        for (int i=0; i<ans.size(); i++) {
            if (ans[i] != relu_pos[i]) {
                res = false;
                cout << relu_pos[i] << " " << ans[i] << "\n";
                //break;
            }
        }
        if (res) {
            cout << "correct\n";
        } else {
            cout << "incorrect\n";
        }
    }
}
void runRangeBetween(int range_begin, int range_end, BsiAttribute<uint64_t>* bsi, vector<long> array) {
    //--- rangeBetween ---
    HybridBitmap<uint64_t> range;
    double time = 0;
    for (int i=0; i<5; i++) {
        auto start = chrono::high_resolution_clock::now();
        range = bsi->rangeBetween(range_begin,range_end);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        time += duration.count();
    }
    cout << "Time for bsi rangeBetween: " << time/5 << "\n";

    vector<long> ans;
    double time1 = 0;
    cout << "size of array " << array.size() << "\n";
    for (int i=0; i<5; i++) {
        ans.clear();
        auto start = chrono::high_resolution_clock::now();
        for (int j=0; j<array.size(); j++) {
            int num = array[j];
            if (num <= range_end && num >= range_begin) {
                ans.push_back(j);
            }
        }
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        time1 += duration.count();
    }
    cout << "Time for linear rangeBetween: " << time1/5 << "\n";

    // check for accuracy
    if (ans.size() != range.numberOfOnes()) {
        cout << "incorrect\n";
        /*vector<int> range_pos = range.positionsToVector();
        sort(range_pos.begin(),range_pos.end(),less<long>());
        sort(ans.begin(),ans.end(),less<long>());
        bool res = true;
        for (int i=0; i<range_pos.size(); i++) {
            if (ans[i] != range_pos[i]) {
                res = false;
                cout << ans[i] << " " << range_pos[i] << "\n";
                //break;
            }
        }*/
    } else {
        vector<int> range_pos = range.positionsToVector();
        sort(range_pos.begin(),range_pos.end(),greater<long>());
        sort(ans.begin(),ans.end(),greater<long>());
        bool res = true;
        for (int i=0; i<ans.size(); i++) {
            if (ans[i] != range_pos[i]) {
                res = false;
                break;
            }
        }
        if (res) {
            cout << "correct\n";
        } else {
            cout << "incorrect\n";
        }
    }
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
    vector<int> topkmax_positions;
    double time2 = 0;
    for (int i=0; i<5; i++) {
        auto start = chrono::high_resolution_clock::now();
        topkmax_positions = topkmax.positionsToVector();
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        time2 += duration.count();
        //topkmax_positions.clear();
    }
    cout << "Time for hybridbitmap positionstovector: " << time2/5 << "\n";

    vector<long> topkmax_vector;
    double time3 = 0;
    for (int i=0; i<5; i++) {
        auto start = chrono::high_resolution_clock::now();
        topkmax_vector = bsi->positionsToVector(topkmax_positions);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        time3 += duration.count();
        //topkmax_vector.clear();
    }
    cout << "Time for bsiattribute positionstovector: " << time3/5 << "\n";

    cout << "array length: " << topkmax_vector.size() << "\n";
    sort(topkmax_vector.begin(),topkmax_vector.end(),greater<long>());

    vector<long> sorted_array;
    for (int el: array) {sorted_array.push_back(el);}
    sort(sorted_array.begin(),sorted_array.end(),greater<long>());

    //--- verify accuracy ---
    int j = 0;
    bool correct = true;
    while (j<topkmax_vector.size()) {
        //cout << topkmax_vector[j] << " " << array[j] << "\n";
        if (topkmax_vector[j] != sorted_array[j]) {
            correct = false;
            //cout << j << "\n";
            break;
        }
        j++;
    }
    if (correct && topkmax_vector.size() >= k) {
        cout << "correct" << "\n\n";
    } else {
        cout << "incorrect" << "\n\n";
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
    vector<int> topkmin_positions = topkmin.positionsToVector();
    vector<long> topkmin_vector = bsi->positionsToVector(topkmin_positions);

    cout << "array length: " << topkmin_vector.size() << "\n";
    sort(array.begin(),array.end(),less<long>());
    sort(topkmin_vector.begin(),topkmin_vector.end(),less<long>());

    //--- verify accuracy ---
    int i = 0;
    bool correct = true;
    while (i<topkmin_vector.size()) {
        //cout << topkmin_vector[i] << " " << array[i] << "\n";
        if (topkmin_vector[i] != array[i]) {
            correct = false;
            //cout << i << "\n";
            break;
        }
        i++;
    }
    if (correct && topkmin_vector.size() >= k) {
        cout << "correct" << "\n\n";
    } else {
        cout << "incorrect" << "\n\n";
    }
}