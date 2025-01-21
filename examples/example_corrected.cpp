//
// Created by gheor on 5/14/2022.
//


#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>

#include "../bsi/BsiUnsigned.hpp"
#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiAttribute.hpp"
#include "../bsi/hybridBitmap/hybridbitmap.h"
#include "../bsi/hybridBitmap/UnitTestsOfHybridBitmap.hpp"


int main(){
    BsiUnsigned<uint64_t> ubsi;
    BsiUnsigned<uint64_t> ubsi_1;
    BsiAttribute<uint64_t> *bsi_1;
    BsiAttribute<uint64_t> *bsi_2;
    BsiAttribute<uint64_t> *bsi_3;
    BsiAttribute<uint64_t> *bsi_result;

    BsiSigned<uint64_t> *bsi_s = new BsiSigned<uint64_t>();
    HybridBitmap<uint64_t> hybridBitmap;
    //ifstream fin1,fin2;
    //ofstream fout1;
    //fin1.open("/Users/adityapatel/multiplicationTestData1.txt");
    //fin2.open("/Users/adityapatel/multiplicationTestData2.txt");
    //fout1.open("/Users/adityapatel/multiplicationTestResult1.txt");

    vector<uint64_t> result_arr;
    vector<long> array1;
    vector<long> array2;
    vector<long> result;
    vector<double> v1;
    vector<double> v2;
    vector<double> vres;
    string line_str;
    int range1 = 10000;
    int range2 = 10000;
    int vectorLen = 1000;

    int arr1[5] = {84, 624, 9, 330, 240};
    int arr2[5] = {3, 6, 7, 9, 8};
    srand (time(NULL));
    for (int i=0; i<vectorLen; i++){
        array1.push_back(std::rand()%range1);
        array2.push_back(std::rand()%range2);
        v1.push_back((double(rand())/RAND_MAX));
        v2.push_back((double(rand())/RAND_MAX));
        //array1.push_back(arr1[i]);
        //array2.push_back(arr2[i]);

    }
//
//    while (getline(fin1, line_str)) {
//        istringstream buffer(line_str);
//        vector<int> line((istream_iterator<int>(buffer)),
//                         istream_iterator<int>());
//        for(auto it = line.begin(); it != line.end(); it++){
//            array1.push_back(*it);
//        }
//    }
//
//    while (getline(fin2, line_str)) {
//        istringstream buffer(line_str);
//        vector<int> line((istream_iterator<int>(buffer)),
//                         istream_iterator<int>());
//        for(auto it = line.begin(); it != line.end(); it++){
//            array2.push_back(*it);
//        }
//    }

    bsi_1 = ubsi.buildBsiAttributeFromVector(array1, 0.2);
    bsi_1->setPartitionID(0);
    bsi_1->setFirstSliceFlag(true);
    bsi_1->setLastSliceFlag(true);
    bsi_2 = ubsi.buildBsiAttributeFromVector(array2, 0.2);
    bsi_2->setPartitionID(0);
    bsi_2->setFirstSliceFlag(true);
    bsi_2->setLastSliceFlag(true);
    // bsi_result = ubsi.buildBsiAttributeFromArray(result, result.size(), 0.2);


//    HybridBitmap<uint64_t> andNot = bsi_1->bsi[1].andNot(bsi_1->bsi[0]);
//    HybridBitmap<uint64_t> negateAnd = bsi_1->bsi[0].Not().And(bsi_1->bsi[1]);


    for (int  i= 0 ; i< bsi_1->getNumberOfSlices() ; i++){
        ubsi.addSlice(bsi_1->getSlice(i));
    }
    HybridBitmap<uint64_t> existtenceBitmap = bsi_1->getExistenceBitmap();
    ubsi.setExistenceBitmap(bsi_1->getExistenceBitmap());
    ubsi.setNumberOfRows(bsi_1->rows);
    ubsi.setPartitionID(bsi_1->getPartitionID());
    ubsi.setFirstSliceFlag(true);
    ubsi.setLastSliceFlag(true);

    for (int  i= 0 ; i< bsi_2->getNumberOfSlices() ; i++){
        ubsi_1.addSlice(bsi_2->getSlice(i));
    }
    ubsi_1.setExistenceBitmap(bsi_2->getExistenceBitmap());
    ubsi_1.setNumberOfRows(bsi_2->rows);
    ubsi_1.setPartitionID(bsi_2->getPartitionID());
    ubsi_1.setFirstSliceFlag(true);
    ubsi_1.setLastSliceFlag(true);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    double sum = 0;
    long res;
    for (int i=0;i<vectorLen;i++){
        res = array1[i]*array2[i];
        //res = v1[i]*v2[i];
        result.push_back(res);
        //vres.push_back(res);
        sum= sum+res;
    }
    //bsi_3 = ubsi.multiplyWithBSI(ubsi_1);
    //bsi_3 = ubsi.peasantMultiply(ubsi_1);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    cout <<"Duration multiply array: \t\t"<< duration1<<endl;
    //cout<<"sum is: "<<sum<<endl;


    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();

//    for (int i=0;i<vectorLen;i++){
//        result.push_back(array1[i]*array2[i]);
//    }
    //bsi_3 = ubsi.peasantMultiply(ubsi_1);

    //bsi_3 = ubsi.SUMunsigned(bsi_2)->SUM(bsi_2)->SUM(bsi_2);
    //bsi_3 = ubsi.multiplyBSI(ubsi_1);
    bsi_3 = ubsi.multiplyWithBsiHorizontal(&ubsi_1,6);
    //bsi_3.su
    std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();
    cout <<"Duration bsi multiply inplace: \t"<< duration2<<endl;


    std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();

//    for (int i=0;i<vectorLen;i++){
//        result.push_back(array1[i]*array2[i]);
//    }
    //bsi_3 = ubsi.peasantMultiply(ubsi_1);

    //bsi_3 = ubsi.SUMunsigned(bsi_2)->SUM(bsi_2)->SUM(bsi_2);
    bsi_3 = ubsi.multiplyBSI(&ubsi_1);
    std::chrono::high_resolution_clock::time_point t6 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>( t6 - t5 ).count();
    cout <<"Duration bsi multiply : \t\t"<< duration3<<endl;


//
//    for(int i=0; i< result.size(); i++){
//        std::cout << result[i] <<" ";
//    }
//
//    std::cout << endl;
//
//    for(int i=0; i< array2.size(); i++){
//        string s1 = std::to_string(bsi_3->getValue(i));
//        std::cout << s1 <<" ";
//    }


//    ifstream fin1,fin2;
//    fin1.open("/Users/adityapatel/datasets_BSI/dataset.txt");
//    fin2.open("/Users/adityapatel/datasets_BSI/result_dataset_c++");
//    cout<<"\nHello world\n";
//    //std::cout<<bsi->isLastSlice();
//    vector<vector<uint64_t>> rawData;
//    vector<uint64_t> myvector1;
//    vector<BsiAttribute<uint64_t>*> bsiData;
//
//
////    bsi.push_back(ubsi.buildBsiAttributeFromArray(row, row.size(), 0.2));
//
//    string temp;
//
//    if(fin1.good()){
//        while(getline(fin1,temp))
//        {
//            stringstream   linestream(temp);
//            string         value;
//            vector<uint64_t> buffer;
//            while(getline(linestream,value,','))
//            {
//                uint64_t value_a;
//                std::istringstream iss(value);
//                iss >> value_a;
//                buffer.push_back(value_a);
//            }
//            rawData.push_back(buffer);
//        }
//    }else{
//        cerr<<"File can't be open"<<endl;
//        return -1;
//    }
//    int atts = rawData[0].size();
//    int rows = rawData.size();
//
//    cout<<"Total rows: "<<rows<<" Attributes: "<<atts<<endl;
//    for(vector<uint64_t>::iterator it = rawData[0].begin(); it < rawData[0].end(); it++){
//        cout<<*it<<" ";
//    }
//    cout<<endl;
//
//    const clock_t begin_time = clock();
//    for(int j=0;j<atts;j++){
//        vector<uint64_t> buffer;
//        BsiUnsigned<uint64_t> ubsi;
//
//        for(int i=1;i<rows;i++){
//            buffer.push_back(rawData[i][j]);
//            if(j==1){
//                myvector1.push_back(rawData[i][j]);
//            }
//        }
//        //cout<<" buffer size: "<<buffer.size()<<endl;
//        bsiData.push_back(ubsi.buildBsiAttributeFromArray(buffer, buffer.size(), 0.2));
//    }
//    for(int i=0; i< myvector1.size(); i++){
//        cout<<myvector1[i]<<" ";
//    }
//    cout<<endl;
//
//    HybridBitmap<uint64_t> topMaxValues = bsiData[1]->topKMax(1);
//    topMaxValues.printout();
//    topMaxValues.getDensity();
//    std::cout << "Total time"<<float( clock () - begin_time ) /  CLOCKS_PER_SEC<<endl;
//



//    for (int i=0; i<=1115; i++){
//        int temp = rand()%5000000;
//        myvector.push_back(temp);
//        myvector2.push_back(temp+7);
//        myvector1.push_back(i);
//    }

//    int vectorSize = myvector.size();
//    int vectorSize1 = myvector1.size();
//    int vectorSize2 = myvector2.size();
//    bsi_1 = bsi.buildBsiAttributeFromArray(myvector, vectorSize, 0.2);
//    bsi_2 = bsi.buildBsiAttributeFromArray(myvector1, vectorSize1, 0.2);
//    bsi_4 = bsi.buildBsiAttributeFromArray(myvector2, vectorSize2, 0.2);
//    int rows = bsi_1->getNumberOfSlices();
//
//
//    for (int  i= 0 ; i< bsi_1->getNumberOfSlices() ; i++){
//        unbsi.addSlice(bsi_1->getSlice(i));
//    }
//    HybridBitmap<uint64_t> existtenceBitmap = bsi_1->getExistenceBitmap();
//    unbsi.setExistenceBitmap(bsi_1->getExistenceBitmap());
//    bsi_3 = unbsi.multiplyByConstant(5);
//
//    cout<<"rows: "<<rows<<endl;
//   // bsi_2 = bsi_4->SUM(7);
//    HybridBitmap<uint64_t> b1 =bsi_1->topKMin(4);
//    HybridBitmap<uint64_t> b2 = bsi_1->topKMax(4);
//    cout<<"Minimum"<<endl;
//    b1.printout();
//    cout<<"Maximum"<<endl;
//    b2.printout();

















//    std::cout << "[testing testInEqualityEWAHBoolArray] sizeof(uword)="<< sizeof(uint64_t) << std::endl;
//    UnitTestOfHybridbitmap<uint64_t> unitTest;
//    HybridBitmap<uint64_t> b2(true);
//    HybridBitmap<uint64_t> b5(true);
//    HybridBitmap<uint64_t> b = HybridBitmap<uint64_t>::bitmapOf(1,1001);
//    HybridBitmap<uint64_t> b1 = HybridBitmap<uint64_t>::bitmapOf(256,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,138,140,142,144,146,148,150,152,154,156,158,160,162,164,166,168,170,172,174,176,178,180,182,184,186,188,190,192,194,196,198,200,202,204,206,208,210,212,214,216,218,220,222,224,226,228,230,232,234,236,238,240,242,244,246,248,250,252,254,256,258,260,262,264,266,268,270,272,274,276,278,280,282,284,286,288,290,292,294,296,298,300,302,304,306,308,310,312,314,316,318,320,322,324,326,328,330,332,334,336,338,340,342,344,346,348,350,352,354,356,358,360,362,364,366,368,370,372,374,376,378,380,382,384,386,388,390,392,394,396,398,400,402,404,406,408,410,412,414,416,418,420,422,424,426,428,430,432,434,436,438,440,442,444,446,448,450,452,454,456,458,460,462,464,466,468,470,472,474,476,478,480,482,484,486,488,490,492,494,496,498,500,502,504,506,508,510,512);
//
//    HybridBitmap<uint64_t> b3 = HybridBitmap<uint64_t>::bitmapOf(256,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127,129,131,133,135,137,139,141,143,145,147,149,151,153,155,157,159,161,163,165,167,169,171,173,175,177,179,181,183,185,187,189,191,193,195,197,199,201,203,205,207,209,211,213,215,217,219,221,223,225,227,229,231,233,235,237,239,241,243,245,247,249,251,253,255,257,259,261,263,265,267,269,271,273,275,277,279,281,283,285,287,289,291,293,295,297,299,301,303,305,307,309,311,313,315,317,319,321,323,325,327,329,331,333,335,337,339,341,343,345,347,349,351,353,355,357,359,361,363,365,367,369,371,373,375,377,379,381,383,385,387,389,391,393,395,397,399,401,403,405,407,409,411,413,415,417,419,421,423,425,427,429,431,433,435,437,439,441,443,445,447,449,451,453,455,457,459,461,463,465,467,469,471,473,475,477,479,481,483,485,487,489,491,493,495,497,499,501,503,505,507,509,511,513);
//    HybridBitmap<uint64_t> b4 = HybridBitmap<uint64_t>::bitmapOf(4, 1, 10, 11, 14);
//    b2 = HybridBitmap<uint64_t>::verbatimBitmapOf(256,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,138,140,142,144,146,148,150,152,154,156,158,160,162,164,166,168,170,172,174,176,178,180,182,184,186,188,190,192,194,196,198,200,202,204,206,208,210,212,214,216,218,220,222,224,226,228,230,232,234,236,238,240,242,244,246,248,250,252,254,256,258,260,262,264,266,268,270,272,274,276,278,280,282,284,286,288,290,292,294,296,298,300,302,304,306,308,310,312,314,316,318,320,322,324,326,328,330,332,334,336,338,340,342,344,346,348,350,352,354,356,358,360,362,364,366,368,370,372,374,376,378,380,382,384,386,388,390,392,394,396,398,400,402,404,406,408,410,412,414,416,418,420,422,424,426,428,430,432,434,436,438,440,442,444,446,448,450,452,454,456,458,460,462,464,466,468,470,472,474,476,478,480,482,484,486,488,490,492,494,496,498,500,502,504,506,508,510);
//    b5 = HybridBitmap<uint64_t>::verbatimBitmapOf(7,1, 2,3,4,6,8,11);
//    cout<<"b"<<endl;
//    //b.printout();
//    cout<<"b1"<<endl;
//    //b1.printout();
//    cout<<"b2"<<endl;
//  //  b2.printout();
//    cout<<"b3"<<endl;
////    b3.printout();
//    cout<<"b4"<<endl;
//   // b4.printout();
//    cout<<"b5"<<endl;
//    //b5.printout();
//    cout<<"b2.Xor(b4).printout()"<<endl;
//    HybridBitmap<uint64_t> ans = b3.xorHybrid(b2);
//    //ans.Or(b3);
//
//    cout<<"ans:"<<endl;
//    size_t numOfonesInAns = ans.numberOfOnes();
//    cout<<"numOfonesInAns: "<<numOfonesInAns<<endl;
////    ans.printout();
//


//
//    int arr[7] = {7, 5, 3, 4, 1, 0, 2};
//    int arr1[7] = {3, 4, 2, 2, 5, 7, 5};
//    int arr_result[7] = {75, 100, 180, 120, 132, 14, 50};




}
