#include <iostream>
#include <fstream>
#include <chrono>

#include "BsiUnsigned.hpp"
#include "BsiSigned.hpp"
#include "BsiVector.hpp"
#include <sstream>

//void MultiplyVectorByScalar(vector<long>& v, int k) {
//    transform(v.begin(), v.end(), v.begin(), [k](long& c) { return c * k; });
//}
using namespace std;

void testInverse();
void testCompareTo();
void testMultByConstant();
void testMultSum();
void testUncompressedShift();
void testCompressedShift();
void testBuiltinOperators();
void testRelu();
void testSumConstant();
void testRandom();

vector<BsiVector<uint64_t>*> inv(vector<BsiVector<uint64_t>*> matrix);
void sgesv(int n, int m, vector<BsiVector<uint64_t>*> a, vector<int> ipiv, vector<BsiVector<uint64_t>*> b);
void sgetrf(int m, int n, vector<BsiVector<uint64_t>*> a, vector<int> ipiv);
void sgetrf2(int m, int n, vector<BsiVector<uint64_t>*> a, vector<int> ipiv);
void sgetrs(int n, int m, vector<BsiVector<uint64_t>*> a, vector<int> ipiv, vector<BsiVector<uint64_t>*> b);
int main() {
    testSumConstant();
    testRandom();
    return 0;
}

void testRandom() {
    int size = 640;
    BsiSigned<uint64_t> bsi;
    int precision = static_cast<int>(pow(10,5));
    BsiVector<uint64_t>* res;

    // test 1
    double high = 4.107443048544934;
    double low = 3.3826798753475913;
//    cout << static_cast<int>((high-low)*precision) << "\n";
//    res = bsi.createRandomBsi(size,static_cast<int>((high-low)*precision),0.5);

//    for (int i=0; i<res->rows; i++) {
//        cout << res->getValue(i) << " ";
//    }
//    cout << "\n";
//    res = res->SUM(static_cast<long>(low*precision));
//    res->decimals = precision;
//    for (int i=0; i<res->rows; i++) {
//        cout << res->getValue(i) << " ";
//    }
//    cout << "\n";

    // test 2
//    high = 3.918487309781472;
//    low = 3.1357142448781126;
//    res = bsi.createRandomBsi(size,static_cast<int>((high-low)*precision),0.5);
//    res = res->SUM(static_cast<long>(low*precision));
//    res->decimals = precision;
//    for (int i=0; i<res->rows; i++) {
//        cout << res->getValue(i) << " ";
//    }
//    cout << res->rows << "\n";

    // test 3
    high = 1;
    low = -1;
    res = bsi.createRandomBsi(size,static_cast<int>((high-low)*precision),0.5);
//    vector<long> randomVec = {16807, 75249, 50073, 143658, 108930, 11272, 27544, 50878, 177923, 37709, 164440, 38165, 84492, 43042, 7987, 122503, 82327, 131729, 178840, 142612, 144303, 33169, 17709, 97157, 129560, 170933, 93099, 80278, 151816, 195335, 99097, 7826, 13512, 129267, 123810, 77633, 180979, 179149, 36579, 158821, 111967, 10672, 101393, 19336, 125485, 91745, 25228, 94091, 40194, 186357, 35001, 121153, 16708, 157944, 115668, 171490, 188124, 102196, 109530, 60903, 127722, 54666, 128549, 118024, 197801, 6853, 140977, 47408, 108228, 174933, 60298, 138981, 28635, 8013, 23865, 189814, 139063, 154536, 139425, 1669, 124115, 20094, 75629, 46501, 146517, 134195, 30105, 10404, 119451, 194298, 182188, 21123, 199505, 46882, 6752, 131566, 136716, 130337, 154438, 53144, 51501, 44897, 183871, 155828, 111137, 19358, 153177, 182397, 122294, 77904, 21609, 146231, 159745, 44175, 53635, 66298, 98142, 80399, 27968, 90412, 260, 73557, 41594, 88008, 116395, 125968, 119113, 4530, 71006, 110962, 197942, 85365, 42082, 179852, 130767, 33821, 185695, 24712, 173671, 59901, 48590, 124831, 114738, 153057, 31616, 19790, 155640, 110679, 37335, 77006, 97972, 136098, 49095, 175319, 183454, 145223, 112289, 156760, 148905, 186126, 36123, 129506, 45813, 76770, 135238, 129094, 96220, 187844, 112366, 140534, 125226, 115394, 141363, 118738, 7844, 159590, 36550, 145159, 8623, 164947, 197385, 107217, 195272, 155539, 199247, 42385, 13496, 94885, 140623, 82420, 28144, 131968, 135735, 99915, 161625, 63534, 46042, 196011, 54679, 24152, 175244, 177295, 78818, 190396, 87692, 101815, 112991, 140033, 122669, 119397, 89553, 10547, 175825, 121210, 129662, 30211, 133808, 73377, 22761, 625, 137335, 1868, 177995, 2776, 62767, 197439, 29874, 89331, 140556, 56301, 68872, 100560, 120094, 118984, 113755, 39789, 51407, 28015, 30193, 22769, 15680, 103455, 16855, 67506, 170963, 33502, 186676, 63108, 163249, 2331, 77844, 87638, 160808, 139997, 28651, 54849, 45203, 95731, 196531, 198014, 195419, 36775, 42009, 130180, 87929, 195223, 44054, 188260, 76737, 191545, 147317, 37525, 4200, 134256, 21564, 160597, 106648, 137704, 25550, 125150, 141976, 101412, 138554, 82797, 102504, 148381, 34748, 7065, 59378, 47699, 78209, 125129, 178553, 65483, 50447, 64607, 180773, 45322, 64305, 155176, 170053, 127224, 184630, 80366, 5400, 60444, 95370, 170285, 83016, 193898, 8155, 41133, 115557, 96576, 153178, 58266, 187357, 21711, 27878, 120614, 74819, 195737, 183133, 20591, 23720, 48762, 106633, 176197, 176031, 38588, 60589, 54873, 198877, 37304, 179358, 61200, 177254, 68960, 14915, 93947, 45480, 7730, 69955, 74546, 37107, 41238, 140000, 184926, 145035, 174857, 161113, 50114, 129593, 1360, 98354, 168418, 25357, 142585, 113729, 121015, 176563, 106102, 60917, 186643, 19419, 62967, 147747, 45269, 55395, 23303, 4473, 119103, 111748, 24385, 5658, 163459, 45406, 73930, 139824, 85132, 100973, 159603, 46897, 107920, 3950, 135231, 88480, 20203, 13900, 24520, 62533, 81258, 17003, 18676, 20950, 23934, 158780, 79879, 141832, 82574, 33549, 174542, 106249, 52771, 166310, 115879, 983, 69970, 192040, 82723, 73650, 102971, 153229, 190318, 117746, 110299, 87230, 185621, 70776, 50124, 68244, 76958, 147696, 66771, 29064, 191560, 56598, 118751, 69940, 164503, 189551, 192801, 27205, 115080, 180418, 159274, 59649, 40413, 183320, 119025, 122012, 101783, 54788, 24117, 24008, 81550, 65324, 119195, 51257, 24511, 35690, 38666, 62410, 160593, 90553, 9565, 190960, 28742, 151403, 54352, 7307, 56141, 44910, 13200, 192799, 143127, 120171, 157787, 134414, 31625, 116641, 19517, 55348, 101842, 114315, 115974, 181445, 11373, 172220, 139911, 109239, 45054, 41305, 156929, 182253, 154189, 15166, 4356, 159304, 59860, 45898, 5592, 101720, 66116, 151580, 98867, 149352, 26939, 178698, 94901, 160811, 116616, 156891, 48671, 131728, 126671, 151661, 77045, 103120, 20240, 117158, 99453, 169627, 85350, 53711, 135563, 29594, 7533, 1727, 107827, 198795, 119531, 137442, 151516, 85006, 78515, 66924, 42601, 27097, 168661, 167103, 34324, 180108, 44936, 2281, 117015, 43710, 163216, 101905, 175973, 167781, 82720, 70791, 81169, 97865, 115827, 129536, 16728, 157125, 120384, 180167, 189949, 100476, 16046, 121576, 168666, 177467, 60520, 27384, 141991, 25210, 133258, 28211, 68744, 19518, 85718, 121934, 66825, 80767, 58689, 49818, 73014, 11825, 102918, 158028, 133029, 78034, 6727, 151240, 133754, 86540, 8396, 90935, 3083, 48048, 46865, 87011, 15252, 153064, 177876, 17237, 32675, 70245, 82710, 70415, 128192, 24710, 54747, 63037, 101299, 168447, 149715, 41693, 59579, 167129, 195164, 133979, 153501, 95525, 110289, 174958, 114033, 11800, 165089, 87670, 135217, 80846, 121038, 109543, 40338, 185936, 3321, 197719, 136496, 191247, 143788, 28359, 25987, 145035, 165235, 183912, 185684, 145989, 83784, 63734, 22667, 100519, 172615, 167045, 65976};
//    res = bsi.buildBsiVector(randomVec,0.5);
//    for (int i=0; i<res->rows; i++) {
//        cout << res->getValue(i) << ", ";
//    }
    long lowPrecise = low*precision;
    BsiVector<uint64_t>* test = res->SUM(lowPrecise);
    for (int i=0; i<res->rows; i++) {
        if (test->getValue(i) != res->getValue(i)+lowPrecise)
            cout << "expected " << res->getValue(i)+lowPrecise << " but got " << test->getValue(i) << "\n";
    }
}

void testSumConstant() {
//    vector<long> v = {6,4,3};
//    long c = 0;
//    BsiSigned<uint64_t> bsi;
//    BsiVector<uint64_t> *test = bsi.buildBsiVector(v,0.5);
//    BsiVector<uint64_t> *sol = bsi.buildBsiVector(v,0.5);
//    test = test->SUM(c);
//    for (int i=0; i<v.size(); i++) {
//        if (test->getValue(i) != sol->getValue(i)) {
//            cout << "Got " << test->getValue(i) << " but expected " << sol->getValue(i);
//            break;
//        }
//    }

    long c = -100000;
    BsiSigned<uint64_t> bsi;
    vector<long> vec = {16807, 75249, 50073, 143658};
//    vector<long> vec = {16807, 75249, 50073, 143658, 108930, 11272, 27544, 50878, 177923, 37709, 164440, 38165, 84492, 43042, 7987, 122503, 82327, 131729, 178840, 142612, 144303, 33169, 17709, 97157, 129560, 170933, 93099, 80278, 151816, 195335, 99097, 7826, 13512, 129267, 123810, 77633, 180979, 179149, 36579, 158821, 111967, 10672, 101393, 19336, 125485, 91745, 25228, 94091, 40194, 186357, 35001, 121153, 16708, 157944, 115668, 171490, 188124, 102196, 109530, 60903, 127722, 54666, 128549, 118024, 197801, 6853, 140977, 47408, 108228, 174933, 60298, 138981, 28635, 8013, 23865, 189814, 139063, 154536, 139425, 1669, 124115, 20094, 75629, 46501, 146517, 134195, 30105, 10404, 119451, 194298, 182188, 21123, 199505, 46882, 6752, 131566, 136716, 130337, 154438, 53144, 51501, 44897, 183871, 155828, 111137, 19358, 153177, 182397, 122294, 77904, 21609, 146231, 159745, 44175, 53635, 66298, 98142, 80399, 27968, 90412, 260, 73557, 41594, 88008, 116395, 125968, 119113, 4530, 71006, 110962, 197942, 85365, 42082, 179852, 130767, 33821, 185695, 24712, 173671, 59901, 48590, 124831, 114738, 153057, 31616, 19790, 155640, 110679, 37335, 77006, 97972, 136098, 49095, 175319, 183454, 145223, 112289, 156760, 148905, 186126, 36123, 129506, 45813, 76770, 135238, 129094, 96220, 187844, 112366, 140534, 125226, 115394, 141363, 118738, 7844, 159590, 36550, 145159, 8623, 164947, 197385, 107217, 195272, 155539, 199247, 42385, 13496, 94885, 140623, 82420, 28144, 131968, 135735, 99915, 161625, 63534, 46042, 196011, 54679, 24152, 175244, 177295, 78818, 190396, 87692, 101815, 112991, 140033, 122669, 119397, 89553, 10547, 175825, 121210, 129662, 30211, 133808, 73377, 22761, 625, 137335, 1868, 177995, 2776, 62767, 197439, 29874, 89331, 140556, 56301, 68872, 100560, 120094, 118984, 113755, 39789, 51407, 28015, 30193, 22769, 15680, 103455, 16855, 67506, 170963, 33502, 186676, 63108, 163249, 2331, 77844, 87638, 160808, 139997, 28651, 54849, 45203, 95731, 196531, 198014, 195419, 36775, 42009, 130180, 87929, 195223, 44054, 188260, 76737, 191545, 147317, 37525, 4200, 134256, 21564, 160597, 106648, 137704, 25550, 125150, 141976, 101412, 138554, 82797, 102504, 148381, 34748, 7065, 59378, 47699, 78209, 125129, 178553, 65483, 50447, 64607, 180773, 45322, 64305, 155176, 170053, 127224, 184630, 80366, 5400, 60444, 95370, 170285, 83016, 193898, 8155, 41133, 115557, 96576, 153178, 58266, 187357, 21711, 27878, 120614, 74819, 195737, 183133, 20591, 23720, 48762, 106633, 176197, 176031, 38588, 60589, 54873, 198877, 37304, 179358, 61200, 177254, 68960, 14915, 93947, 45480, 7730, 69955, 74546, 37107, 41238, 140000, 184926, 145035, 174857, 161113, 50114, 129593, 1360, 98354, 168418, 25357, 142585, 113729, 121015, 176563, 106102, 60917, 186643, 19419, 62967, 147747, 45269, 55395, 23303, 4473, 119103, 111748, 24385, 5658, 163459, 45406, 73930, 139824, 85132, 100973, 159603, 46897, 107920, 3950, 135231, 88480, 20203, 13900, 24520, 62533, 81258, 17003, 18676, 20950, 23934, 158780, 79879, 141832, 82574, 33549, 174542, 106249, 52771, 166310, 115879, 983, 69970, 192040, 82723, 73650, 102971, 153229, 190318, 117746, 110299, 87230, 185621, 70776, 50124, 68244, 76958, 147696, 66771, 29064, 191560, 56598, 118751, 69940, 164503, 189551, 192801, 27205, 115080, 180418, 159274, 59649, 40413, 183320, 119025, 122012, 101783, 54788, 24117, 24008, 81550, 65324, 119195, 51257, 24511, 35690, 38666, 62410, 160593, 90553, 9565, 190960, 28742, 151403, 54352, 7307, 56141, 44910, 13200, 192799, 143127, 120171, 157787, 134414, 31625, 116641, 19517, 55348, 101842, 114315, 115974, 181445, 11373, 172220, 139911, 109239, 45054, 41305, 156929, 182253, 154189, 15166, 4356, 159304, 59860, 45898, 5592, 101720, 66116, 151580, 98867, 149352, 26939, 178698, 94901, 160811, 116616, 156891, 48671, 131728, 126671, 151661, 77045, 103120, 20240, 117158, 99453, 169627, 85350, 53711, 135563, 29594, 7533, 1727, 107827, 198795, 119531, 137442, 151516, 85006, 78515, 66924, 42601, 27097, 168661, 167103, 34324, 180108, 44936, 2281, 117015, 43710, 163216, 101905, 175973, 167781, 82720, 70791, 81169, 97865, 115827, 129536, 16728, 157125, 120384, 180167, 189949, 100476, 16046, 121576, 168666, 177467, 60520, 27384, 141991, 25210, 133258, 28211, 68744, 19518, 85718, 121934, 66825, 80767, 58689, 49818, 73014, 11825, 102918, 158028, 133029, 78034, 6727, 151240, 133754, 86540, 8396, 90935, 3083, 48048, 46865, 87011, 15252, 153064, 177876, 17237, 32675, 70245, 82710, 70415, 128192, 24710, 54747, 63037, 101299, 168447, 149715, 41693, 59579, 167129, 195164, 133979, 153501, 95525, 110289, 174958, 114033, 11800, 165089, 87670, 135217, 80846, 121038, 109543, 40338, 185936, 3321, 197719, 136496, 191247, 143788, 28359, 25987, 145035, 165235, 183912, 185684, 145989, 83784, 63734, 22667, 100519, 172615, 167045, 65976};
    BsiVector<uint64_t> *test = bsi.buildBsiVector(vec,0.5);
    test = test->SUM(c);
    for (int i=0; i<test->rows; i++) {
        if (test->getValue(i) != vec[i]+c) {
            cout << "Got " << test->getValue(i) << " but expected " << vec[i]+c;
            break;
        }
    }
}

void testRelu() {
    BsiSigned<uint64_t> bsi;
    std::vector<long> vec = {0,1};
    BsiVector<uint64_t>* bsi_test = bsi.buildBsiVector(vec,0.5);
    std::vector<long> vec1 = {10,20,-30,4,3, -3,-102,30000};
    std::vector<long> vec2 = {5,-3,  2,12,20,-23, 9, 103000};
    BsiVector<uint64_t>* a = bsi.buildBsiVector(vec1,0.5);
    BsiVector<uint64_t>* b = bsi.buildBsiVector(vec2,0.5);
    HybridBitmap<uint64_t> res = a->reLU(b);
    std::vector<int> sol = {1,1,0,0,0,1,0,0};
    for (int i=0; i<vec1.size(); i++) {
        if (res.get(i) != sol[i]) {
            cout << "Difference at index " << i << ": " << res.get(i) << " is not the same as " << sol[i] << "\n";
        }
    }
}

void testBuiltinOperators() {
    BsiSigned<uint64_t> bsi;
    std::vector<long> vec1 = {10,20,-30,4};
    std::vector<long> vec2 = {5,-3,2,12};
    BsiVector<uint64_t>* a = bsi.buildBsiVector(vec1,0.5);
    BsiVector<uint64_t>* b = bsi.buildBsiVector(vec2,0.5);

    BsiVector<uint64_t>* sum = (*a)+b;
    std::vector<long> vec_sum = {15,17,-28,16};
    for (int i=0; i<vec_sum.size(); i++) {
        if (sum->getValue(i) != vec_sum[i]) {
            cout << "Difference at index " << i << ": " << sum->getValue(i) << " is not the same as " << vec_sum[i] << "\n";
        }
    }

    BsiVector<uint64_t>* mult = (*a)*b;
    std::vector<long> vec_mult = {50,-60,-60,48};
    for (int i=0; i<vec_mult.size(); i++) {
        if (mult->getValue(i) != vec_mult[i]) {
            cout << "Difference at index " << i << ": " << mult->getValue(i) << " is not the same as " << vec_mult[i] << "\n";
        }
    }

    BsiVector<uint64_t>* mult_const = (*a)*3;
    std::vector<long> vec_mult_const = {30,60,-90,12};
    for (int i=0; i<vec_mult_const.size(); i++) {
        if (mult_const->getValue(i) != vec_mult_const[i]) {
            cout << "Difference at index " << i << ": " << mult_const->getValue(i) << " is not the same as " << vec_mult_const[i] << "\n";
        }
    }
}

void testUncompressedShift() {
    BsiSigned<uint64_t> bsi;
    std::vector<long> vec = {10,20,-30,40};
    BsiVector<uint64_t>* orig = bsi.buildBsiVector(vec,0.5);

    cout << "Left uncompressed\n";
    BsiVector<uint64_t>* left_shifted = orig->shift(-1);
    std::vector<int> expected_left = {20,-30,40,0};
    for (int i=0;i<vec.size(); i++) {
        if (left_shifted->getValue(i) != expected_left[i]) {
            cout << "Difference at index " << i << ": " << left_shifted->getValue(i) << " is not the same as " << expected_left[i] << "\n";
        }
    }

    cout << "Right uncompressed\n";
    BsiVector<uint64_t>* right_shifted = orig->shift(1);
    std::vector<int> expected_right = {0,10,20,-30};
    for (int i=0;i<vec.size(); i++) {
        if (right_shifted->getValue(i) != expected_right[i]) {
            cout << "Difference at index " << i << ": " << right_shifted->getValue(i) << " is not the same as " << expected_right[i] << "\n";
        }
    }
}
void testCompressedShift() {
    BsiSigned<uint64_t> bsi;
    std::vector<long> vec = {80,10,-20,-80};
    BsiVector<uint64_t>* orig = bsi.buildBsiVector(vec,1.1);

    cout << "Left compressed\n";
    BsiVector<uint64_t>* left_shifted = orig->shift(-1);
    std::vector<int> expected_left = {10,-20,-80,0};
    for (int i=0;i<vec.size(); i++) {
        if (left_shifted->getValue(i) != expected_left[i]) {
            cout << "Difference at index " << i << ": " << left_shifted->getValue(i) << " is not the same as " << expected_left[i] << "\n";
        }
    }

    cout << "Right compressed\n";
    BsiVector<uint64_t>* right_shifted = orig->shift(1);
    std::vector<int> expected_right = {0,80,10,-20};
    for (int i=0;i<vec.size(); i++) {
        if (right_shifted->getValue(i) != expected_right[i]) {
            cout << "Difference at index " << i << ": " << right_shifted->getValue(i) << " is not the same as " << expected_right[i] << "\n";
        }
    }
}
void testMultSum() {
    BsiSigned<uint64_t> bsi;
    ifstream file("./testcase.txt");
    vector<BsiVector<uint64_t>*> H_bsi;
    for (int i=0; i<9; i++) {
        string line;
        getline(file,line);
        stringstream ss(line);
        long num;
        vector<long> H;
        while (ss >> num) {
            H.push_back(num);
        }
        H_bsi.push_back(bsi.buildBsiVector(H,0.5));
    }
    string line;
    getline(file,line);
    stringstream ss(line);
    long num;
    vector<long> i;
    while (ss >> num) {
        i.push_back(num);
    }
    BsiVector<uint64_t>* i_bsi = bsi.buildBsiVector(i,0.5);
    getline(file,line);
    stringstream ss2(line);
    vector<long> j;
    while (ss2 >> num) {
        j.push_back(num);
    }
    int PRECISION = 1;
    BsiVector<uint64_t>* j_bsi = bsi.buildBsiVector(j,0.5);

    /*BsiVector<uint64_t>* u_bsi = H_bsi[0]->multiplyWithBsiHorizontal(j_bsi,PRECISION)->SUM(H_bsi[1]->multiplyWithBsiHorizontal(i_bsi,PRECISION)->SUM(H_bsi[2]));
    BsiVector<uint64_t>* v_bsi = H_bsi[3]->multiplyWithBsiHorizontal(j_bsi,PRECISION)->SUM(H_bsi[4]->multiplyWithBsiHorizontal(i_bsi,PRECISION)->SUM(H_bsi[5]));
    BsiVector<uint64_t>* w_bsi = H_bsi[6]->multiplyWithBsiHorizontal(j_bsi,PRECISION)->SUM(H_bsi[7]->multiplyWithBsiHorizontal(i_bsi,PRECISION)->SUM(H_bsi[8]));

    for (int k=0; k<u_bsi->rows; k++) {
        long u = H_bsi[0]->getValue(k) * j_bsi->getValue(k) + H_bsi[1]->getValue(k) * i_bsi->getValue(k) + H_bsi[2]->getValue(k);
        long v = H_bsi[3]->getValue(k) * j_bsi->getValue(k) + H_bsi[4]->getValue(k) * i_bsi->getValue(k) + H_bsi[5]->getValue(k);
        long w = H_bsi[6]->getValue(k) * j_bsi->getValue(k) + H_bsi[7]->getValue(k) * i_bsi->getValue(k) + H_bsi[8]->getValue(k);
        cout << "u: " << u_bsi->getValue(k) << " " << u << " v: " << v_bsi->getValue(k) << " " << v << " w: " << w_bsi->getValue(k) << " " << w << "\n";
    }*/
    BsiVector<uint64_t>* u_bsi1 = H_bsi[0]->multiplyWithBsiHorizontal(j_bsi,PRECISION);
    BsiVector<uint64_t>* u_bsi2 = H_bsi[1]->multiplyWithBsiHorizontal(i_bsi,PRECISION);
    BsiVector<uint64_t>* u_bsi3 = u_bsi1->SUM(u_bsi2);
    BsiVector<uint64_t>* u_bsi4 = u_bsi3->SUM(H_bsi[2]);
    u_bsi3->getValue(0);
    for (int k=0; k<u_bsi1->rows; k++) {
        long u1 = H_bsi[0]->getValue(k) * j_bsi->getValue(k);
        long u2 = H_bsi[1]->getValue(k) * i_bsi->getValue(k);
        long u3 = u1+u2;
        long u4 = u3+H_bsi[2]->getValue(k);
        cout << "u1: " << u_bsi1->getValue(k) << " " << u1 << " u2: " << u_bsi2->getValue(k) << " " << u2 << " u3: " << u_bsi3->getValue(k) << " " << u3 << " u4: " << u_bsi4->getValue(k) << " " << u4 << "\n";
    }
}

void testMultByConstant() {
    vector<long> v = {6,4,3};
    int c = 100000000;
    BsiSigned<uint64_t> bsi;
    BsiVector<uint64_t> *test = bsi.buildBsiVector(v,0.5);
    for (int i=0; i<v.size(); i++) {
        v[i] *= c;
    }
    BsiVector<uint64_t> *sol = bsi.buildBsiVector(v,0.5);
    test = test->multiplyByConstant(c);
    for (int i=0; i<v.size(); i++) {
        cout << test->getValue(i) << " ";
    }
    cout << "\n";
    for (int i=0; i<test->getNumberOfSlices(); i++) {
        if (test->getSlice(i) != sol->getSlice(i)) {
            break;
        }
    }
    cout << "finish testing mult by constant";
}

void testInverse() {
    BsiSigned<uint64_t> bsi;
    vector<BsiVector<uint64_t>*> mat;
    vector<long> r1 = {4,3};
    vector<long> r2 = {3,2};
    mat.push_back(bsi.buildBsiVector(r1, 0.5));
    mat.push_back(bsi.buildBsiVector(r2, 0.5));
    vector<BsiVector<uint64_t>*> res = inv(mat);
}

vector<BsiVector<uint64_t>*> inv(vector<BsiVector<uint64_t>*> mat) {
    int precision = 10000;
    int n = mat.size();
    vector<BsiVector<uint64_t>*> res; // initialize as identity matrix
    vector<long> row;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            cout << mat[i]->getValue(j) << " ";
        }
        cout <<"\n";
    }
    for (int i=0; i<n; i++) {
        row.push_back(0);
        mat[i] = mat[i]->multiplyByConstant(precision);
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            cout << mat[i]->getValue(j) << " ";
        }
        cout <<"\n";
    }
    BsiSigned<uint64_t> bsi;
    for (int i=0; i<n; i++) {
        row[i] = precision;
        res.push_back(bsi.buildBsiVector(row,0.5));
        row[i] = 0;
    }

    // Gaussian elimination with partial pivoting
    for (int i=0; i<n-1; i++) {
        // Find maximum possible pivot in column for numerical stability
        int piv = i;
        for (int j = i+1; j<n; j++) {
            if (mat[i]->compareTo(mat[j],i) < 0) {
                piv = j;
            }
        }
        // Interchange rows
        BsiVector<uint64_t>* temp = mat[i];
        mat[i] = mat[piv];
        mat[piv] = temp;

        temp = res[i];
        res[i] = res[piv];
        res[piv] = temp;

        // Calculate rows (i+1):n
        for (int j = i+1; j<n; j++) {
            int l = mat[j]->getValue(i)/mat[i]->getValue(i)*(-1);
            mat[j] = mat[j]->SUM(mat[i]->multiplyByConstant(l));
        }
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            cout << res[i]->getValue(j) << " ";
        }
        cout <<"\n";
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            cout << res[i]->getValue(j) << " ";
        }
        cout <<"\n";
    }
    return res;
}
/*
 * LAPACK subroutine for computing the solution to a real system of linear equations
 * A * X = B
 * where A is an n-by-n matrix, and X and B are n-by-m matrices.
 * LU decomposition with partial pivoting and row interchanges is used to factor A as
 * A = P * L * U
 * Matrices are implemented with arrays of signed BsiVector's
*/
void sgesv(int n, int m, vector<BsiVector<uint64_t>*> a, vector<int> ipiv, vector<BsiVector<uint64_t>*> b) {
    // Compute LU factorization of A
    sgetrf(n, n, a, ipiv);
    // Solve the system A * X = B by overwriting B with X
    sgetrs(n, m, a, ipiv, b);
}

/*
 * Auxiliary call to LAPACK subroutine for computing an LU factorization
*/
void sgetrf(int m, int n, vector<BsiVector<uint64_t>*> a, vector<int> ipiv) {
    if (m == 0 || n == 0) return;
    sgetrf2(m, n, a, ipiv);
}

/*
 * LAPACK subroutine for computing an LU factorization of a general m-by-n matrix A
 * using partial pivoting with row interchanges
 * A = P * L * U
 * where P is a permutation matrix, L is lower triangular with unit diagonal elements
 * (lower trapezoidal if m > n), and U is upper triangular (upper trapezoidal if m < n).
 * Recursively divides matrix into four submatrices
 * A = [ A11 A12 ]
 *     [ A21 A22 ]
 * where A11 is n1-by-n1, A22 is n2-by-n2, n1 = min(m,n)/2, n2 = n-n1
 * Matrices are implemented with arrays of signed BsiVector's
*/
void sgetrf2(int m, int n, vector<BsiVector<uint64_t>*> a, vector<int> ipiv) {
    if (m == 0 || n == 0) return;
    if (m == 1) ipiv[0] = 0;
    else if (n == 1) {
        // Find pivot and test for singularity
        int i = 0;
        for (int j = 0; j<m; j++) {
            if (a[i]->compareTo(a[j],0) < 0) {
                i = j;
            }
        }
        ipiv[0] = i;
        if (a[i]->getValue(0) != 0) {
            // Apply the interchange
            if (i != 0) {
                auto temp = a[0];
                a[0] = a[i];
                a[i] = temp;
            }
            // Compute elements 2:M of the column

        }
    }
}

void testCompareTo() {
    BsiSigned<uint64_t> build;
    int n = 5;
    string line;
    ifstream file("./testcase.txt");
    int j = 0;
    while (j < 80000) {
        vector<long> v1;
        vector<long> v2;
        string s1;
        string s2;
        for (int i = 0; i < n; i++) {
            getline(file, line);
            cout << line;
            try {v1.push_back(stol(line));}
            catch (invalid_argument e) {
                cout << "can't convert to long: line " << (j*10 + i) << " element: " << line << "\n";
                return;
            }
            s1 += line +" ";
        }
        for (int i = 0; i < n; i++) {
            getline(file, line);
            try {v2.push_back(stol(line));}
            catch (invalid_argument e) {
                cout << "can't convert to long: line " << (j*10 + 5 + i) << " element: " << line << "\n";
                return;
            }
            s2 += line +" ";
        }
        //if (j < 17) {j++;continue;}
        for (int i = 0; i < n; i++) {
            //cout << v1[i] << " " << v2[i] << "\n";
            BsiVector<uint64_t> *bsi1 = build.buildBsiVector(v1, 0.5);
            BsiVector<uint64_t> *bsi2 = build.buildBsiVector(v2, 0.5);
            int res = bsi1->compareTo(bsi2,i);
            if (v1[i] < v2[i]) {
                if (res != -1) {
                    cout << j << "th iteration: " << s1 << ", " << s2 << "\n";
                    cout << "index " << i << ": " << v1[i] << ", " << v2[i] << " got result " << res << "\n";
                    res = bsi1->compareTo(bsi2,i);
                }
            } else if (v1[i] == v2[i]) {
                if (res != 0) {
                    cout << j << "th iteration: " << s1 << ", " << s2 << "\n";
                    cout << "index " << i << ": " << v1[i] << ", " << v2[i] << " got result " << res << "\n";
                    res = bsi1->compareTo(bsi2,i);
                }
            } else {
                if (res != 1) {
                    cout << j << "th iteration: " << s1 << ", " << s2 << "\n";
                    cout << "index " << i << ": " << v1[i] << ", " << v2[i] << " got result " << res << "\n";
                    res = bsi1->compareTo(bsi2,i);
                }
            }
        }
        v1.clear();
        v2.clear();
        j ++;
    }
}

void sgetrs(int n, int m, vector<BsiVector<uint64_t>*> a, vector<int> ipiv, vector<BsiVector<uint64_t>*> b) {

}