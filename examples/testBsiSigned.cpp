#include <fstream>
#include "testBsiSigned.h"
#include "BsiAttribute.hpp"
#include "BsiSigned.hpp"
#include "BsiUnsigned.hpp"

using namespace std;

int main() {
    BsiSigned<uint64_t> build;
    BsiAttribute<uint64_t>* bsi;
    int k = 20;

    //--- preset array ---
    //vector<long> array{-4981, -4979, -4966, -4933, -4932, -4931, -4927, -4927, -4925, -4916, -4914, -4914, -4907, -4905, -4902, -4898, -4886, -4885, -4882, -4847, -4841, -4823, -4816, -4804, -4801, -4796, -4786, -4785, -4779, -4775, -4772, -4765, -4760, -4750, -4735, -4733, -4732, -4726, -4720, -4720, -4716, -4708, -4696, -4663, -4658, -4657, -4649, -4588, -4585, -4569, -4568, -4549, -4545, -4544, -4536, -4526, -4524, -4520, -4520, -4504, -4490, -4484, -4479, -4461, -4456, -4456, -4450, -4437, -4434, -4429, -4421, -4394, -4379, -4367, -4339, -4315, -4281, -4281, -4267, -4265, -4258, -4251, -4245, -4244, -4233, -4231, -4228, -4226, -4225, -4212, -4186, -4165, -4161, -4150, -4148, -4140, -4138, -4108, -4104, -4103, -4099, -4094, -4087, -4086, -4075, -4062, -4058, -4055, -4027, -4016, -4011, -4002, -3986, -3978, -3974, -3973, -3972, -3971, -3969, -3968, -3957, -3950, -3947, -3946, -3941, -3926, -3926, -3922, -3920, -3871, -3860, -3855, -3839, -3839, -3810, -3806, -3802, -3798, -3790, -3757, -3748, -3746, -3745, -3729, -3712, -3711, -3706, -3703, -3683, -3676, -3667, -3649, -3642, -3641, -3637, -3626, -3622, -3619, -3616, -3596, -3580, -3573, -3567, -3548, -3545, -3543, -3519, -3519, -3507, -3491, -3490, -3428, -3428, -3420, -3412, -3409, -3407, -3406, -3356, -3348, -3345, -3297, -3269, -3250, -3249, -3243, -3231, -3220, -3207, -3196, -3176, -3140, -3139, -3137, -3122, -3108, -3095, -3080, -3074, -3067, -3065, -3049, -3046, -3023, -3017, -3003, -3001, -2993, -2981, -2975, -2975, -2963, -2963, -2930, -2925, -2917, -2905, -2902, -2900, -2898, -2877, -2874, -2871, -2851, -2839, -2833, -2827, -2819, -2808, -2804, -2781, -2760, -2750, -2746, -2743, -2735, -2719, -2717, -2709, -2699, -2696, -2693, -2693, -2685, -2675, -2659, -2653, -2634, -2630, -2628, -2623, -2619, -2610, -2607, -2603, -2597, -2595, -2594, -2576, -2565, -2562, -2555, -2538, -2534, -2522, -2517, -2502, -2500, -2494, -2489, -2489, -2480, -2472, -2449, -2448, -2439, -2431, -2422, -2419, -2418, -2412, -2401, -2394, -2391, -2387, -2376, -2375, -2363, -2363, -2351, -2343, -2336, -2331, -2324, -2319, -2317, -2315, -2299, -2283, -2280, -2273, -2272, -2258, -2252, -2248, -2247, -2205, -2204, -2189, -2181, -2180, -2178, -2178, -2178, -2163, -2159, -2143, -2093, -2058, -2053, -2051, -2021, -2012, -2007, -2007, -2001, -1999, -1997, -1995, -1991, -1985, -1979, -1974, -1971, -1955, -1943, -1937, -1935, -1931, -1926, -1919, -1917, -1908, -1852, -1852, -1833, -1817, -1814, -1813, -1802, -1789, -1787, -1768, -1754, -1740, -1736, -1731, -1724, -1713, -1676, -1669, -1669, -1664, -1654, -1654, -1627, -1622, -1620, -1615, -1612, -1609, -1580, -1565, -1558, -1557, -1554, -1549, -1537, -1532, -1531, -1522, -1518, -1509, -1484, -1474, -1464, -1447, -1437, -1436, -1428, -1428, -1425, -1409, -1391, -1390, -1384, -1373, -1366, -1365, -1331, -1330, -1321, -1318, -1283, -1283, -1253, -1250, -1245, -1240, -1236, -1234, -1227, -1212, -1194, -1189, -1182, -1181, -1179, -1162, -1150, -1143, -1142, -1137, -1099, -1095, -1088, -1078, -1078, -1077, -1076, -1074, -1057, -1056, -1027, -1020, -983, -981, -980, -946, -942, -938, -931, -926, -926, -918, -913, -902, -894, -865, -863, -805, -796, -794, -791, -790, -784, -774, -774, -749, -742, -716, -714, -686, -685, -682, -672, -652, -652, -641, -635, -628, -621, -601, -570, -560, -540, -539, -530, -528, -513, -487, -462, -458, -446, -440, -434, -411, -396, -393, -372, -341, -339, -336, -333, -326, -316, -311, -310, -306, -293, -285, -282, -276, -275, -271, -269, -263, -237, -226, -225, -209, -191, -186, -185, -180, -170, -166, -138, -91, -78, -61, -17, -11, 0, 11, 16, 22, 27, 37, 45, 114, 114, 116, 118, 119, 130, 137, 145, 147, 157, 174, 182, 187, 210, 243, 243, 253, 255, 270, 276, 287, 288, 306, 324, 335, 339, 348, 358, 369, 378, 385, 399, 404, 414, 439, 456, 457, 460, 482, 484, 501, 505, 515, 532, 547, 551, 571, 581, 585, 602, 630, 636, 643, 663, 677, 723, 725, 726, 727, 740, 752, 779, 784, 799, 801, 838, 850, 876, 885, 892, 896, 911, 924, 924, 937, 943, 961, 965, 969, 976, 988, 1003, 1030, 1035, 1042, 1043, 1044, 1053, 1053, 1061, 1085, 1089, 1090, 1096, 1131, 1149, 1153, 1153, 1153, 1163, 1167, 1173, 1207, 1232, 1238, 1252, 1255, 1289, 1308, 1314, 1314, 1333, 1339, 1342, 1343, 1374, 1380, 1380, 1398, 1411, 1416, 1423, 1437, 1453, 1481, 1493, 1494, 1527, 1531, 1535, 1544, 1547, 1577, 1607, 1607, 1636, 1647, 1668, 1690, 1696, 1702, 1704, 1716, 1737, 1747, 1754, 1758, 1764, 1774, 1775, 1792, 1793, 1797, 1803, 1806, 1822, 1826, 1826, 1850, 1853, 1861, 1878, 1898, 1906, 1908, 1921, 1924, 1934, 1942, 1942, 1944, 1957, 1965, 1970, 1995, 2011, 2013, 2019, 2019, 2024, 2027, 2030, 2039, 2039, 2040, 2042, 2061, 2066, 2076, 2076, 2098, 2107, 2133, 2134, 2138, 2169, 2190, 2199, 2207, 2207, 2224, 2234, 2243, 2249, 2255, 2264, 2266, 2279, 2291, 2292, 2317, 2322, 2346, 2354, 2355, 2359, 2373, 2386, 2396, 2406, 2417, 2456, 2461, 2465, 2487, 2496, 2513, 2526, 2534, 2545, 2552, 2552, 2572, 2582, 2589, 2590, 2617, 2623, 2636, 2640, 2644, 2650, 2658, 2659, 2666, 2669, 2685, 2690, 2694, 2697, 2706, 2717, 2722, 2723, 2729, 2731, 2736, 2738, 2764, 2768, 2778, 2781, 2781, 2784, 2788, 2807, 2833, 2842, 2855, 2874, 2880, 2890, 2898, 2904, 2907, 2908, 2909, 2913, 2921, 2932, 2952, 2954, 2961, 2980, 3009, 3050, 3052, 3053, 3055, 3066, 3068, 3108, 3114, 3153, 3153, 3164, 3172, 3179, 3203, 3215, 3224, 3227, 3243, 3249, 3266, 3277, 3279, 3297, 3310, 3318, 3333, 3344, 3357, 3358, 3370, 3380, 3388, 3395, 3398, 3416, 3417, 3427, 3442, 3449, 3468, 3468, 3474, 3476, 3476, 3480, 3486, 3517, 3534, 3558, 3568, 3570, 3577, 3591, 3605, 3609, 3610, 3659, 3665, 3674, 3675, 3679, 3695, 3697, 3716, 3735, 3739, 3756, 3779, 3786, 3789, 3830, 3834, 3845, 3853, 3873, 3879, 3895, 3917, 3922, 3931, 3944, 3949, 3959, 3962, 3980, 4025, 4037, 4040, 4048, 4066, 4068, 4075, 4077, 4079, 4092, 4097, 4104, 4104, 4112, 4121, 4125, 4143, 4148, 4153, 4161, 4162, 4176, 4213, 4214, 4218, 4247, 4255, 4258, 4258, 4259, 4270, 4276, 4298, 4301, 4302, 4307, 4329, 4336, 4350, 4350, 4351, 4364, 4387, 4392, 4395, 4426, 4431, 4435, 4440, 4447, 4448, 4453, 4456, 4494, 4517, 4524, 4533, 4535, 4548, 4574, 4583, 4583, 4590, 4592, 4600, 4621, 4627, 4632, 4637, 4647, 4653, 4668, 4672, 4675, 4691, 4733, 4760, 4763, 4765, 4772, 4778, 4785, 4787, 4796, 4803, 4804, 4814, 4818, 4829, 4846, 4857, 4911, 4919, 4921, 4937, 4948, 4950, 4950, 4958, 4982};

    //--- randomize array ---
    /*vector<long> array;
    int len = 100;
    int range = 500;
    srand(time(0));
    for (int i=0; i<len; i++) {
        array.push_back(std::rand()%range-range/2);
    }*/

    //--- read zipf generated array ---
    vector<long> array;
    string line;
    ifstream file("/Users/zhang/CLionProjects/bsiCPP/examples/fastzipf_generated_vector");
    while (getline(file, line)) {
        array.push_back(stoi(line.substr(0,line.size()-1)));
    }
    file.close();

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
            //cout << bsi->getValue(i) << " ";
        }
    }
    cout << "array length: " << topkmax_vector.size() << "\n";
    sort(topkmax_vector.begin(),topkmax_vector.end(),greater<long>());

    //--- verify accuracy ---
    int j = 0;
    bool correct = true;
    while (j<topkmax_vector.size()) {
        if (topkmax_vector[j] != array[array.size()-j-1]) {
            correct = false;
            break;
        }
        j++;
    }
    if (correct && topkmax_vector.size() >= k) {
        cout << "\n" << "correct" << "\n";
    } else {
        cout << "\n" << "incorrect" << "\n";
    }

    //--- topKMin ---
    HybridBitmap<uint64_t> topkmin = bsi->topKMin(k);
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
    correct = true;
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
    array.clear();
    return 0;
}
