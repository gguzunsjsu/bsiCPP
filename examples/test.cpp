#include <iostream>
#include <chrono>

#include "BsiUnsigned.hpp"
#include "BsiSigned.hpp"
#include "BsiAttribute.hpp"


//void MultiplyVectorByScalar(vector<long>& v, int k) {
//    transform(v.begin(), v.end(), v.begin(), [k](long& c) { return c * k; });
//}
using namespace std;
vector<BsiAttribute<uint64_t>*> inv(vector<BsiAttribute<uint64_t>*> matrix);
void sgesv(int n, int m, vector<BsiAttribute<uint64_t>*> a, vector<int> ipiv, vector<BsiAttribute<uint64_t>*> b);
int main() {
    BsiSigned<uint64_t> bsi;
    vector<BsiAttribute<uint64_t>*> mat;
    vector<long> r1 = {4,3};
    vector<long> r2 = {3,2};
    mat.push_back(bsi.buildBsiAttributeFromVectorSigned(r1, 0.5));
    mat.push_back(bsi.buildBsiAttributeFromVectorSigned(r2, 0.5));
    vector<BsiAttribute<uint64_t>*> res = inv(mat);
    return 0;
}

vector<BsiAttribute<uint64_t>*> inv(vector<BsiAttribute<uint64_t>*> mat) {
    int n = mat.size();
    vector<int> ipiv;
    vector<BsiAttribute<uint64_t>*> res;
    for (int i=0; i<n; i++) {
        res.push_back(new BsiSigned<uint64_t>(mat.at(0)->getNumberOfSlices()+1));
    }
    sgesv(n,n,mat,ipiv,res);
    return res;
}
void sgesv(int n, int m, vector<BsiAttribute<uint64_t>*> a, vector<int> ipiv, vector<BsiAttribute<uint64_t>*> b) {

}