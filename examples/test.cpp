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
void main() {

}

vector<BsiAttribute<uint64_t>*> inv(vector<BsiAttribute<uint64_t>*> mat) {
    int n = mat.size();
    vector<int> ipiv;
    vector<BsiAttribute<uint64_t>*> res;
    for (int i=0; i<n; i++) {
        res.push_back(new BsiSigned<uint64_t>(mat[0].size));
    }
    sgesv(n,n,mat,ipiv,res);
}
void sgesv(int n, int m, vector<BsiAttribute<uint64_t>*> a, vector<int> ipiv, vector<BsiAttribute<uint64_t>*> b) {

}