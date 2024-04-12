#include <iostream>
#include <fstream>
#include <chrono>

#include "BsiUnsigned.hpp"
#include "BsiSigned.hpp"
#include "BsiAttribute.hpp"


//void MultiplyVectorByScalar(vector<long>& v, int k) {
//    transform(v.begin(), v.end(), v.begin(), [k](long& c) { return c * k; });
//}
using namespace std;

void testInverse();
void testCompareTo();

vector<BsiAttribute<uint64_t>*> inv(vector<BsiAttribute<uint64_t>*> matrix);
void sgesv(int n, int m, vector<BsiAttribute<uint64_t>*> a, vector<int> ipiv, vector<BsiAttribute<uint64_t>*> b);
void sgetrf(int m, int n, vector<BsiAttribute<uint64_t>*> a, vector<int> ipiv);
void sgetrf2(int m, int n, vector<BsiAttribute<uint64_t>*> a, vector<int> ipiv);
void sgetrs(int n, int m, vector<BsiAttribute<uint64_t>*> a, vector<int> ipiv, vector<BsiAttribute<uint64_t>*> b);
int main() {
    testCompareTo();
    cout << "done testing compareTo\n";
    return 0;
}

void testInverse() {
    BsiSigned<uint64_t> bsi;
    vector<BsiAttribute<uint64_t>*> mat;
    vector<long> r1 = {4,3};
    vector<long> r2 = {3,2};
    mat.push_back(bsi.buildBsiAttributeFromVectorSigned(r1, 0.5));
    mat.push_back(bsi.buildBsiAttributeFromVectorSigned(r2, 0.5));
    vector<BsiAttribute<uint64_t>*> res = inv(mat);
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
/*
 * LAPACK subroutine for computing the solution to a real system of linear equations
 * A * X = B
 * where A is an n-by-n matrix, and X and B are n-by-m matrices.
 * LU decomposition with partial pivoting and row interchanges is used to factor A as
 * A = P * L * U
 * Matrices are implemented with arrays of signed BsiAttribute's
*/
void sgesv(int n, int m, vector<BsiAttribute<uint64_t>*> a, vector<int> ipiv, vector<BsiAttribute<uint64_t>*> b) {
    // Compute LU factorization of A
    sgetrf(n, n, a, ipiv);
    // Solve the system A * X = B by overwriting B with X
    sgetrs(n, m, a, ipiv, b);
}

/*
 * Auxiliary call to LAPACK subroutine for computing an LU factorization
*/
void sgetrf(int m, int n, vector<BsiAttribute<uint64_t>*> a, vector<int> ipiv) {
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
 * Matrices are implemented with arrays of signed BsiAttribute's
*/
void sgetrf2(int m, int n, vector<BsiAttribute<uint64_t>*> a, vector<int> ipiv) {
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
            if (i != 1) {
                int temp = a[0]->getValue(0);
            }
        }
    }
}

void testCompareTo() {
    BsiSigned<uint64_t> build;
    int n = 5;
    string line;
    ifstream file("/Users/zhang/CLionProjects/bsiCPP/examples/testcase.txt");
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
            BsiAttribute<uint64_t> *bsi1 = build.buildBsiAttributeFromVectorSigned(v1, 0.5);
            BsiAttribute<uint64_t> *bsi2 = build.buildBsiAttributeFromVectorSigned(v2, 0.5);
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

void sgetrs(int n, int m, vector<BsiAttribute<uint64_t>*> a, vector<int> ipiv, vector<BsiAttribute<uint64_t>*> b) {

}