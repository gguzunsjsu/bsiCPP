#include <iostream>
#include <fstream>
#include <chrono>

#include "BsiUnsigned.hpp"
#include "BsiSigned.hpp"
#include "BsiAttribute.hpp"
#include <sstream>

//void MultiplyVectorByScalar(vector<long>& v, int k) {
//    transform(v.begin(), v.end(), v.begin(), [k](long& c) { return c * k; });
//}
using namespace std;

void testInverse();
void testCompareTo();
void testMultByConstant();
void testMultSum();

vector<BsiAttribute<uint64_t>*> inv(vector<BsiAttribute<uint64_t>*> matrix);
void sgesv(int n, int m, vector<BsiAttribute<uint64_t>*> a, vector<int> ipiv, vector<BsiAttribute<uint64_t>*> b);
void sgetrf(int m, int n, vector<BsiAttribute<uint64_t>*> a, vector<int> ipiv);
void sgetrf2(int m, int n, vector<BsiAttribute<uint64_t>*> a, vector<int> ipiv);
void sgetrs(int n, int m, vector<BsiAttribute<uint64_t>*> a, vector<int> ipiv, vector<BsiAttribute<uint64_t>*> b);
int main() {
    //testMultByConstant();
    //testInverse();
    testMultSum();
    return 0;
}
void testMultSum() {
    BsiSigned<uint64_t> bsi;
    ifstream file("/Users/zhang/CLionProjects/bsiCPP/examples/testcase.txt");
    vector<BsiAttribute<uint64_t>*> H_bsi;
    for (int i=0; i<9; i++) {
        string line;
        getline(file,line);
        stringstream ss(line);
        long num;
        vector<long> H;
        while (ss >> num) {
            H.push_back(num);
        }
        H_bsi.push_back(bsi.buildBsiAttributeFromVectorSigned(H,0.5));
    }
    string line;
    getline(file,line);
    stringstream ss(line);
    long num;
    vector<long> i;
    while (ss >> num) {
        i.push_back(num);
    }
    BsiAttribute<uint64_t>* i_bsi = bsi.buildBsiAttributeFromVectorSigned(i,0.5);
    getline(file,line);
    stringstream ss2(line);
    vector<long> j;
    while (ss2 >> num) {
        j.push_back(num);
    }
    int PRECISION = 1;
    BsiAttribute<uint64_t>* j_bsi = bsi.buildBsiAttributeFromVectorSigned(j,0.5);

    /*BsiAttribute<uint64_t>* u_bsi = H_bsi[0]->multiplyWithBsiHorizontal(j_bsi,PRECISION)->SUM(H_bsi[1]->multiplyWithBsiHorizontal(i_bsi,PRECISION)->SUM(H_bsi[2]));
    BsiAttribute<uint64_t>* v_bsi = H_bsi[3]->multiplyWithBsiHorizontal(j_bsi,PRECISION)->SUM(H_bsi[4]->multiplyWithBsiHorizontal(i_bsi,PRECISION)->SUM(H_bsi[5]));
    BsiAttribute<uint64_t>* w_bsi = H_bsi[6]->multiplyWithBsiHorizontal(j_bsi,PRECISION)->SUM(H_bsi[7]->multiplyWithBsiHorizontal(i_bsi,PRECISION)->SUM(H_bsi[8]));

    for (int k=0; k<u_bsi->rows; k++) {
        long u = H_bsi[0]->getValue(k) * j_bsi->getValue(k) + H_bsi[1]->getValue(k) * i_bsi->getValue(k) + H_bsi[2]->getValue(k);
        long v = H_bsi[3]->getValue(k) * j_bsi->getValue(k) + H_bsi[4]->getValue(k) * i_bsi->getValue(k) + H_bsi[5]->getValue(k);
        long w = H_bsi[6]->getValue(k) * j_bsi->getValue(k) + H_bsi[7]->getValue(k) * i_bsi->getValue(k) + H_bsi[8]->getValue(k);
        cout << "u: " << u_bsi->getValue(k) << " " << u << " v: " << v_bsi->getValue(k) << " " << v << " w: " << w_bsi->getValue(k) << " " << w << "\n";
    }*/
    BsiAttribute<uint64_t>* u_bsi1 = H_bsi[0]->multiplyWithBsiHorizontal(j_bsi,PRECISION);
    BsiAttribute<uint64_t>* u_bsi2 = H_bsi[1]->multiplyWithBsiHorizontal(i_bsi,PRECISION);
    BsiAttribute<uint64_t>* u_bsi3 = u_bsi1->SUM(u_bsi2);
    BsiAttribute<uint64_t>* u_bsi4 = u_bsi3->SUM(H_bsi[2]);
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
    BsiAttribute<uint64_t> *test = bsi.buildBsiAttributeFromVectorSigned(v,0.5);
    for (int i=0; i<v.size(); i++) {
        v[i] *= c;
    }
    BsiAttribute<uint64_t> *sol = bsi.buildBsiAttributeFromVectorSigned(v,0.5);
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
    vector<BsiAttribute<uint64_t>*> mat;
    vector<long> r1 = {4,3};
    vector<long> r2 = {3,2};
    mat.push_back(bsi.buildBsiAttributeFromVectorSigned(r1, 0.5));
    mat.push_back(bsi.buildBsiAttributeFromVectorSigned(r2, 0.5));
    vector<BsiAttribute<uint64_t>*> res = inv(mat);
}

vector<BsiAttribute<uint64_t>*> inv(vector<BsiAttribute<uint64_t>*> mat) {
    int precision = 10000;
    int n = mat.size();
    vector<BsiAttribute<uint64_t>*> res; // initialize as identity matrix
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
        res.push_back(bsi.buildBsiAttributeFromVectorSigned(row,0.5));
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
        BsiAttribute<uint64_t>* temp = mat[i];
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