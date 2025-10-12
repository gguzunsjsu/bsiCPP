#include <iostream>
#include <vector>

#include "../bsi/BsiSigned.hpp"
#include "../bsi/BsiUnsigned.hpp"

int main() {
    std::vector<long> databaseValues{3, 1, 4, 1};
    std::vector<long> queryValues{5, 9, 2, 6};

    BsiUnsigned<uint64_t> builder;
    BsiVector<uint64_t>* database = builder.buildBsiVector(databaseValues, 0.0);
    BsiVector<uint64_t>* query = builder.buildBsiVector(queryValues, 0.0);

    const long long whole_dot_product = database->dot(query);
    int threshold = 5;
    const long long prunedZero = database->dot_with_pruning(query, threshold);
    const long long prunedBeyond = database->dot_with_pruning(query, whole_dot_product + 1);

    std::cout << "dot() = " << whole_dot_product << '\n';
    std::cout << "dot_with_pruning(threshold = "<< threshold << ") = " << prunedZero << '\n';
    std::cout << "dot_with_pruning(threshold = " << (whole_dot_product + 1) << ") = " << prunedBeyond << " (pruned)" << '\n';

    delete database;
    delete query;

    return 0;
}
