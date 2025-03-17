//
// Created by poorna on 3/15/25.
//

#include <iostream>
#include <vector>
#include <cstdint>
#include <stdexcept>

using namespace std;

// --- Helper Functions for Bit-Sliced Representations --- //

// Converts a vector of unsigned numbers (size m) into a bit-sliced representation.
// The result is a vector of 'bitWidth' slices. In each slice (a uint64_t),
// the bit at position j corresponds to the j-th number's bit at that bit position.
vector<uint64_t> vectorToBitSliced(const vector<uint64_t>& numbers, size_t bitWidth) {
    size_t m = numbers.size();
    vector<uint64_t> slices(bitWidth, 0);
    for (size_t j = 0; j < m; j++) {
        for (size_t bit = 0; bit < bitWidth; bit++) {
            if (numbers[j] & (1ULL << bit)) {
                slices[bit] |= (1ULL << j);
            }
        }
    }
    return slices;
}

// Converts a bit-sliced representation (with LSB at index 0)
// into a vector of unsigned numbers (each number is built from the bits
// in the corresponding positions across slices). 'm' is the number of lanes.
vector<uint64_t> bitSlicedToVector(const vector<uint64_t>& slices, size_t m) {
    size_t bitWidth = slices.size();
    vector<uint64_t> numbers(m, 0);
    for (size_t j = 0; j < m; j++) {
        for (size_t bit = 0; bit < bitWidth; bit++) {
            if (slices[bit] & (1ULL << j)) {
                numbers[j] |= (1ULL << bit);
            }
        }
    }
    return numbers;
}

// Performs a left shift on a bit-sliced number.
// The bit-sliced number is represented as a vector of slices (LSB at index 0).
// Shifting left by one means that for each slice from high to low,
// X[i] = X[i-1] and X[0] becomes 0.
void bitSlicedLeftShift(vector<uint64_t>& slices) {
    if (slices.empty()) return;
    for (size_t i = slices.size() - 1; i > 0; i--) {
        slices[i] = slices[i - 1];
    }
    slices[0] = 0;
}

// Performs bit-sliced subtraction of two numbers A - B,
// where both A and B are represented in bit-sliced form (vector of slices).
// Each slice contains m lanes packed in a 64-bit word. The parameter fullMask
// is used to mask out unused bits (only the lower m bits are valid).
// Borrow propagation is performed per lane.
vector<uint64_t> bitSlicedSubtract(const vector<uint64_t>& A, const vector<uint64_t>& B, uint64_t fullMask) {
    if (A.size() != B.size())
        throw invalid_argument("Bit-sliced numbers must have the same width for subtraction.");
    size_t L = A.size();
    vector<uint64_t> result(L, 0);
    uint64_t borrow = 0; // borrow for each lane (packed in a 64-bit word)
    for (size_t i = 0; i < L; i++) {
        uint64_t a = A[i] & fullMask; // use all lane bits in this slice
        uint64_t b = B[i] & fullMask;
        // Compute difference per lane: diff = a - b - borrow (mod 2)
        uint64_t diff = (a ^ b ^ borrow) & fullMask;
        // Compute new borrow per lane:
        uint64_t newBorrow = (((~a) & (b | borrow)) | (b & borrow)) & fullMask;
        result[i] = diff;
        borrow = newBorrow;
    }
    return result;
}

// --- Bit-Sliced Division for a Vector of Numbers --- //

// This function performs element-wise (parallel) long division
// on a vector of numbers using bit-sliced processing.
// 'dividends' and 'divisors' must have the same size (m <= 64).
// 'bitWidth' is the fixed width for representing numbers (e.g., 8 or 32).
// The function returns quotient and remainder vectors.
void bitSlicedDivisionVector(const vector<uint64_t>& dividends,
                             const vector<uint64_t>& divisors,
                             size_t bitWidth,
                             vector<uint64_t>& quotients,
                             vector<uint64_t>& remainders) {
    size_t m = dividends.size();
    if (m == 0 || m != divisors.size())
        throw invalid_argument("Dividend and divisor vectors must be nonempty and of equal size.");

    // Convert dividends and divisors into bit-sliced representations.
    vector<uint64_t> dividendSlices = vectorToBitSliced(dividends, bitWidth);
    vector<uint64_t> divisorSlices = vectorToBitSliced(divisors, bitWidth);

    // Check for division-by-zero in any lane.
    vector<uint64_t> divisorValues = bitSlicedToVector(divisorSlices, m);
    for (size_t i = 0; i < m; i++) {
        if (divisorValues[i] == 0)
            throw invalid_argument("Division by zero encountered for at least one element.");
    }

    // fullMask: a mask with lower m bits set.
    uint64_t fullMask = (m == 64) ? ~0ULL : ((1ULL << m) - 1);

    // Extend the divisor to have bitWidth+1 slices (by adding a zero slice at the top).
    vector<uint64_t> extendedDivisor = divisorSlices;
    extendedDivisor.push_back(0);

    // Create remainder R with width = bitWidth + 1, initialized to all zeros.
    vector<uint64_t> R(bitWidth + 1, 0);

    // The quotient will have bitWidth slices.
    vector<uint64_t> Q(bitWidth, 0);

    // Process each bit from MSB to LSB.
    // The dividendSlices vector has indices 0..(bitWidth-1) with bitWidth-1 as the MSB.
    for (int i = bitWidth - 1; i >= 0; i--) {
        // Left shift R by 1.
        bitSlicedLeftShift(R);
        // Bring down the i-th slice from the dividend.
        // (R[0] is zero after shifting, so OR in the dividend slice.)
        R[0] |= dividendSlices[i];
        // Compute tentative remainder: temp = R - extendedDivisor.
        vector<uint64_t> temp = bitSlicedSubtract(R, extendedDivisor, fullMask);
        // The sign of the result is in the most significant slice: index bitWidth.
        // For each lane, if that bit is 0 then temp is nonnegative.
        uint64_t mask = (~temp[bitWidth]) & fullMask;  // lane bit = 1 if nonnegative.
        // Set the quotient bit at position i to this mask.
        Q[i] = mask;
        // For each slice j, update R: if mask==1 for a lane, use temp; else leave R unchanged.
        for (size_t j = 0; j < R.size(); j++) {
            R[j] = (mask & temp[j]) | ((~mask) & R[j]);
        }
    }

    // Convert the quotient from bit-sliced representation (Q) back to a vector.
    quotients = bitSlicedToVector(Q, m);
    // For the remainder, use only the lower bitWidth slices of R.
    vector<uint64_t> R_lower(R.begin(), R.begin() + bitWidth);
    remainders = bitSlicedToVector(R_lower, m);
}

int main() {
    // Example vector (element-wise division):
    // Dividends: 13, 27, 100
    // Divisors:   3,  5,   7
    vector<uint64_t> dividends = {13, 27, 100};
    vector<uint64_t> divisors  = {3, 5, 7};

    // Choose a bit width sufficient to represent these numbers.
    // Here, 8 bits (0 to 255) is sufficient.
    size_t bitWidth = 8;

    vector<uint64_t> quotients;
    vector<uint64_t> remainders;

    try {
        bitSlicedDivisionVector(dividends, divisors, bitWidth, quotients, remainders);
    } catch (const exception& ex) {
        cerr << "Error: " << ex.what() << endl;
        return 1;
    }

    // Print the results.
    cout << "Element-wise division using bit-sliced long division:" << endl;
    cout << "-----------------------------------------------------" << endl;
    for (size_t i = 0; i < dividends.size(); i++) {
        cout << "Dividend: " << dividends[i]
             << "  Divisor: " << divisors[i]
             << "  Quotient: " << quotients[i]
             << "  Remainder: " << remainders[i] << endl;
    }
    return 0;
}






//#include <iostream>
//#include <vector>
//#include <cstdint>
//#include <stdexcept>
//
//using namespace std;
//
//// A class that represents a number in bit–sliced form.
//// Each element in 'slices' holds a single bit (in its LSB) of the number.
//// The vector is organized so that index 0 is the least significant bit (LSB)
//// and index (n-1) is the most significant bit.
//class BitSlicedNumber {
//public:
//    vector<uint64_t> slices; // only the LSB of each uint64_t is used
//
//    // Construct with a given number of slices (bits), initializing to 0.
//    BitSlicedNumber(size_t n) : slices(n, 0) {}
//
//    // Construct from an existing vector.
//    BitSlicedNumber(const vector<uint64_t>& s) : slices(s) {}
//
//    size_t width() const { return slices.size(); }
//
//    // Left–shift by one bit (i.e. multiply by 2) by shifting the slices.
//    // For each i from (width()-1) down to 1, we set slices[i] = slices[i-1] and then set slices[0] to 0.
//    void leftShift() {
//        if (slices.empty()) return;
//        for (size_t i = slices.size() - 1; i > 0; i--) {
//            slices[i] = slices[i - 1];
//        }
//        slices[0] = 0;
//    }
//
//    // Set the least significant slice to the given value (should be 0 or 1).
//    void setLSB(uint64_t value) {
//        slices[0] = (value & 1ULL);
//    }
//
//    // Subtract another BitSlicedNumber (assumed to have the same width) from this one.
//    // This performs bit–sliced subtraction on a per–bit (per slice) basis,
//    // propagating a borrow (each borrow is stored in the LSB of a uint64_t).
//    BitSlicedNumber subtract(const BitSlicedNumber& other) const {
//        if (width() != other.width())
//            throw invalid_argument("Widths must match for subtraction.");
//        size_t n = width();
//        vector<uint64_t> result(n, 0);
//        uint64_t borrow = 0; // only LSB is used
//        for (size_t i = 0; i < n; i++) {
//            uint64_t a = slices[i] & 1ULL;
//            uint64_t b = other.slices[i] & 1ULL;
//            int temp = int(a) - int(b) - int(borrow);
//            uint64_t diff, newBorrow;
//            if (temp < 0) {
//                diff = (temp + 2) & 1ULL; // modulo 2
//                newBorrow = 1;
//            } else {
//                diff = temp & 1ULL;
//                newBorrow = 0;
//            }
//            result[i] = diff;
//            borrow = newBorrow;
//        }
//        return BitSlicedNumber(result);
//    }
//
//    // Check if the number is nonnegative.
//    // In two's complement the sign is in the MSB (last slice).
//    // Returns 1 if nonnegative, 0 if negative.
//    // (This is used only for the algorithm's internal decision-making.)
//    uint64_t isNonNegative() const {
//        return ((slices.back() & 1ULL) == 0) ? 1ULL : 0ULL;
//    }
//
//    // Convert the bit–sliced number (using only the LSBs of the slices)
//    // to an unsigned integer.
//    uint64_t toUnsignedInt() const {
//        uint64_t value = 0;
//        size_t n = width();
//        for (size_t i = 0; i < n; i++) {
//            value |= ((slices[i] & 1ULL) << i);
//        }
//        return value;
//    }
//
//    // Print the binary representation (from MSB to LSB).
//    void printBinary() const {
//        for (size_t i = slices.size(); i > 0; i--) {
//            cout << (slices[i - 1] & 1ULL);
//        }
//    }
//};
//
//// This function implements bit-sliced division.
//// 'dividendSlices' and 'divisorSlices' hold the bit-sliced representation of the dividend and divisor,
//// with the LSB at index 0.
//// The function computes the quotient (stored in 'quotientSlices') and remainder (in 'remainderSlices').
//// The algorithm processes the bits from the most significant down to the least significant.
//void bitSlicedDivision(const vector<uint64_t>& dividendSlices, const vector<uint64_t>& divisorSlices,
//                       vector<uint64_t>& quotientSlices, vector<uint64_t>& remainderSlices) {
//    if (dividendSlices.empty() || divisorSlices.empty())
//        throw invalid_argument("Empty dividend or divisor");
//
//    // Check for division by zero.
//    bool divisorIsZero = true;
//    for (auto val : divisorSlices) {
//        if (val & 1ULL) { // if any bit is 1, divisor is not zero
//            divisorIsZero = false;
//            break;
//        }
//    }
//    if (divisorIsZero)
//        throw invalid_argument("Division by zero");
//
//    size_t p = dividendSlices.size(); // number of bits in dividend
//    size_t q = divisorSlices.size();  // number of bits in divisor
//
//    // We use a remainder R with width = p + 1 bits.
//    BitSlicedNumber R(p + 1);
//    BitSlicedNumber dividend(dividendSlices);
//
//    // Extend divisor to width p + 1 (by adding zeros at the MSB).
//    vector<uint64_t> extendedDivisor = divisorSlices;
//    while (extendedDivisor.size() < p + 1) {
//        extendedDivisor.push_back(0);
//    }
//    BitSlicedNumber divisor(extendedDivisor);
//
//    // The quotient will have p bits.
//    quotientSlices.assign(p, 0);
//
//    // Process each bit from MSB to LSB.
//    for (int i = p - 1; i >= 0; i--) {
//        // Left shift R by 1.
//        R.leftShift();
//        // Bring down the i-th bit of the dividend.
//        R.setLSB(dividend.slices[i]);
//        // Compute tentative remainder: temp = R - divisor.
//        BitSlicedNumber temp = R.subtract(divisor);
//        // Use the sign (MSB of temp) to decide:
//        // if temp is nonnegative (i.e. sign bit = 0), then the subtraction is valid.
//        uint64_t mask = temp.isNonNegative(); // 1 if nonnegative, 0 if negative.
//        // Set the quotient bit at position i to mask.
//        quotientSlices[i] = mask; // 0 or 1.
//        // If the result is nonnegative, update R to temp.
//        if (mask == 1ULL) {
//            R = temp;
//        }
//    }
//
//    // Output the remainder (R has width = p + 1 bits).
//    remainderSlices = R.slices;
//}
//
//int main() {
//    // Example: Divide 13 by 3.
//    // To represent 13 correctly as an unsigned 4-bit number, note that 4 bits can represent 0 to 15.
//    // 13 in binary is 1101. Represented as a vector with LSB at index 0, we have: [1, 0, 1, 1].
//    vector<uint64_t> dividendSlices = {1, 0, 1, 1};
//    // 3 in binary is 11, so represented with 2 bits: [1, 1].
//    vector<uint64_t> divisorSlices = {1, 1};
//
//    vector<uint64_t> quotientSlices;
//    vector<uint64_t> remainderSlices;
//
//    try {
//        bitSlicedDivision(dividendSlices, divisorSlices, quotientSlices, remainderSlices);
//    } catch (const exception& ex) {
//        cerr << "Error: " << ex.what() << endl;
//        return 1;
//    }
//
//    // Create BitSlicedNumber objects for printing.
//    BitSlicedNumber dividend(dividendSlices);
//    BitSlicedNumber divisor(divisorSlices);
//    BitSlicedNumber quotient(quotientSlices);
//    BitSlicedNumber remainder(remainderSlices);
//
//    cout << "Dividend: ";
//    dividend.printBinary();
//    cout << " (unsigned decimal " << dividend.toUnsignedInt() << ")" << endl;
//
//    cout << "Divisor:  ";
//    divisor.printBinary();
//    cout << " (unsigned decimal " << divisor.toUnsignedInt() << ")" << endl;
//
//    cout << "Quotient: ";
//    quotient.printBinary();
//    cout << " (unsigned decimal " << quotient.toUnsignedInt() << ")" << endl;
//
//    // For the remainder, we use only the lower 'p' bits.
//    uint64_t remVal = 0;
//    for (size_t i = 0; i < dividend.width() && i < remainder.width(); i++) {
//        remVal |= ((remainder.slices[i] & 1ULL) << i);
//    }
//    cout << "Remainder: ";
//    // Print only the lowest dividend.width() bits.
//    for (size_t i = dividend.width(); i > 0; i--) {
//        cout << (remainder.slices[i - 1] & 1ULL);
//    }
//    cout << " (unsigned decimal " << remVal << ")" << endl;
//
//    // For 13 / 3, we expect quotient = 4 and remainder = 1.
//    return 0;
//}
