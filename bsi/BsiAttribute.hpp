//
//  BsiAttribute.hpp
//

#ifndef BsiAttribute_hpp
#define BsiAttribute_hpp

#include <stdio.h>
#include <iostream>
#include <cmath>
#include "hybridBitmap/hybridbitmap.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <bitset>
#include <algorithm>
#include <thread>
#include <random>


template <class uword = uint64_t> class BsiUnsigned;
template <class uword = uint64_t> class BsiSigned;
//template <class uword = uint32_t>
template <class uword = uint64_t> class BsiAttribute{
public:
    int size; //holds number of slices
    int offset =0;
    int decimals = 0;
    int bits = 8*sizeof(uword);

    std::vector<HybridBitmap<uword> > bsi ;
    HybridBitmap<uword> existenceBitmap;
    HybridBitmap<uword> sign; // sign bitslice

    long rows;  // number of elements
    long index; // if split horizontally

    bool is_signed; // sign flag

    bool firstSlice; //contains first slice
    bool lastSlice; //contains last slice
    
    bool twosComplement;
    
    /*
    * Member functions
    */
        
    bool isLastSlice() const;
    void setLastSliceFlag(bool flag);
    
    bool isFirstSlice() const;
    void setFirstSliceFlag(bool flag);

    bool isSigned()const;    

    void setNumberOfSlices(int s);
    int getNumberOfSlices() const;

    void addSlice(const HybridBitmap<uword>& slice);
    HybridBitmap<uword> getSlice(int i) const;

    int getOffset() const;
    void setOffset(int offset);  
    
    long getNumberOfRows() const;   
    void setNumberOfRows(long rows);

    long getPartitionID() const;
    void setPartitionID(long index);

    HybridBitmap<uword> getExistenceBitmap();
    void setExistenceBitmap(const HybridBitmap<uword> &exBitmap);

    void setTwosFlag(bool flag);
    /*
    * ---------------------- Declarations for getters and setters end---------------------------
    */

    /*
    * ---------------------- Declarations for member functions pertaining vector operations -----------------
    */
    virtual HybridBitmap<uword> topKMax(int k)=0;
    virtual HybridBitmap<uword> topKMin(int k)=0;

    virtual BsiAttribute* SUM(BsiAttribute* a)=0;
    virtual BsiAttribute* SUMsigned(BsiAttribute* a)=0;
    virtual BsiAttribute* SUM(long a)const=0;
    virtual BsiAttribute<uword>* sum_Horizontal(const BsiAttribute<uword> *a) const=0;

    virtual BsiAttribute* convertToTwos(int bits)=0;

    virtual BsiUnsigned<uword>* abs()=0;
    virtual BsiUnsigned<uword>* abs(int resultSlices,const HybridBitmap<uword> &EB)=0;
    virtual BsiUnsigned<uword>* absScale(double range)=0;
    virtual int compareTo(BsiAttribute<uword> *a, int index)=0;

    virtual long getValue(int pos)const=0;

    virtual HybridBitmap<uword> rangeBetween(long lowerBound, long upperBound)=0;
    virtual BsiAttribute<uword>* multiplyByConstantNew(int number)const=0;
    virtual BsiAttribute<uword>* multiplyByConstant(int number)const=0;
    virtual BsiAttribute<uword>* multiplication(BsiAttribute<uword> *a)const=0;
    virtual BsiAttribute<uword>* multiplication_array(BsiAttribute<uword> *a)const=0;
    virtual BsiAttribute<uword>* multiplyBSI(BsiAttribute<uword> *a) const=0;
    virtual BsiAttribute<uword>*  multiplyWithBsiHorizontal(const BsiAttribute<uword> *unbsi, int precision) const=0;
    virtual BsiAttribute<uword>*  multiplyWithBsiHorizontal(const BsiAttribute<uword> *unbsi) const=0;
    virtual BsiAttribute<uword>* multiplication_Horizontal(const BsiAttribute<uword> *a) const=0;
    virtual BsiAttribute<uword>* multiplication_Horizontal_Hybrid(const BsiAttribute<uword> *a) const=0;
    virtual BsiAttribute<uword>* multiplication_Horizontal_Verbatim(const BsiAttribute<uword> *a) const=0;
    virtual BsiAttribute<uword>* multiplication_Horizontal_compressed(const BsiAttribute<uword> *a) const=0;
    virtual BsiAttribute<uword>* multiplication_Horizontal_Hybrid_other(const BsiAttribute<uword> *a) const=0;
    virtual long dotProduct(BsiAttribute<uword>* a) const = 0;
    virtual long long int dot(BsiAttribute<uword>* a) const = 0;
    virtual long long int dot_withoutCompression(BsiAttribute<uword>* a) const = 0;
    virtual void multiplicationInPlace(BsiAttribute<uword> *a)=0;

    /*
     * division implementation
     */
    virtual std::pair<BsiAttribute<uword>*, BsiAttribute<uword>*> divide(
            const BsiAttribute<uword>& dividend,
            const BsiAttribute<uword>& divisor) const = 0;


    virtual BsiAttribute<uword>* negate()=0;

    virtual long sumOfBsi()const=0;

    virtual bool append(long value)=0;
    
    BsiAttribute* buildQueryAttribute(long query, int rows, long partitionID);
    BsiAttribute* buildBsiAttributeFromArray(std::vector<uword> &array, int attRows, double compressThreshold);
    BsiAttribute* buildBsiAttributeFromArray(uword array[], long max, long min, long firstRowID, double compressThreshold);
    BsiAttribute<uword>* buildBsiAttributeFromVector(std::vector<long> nums, double compressThreshold)const;
    BsiAttribute<uword>* createRandomBsi(int vectorLength, int range, double compressThreshold) const;
    BsiAttribute<uword>* buildBsiAttributeFromVector_without_compression(std::vector<long> nums) const;
    BsiAttribute<uword>* buildBsiAttributeFromVectorSigned(std::vector<long> nums, double compressThreshold)const;
    //BsiAttribute<uword>* buildBsiAttributeFromPyList(py::list nums, double compressThreshold)const;
    BsiAttribute<uword>* buildCompressedBsiFromVector(std::vector<long> nums, double compressThreshold) const;
    BsiAttribute<uword> *
    buildBsiVector(std::vector<long> decimalVector, int vectorLength, long min, long max, long firstRowID,
                   double compressThreshold) const;
    
    HybridBitmap<uword> maj(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const;

    HybridBitmap<uword> XOR(const HybridBitmap<uword> &a,const HybridBitmap<uword> &b,const HybridBitmap<uword> &c)const;
    HybridBitmap<uword> orAndNot(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const;
    HybridBitmap<uword> orAnd(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const;
    HybridBitmap<uword> And(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const;
   
    BsiAttribute* signMagnToTwos(int bits);
    BsiAttribute* TwosToSignMagnitue();    
    void signMagnitudeToTwos(int bits);

    void addOneSliceSameOffset(const HybridBitmap<uword> &slice);
    void addOneSliceDiscardCarry(const HybridBitmap<uword> &slice);
    void addOneSliceNoSignExt(const HybridBitmap<uword> &slice);
    void applyExsistenceBitmap(const HybridBitmap<uword> &ex);    
    virtual ~BsiAttribute();
//    size_t getSizeInMemory() const {
//        size_t size_in_memory = sizeof(*this);
//
//        // Add the size of dynamically allocated vectors (assuming they are vectors)
//        size_in_memory += bsi.size() * sizeof(HybridBitmap<>);  // adjust as needed
//        size_in_memory += sizeof(existenceBitmap);  // adjust as needed
//        size_in_memory += sizeof(sign);  // adjust as needed
//
//        return size_in_memory;
//    }
    size_t getSizeInMemory() const{
        size_t total = sizeof(*this);
//        total += sign.getSizeInMemory();
//        total += existenceBitmap.getSizeInMemory();
        for(const auto &slice: bsi){
            total += slice.getSizeInMemory();
        }
        return total;
    }

    /*
     * calculating the bits used to store the slices
     */
    template<typename T>
    size_t getBitsUsedBSI(T max) const{
        if (max==0){
            return 1;
        }
        else{
            return static_cast<size_t>(std::ceil(std::log2(max+1)));
        }
    }


    /*
    * ------------------------Decalrations for private helper methods------------------------------
    */
private:
    void bringTheBitsHelper(const std::vector<long> &array, int slice, int numberOfElements, std::vector<std::vector<uword>> &bitmapDataRaw) const;
    std::vector< std::vector< uword > > bringTheBits(const std::vector<long> &array, int slices, int attRows) const;
    std::vector< std::vector< uword > > bringTheBits(const std::vector<uword> &array, int slices, int attRows) const;
protected:
    int sliceLengthFinder(uword value)const;



};

/*
* ------------------------------------ Function implementations ---------------------------------------
* Provided for all getters and setters + the following functions
* 
* buildQueryAttribute
* sliceLengthFinder
* bringTheBits
* buildCompressedBsiFromVector
* buildBsiAttributeFromVector
*
* maj
  XOR
  orAndNot
  orAnd
  And

* signMagnitudeToTwos
  signMagnToTwos
  TwosToSignMagnitue

* addOneSliceSameOffset
  addOneSliceDiscardCarry
  addOneSliceNoSignExt
  applyExsistenceBitmap
*/

/*
 * Destructor
 */
template <class uword>
BsiAttribute<uword>::~BsiAttribute(){    
};


template <class uword>
bool BsiAttribute<uword>::isLastSlice() const{
    return lastSlice;
};

/*
 *
 * @param flag if the attribute contains the most significant slice then set it to true. Otherwise false.
 */
template <class uword>
void BsiAttribute<uword>::setLastSliceFlag(bool flag){
    lastSlice=flag;
};
/*
 *
 * @return if this attribute contains the first slice(least significant). For internal purposes (when splitting into sub attributes)
 */
template <class uword>
bool BsiAttribute<uword>::isFirstSlice()const{
    return firstSlice;
};
/*
 *
 * @param flag if the attribute contains the least significant slice then set it to true. Otherwise false.
 */
template <class uword>
void BsiAttribute<uword>::setFirstSliceFlag(bool flag){
    firstSlice=flag;
};

/*
 * Returns false if contains only positive numbers
 */
template <class uword>
bool BsiAttribute<uword>::isSigned()const{
    return is_signed;
};

template <class uword>
void BsiAttribute<uword>::addSlice( const HybridBitmap<uword> &slice){
    bsi.push_back(slice);
    size++;
};

/*
 * Don't use for already buily bsi
 */

template <class uword>
void BsiAttribute<uword>::setNumberOfSlices(int s){
    size = s;
}

/**
 * Returns the size of the bsi (how many slices are non zeros) ----> Really ? This just returns the number of slices, right ?
 */
template <class uword>
int BsiAttribute<uword>::getNumberOfSlices()const{
    return bsi.size();
}

/**
 * Returns the slice number i
 */
template <class uword>
HybridBitmap<uword> BsiAttribute<uword>::getSlice(int i) const{
    return bsi[i];
}


/**
 * Returns the offset of the bsi (the first "offset" slices are zero, thus not encoding)
 */
template <class uword>
int BsiAttribute<uword>::getOffset() const{
    return offset;
}


/**
 * Sets the offset of the bsi (the first "offset" slices are zero, thus not encoding)
 */

template <class uword>
void BsiAttribute<uword>::setOffset(int offset){
    BsiAttribute::offset=offset;
}

/**
 * Returns the number of rows for this attribute
 */

template <class uword>
long BsiAttribute<uword>::getNumberOfRows() const{
    return rows;
}


/**
 * Sets the number of rows for this attribute
 */
template <class uword>
void BsiAttribute<uword>::setNumberOfRows(long rows){
    BsiAttribute::rows=rows;
}

/**
 * Returns the index(partition id if horizontally partitioned) for this attribute
 */
template <class uword>
long BsiAttribute<uword>::getPartitionID() const{
    return index;
}

/**
 * Sets the index(partition id if horizontally partitioned) for this attribute
 */
template <class uword>
void BsiAttribute<uword>::setPartitionID(long index){
    BsiAttribute::index=index;
}

/**
 * Returns the Existence bitmap of the bsi attribute
 */
template <class uword>
HybridBitmap<uword> BsiAttribute<uword>::getExistenceBitmap(){
    return existenceBitmap;
}

/**
 * Sets the existence bitmap of the bsi attribute
 */
template <class uword>
void BsiAttribute<uword>::setExistenceBitmap(const HybridBitmap<uword> &exBitmap){
    BsiAttribute::existenceBitmap=exBitmap;
}

/**
 * flag is true when bsi contain data into two's complement form
 */
template <class uword>
void BsiAttribute<uword>::setTwosFlag(bool flag){
    twosComplement=flag;
}

/**
 * builds a BSI attribute with all rows identical given one number (row)
 * @param query
 * @param rows
 * @return the BSI attribute with all rows identical
 */

template <class uword>
BsiAttribute<uword>* BsiAttribute<uword>::buildQueryAttribute(long query, int rows, long partitionID){
    if(query<0){
        uword q = std::abs(query);
        int maxsize = sliceLengthFinder(q);
        BsiAttribute* res = new BsiUnsigned<uword>(maxsize);
        res->setPartitionID(partitionID);
        for(int i=0; i<=maxsize; i++){
            bool currentBit = (q&(1<<i))!=0;
            HybridBitmap<uword> slice;
            slice.setSizeInBits(rows, currentBit);
            if(currentBit){
                slice.density = 1;
            }
            res->addSlice(slice);
        }
        res->setNumberOfRows(rows);
        res->existenceBitmap.setSizeInBits(rows);
        res->existenceBitmap.density=1;
        res->lastSlice=true;
        res->firstSlice=true;
        res->twosComplement=false;
        res->is_signed = true;
        HybridBitmap<uword> temp_sign(true,rows);
        res->sign = temp_sign.Not();//set the sign bits true
        return res;
    }
    else{
        int maxsize = sliceLengthFinder(query);
        BsiAttribute* res = new BsiUnsigned<uword>(maxsize);
        res->setPartitionID(partitionID);
        for(int i=0; i<=maxsize; i++){
            bool currentBit = (query&(1<<i))!=0;
            HybridBitmap<uword> slice;
            slice.setSizeInBits(rows, currentBit);
            if(currentBit){
                slice.density = 1;
            }
            res->addSlice(slice);
        }
        res->setNumberOfRows(rows);
        res->existenceBitmap.setSizeInBits(rows,true);
        res->existenceBitmap.density=1;
        res->lastSlice=true;
        res->firstSlice=true;
        return res;
    }
};

/*
 *
 * sliceLengthFinder find required slices for storing value
 */
template <class uword>
int BsiAttribute<uword>::sliceLengthFinder(uword value) const{
    int lengthCounter =0;
    for(int i = 0; i < bits; i++)
    {
        //uword ai = (static_cast<uword>(1) << i);
        if( ( value & (static_cast<uword>(1) << i ) ) != 0 ){
            lengthCounter = i+1;
        }
    }
    return lengthCounter;
}

/*
 * Used for converting vector into BSI
 * @param compressThreshold determined wether to compress the bit vetor or not
 */

/*
* Creates a bsi-vector from an array
*/
template <class uword>
BsiAttribute<uword>* BsiAttribute<uword>::buildBsiVector(std::vector<long> decimalVector, int vectorLength, long minVal, long maxVal, long firstRowID, double compressThreshold) const {
    int numSlices =  sliceLengthFinder(std::max(std::abs(minVal),std::abs(maxVal))); //number of slices to encode the vector (number of bits for the highest value)

    if(minVal<0){
        BsiSigned<uword>* res = new BsiSigned<uword>(numSlices+1, vectorLength, firstRowID);
        std::vector< std::vector< uword > > bitSlices = bringTheBits(decimalVector,numSlices,vectorLength);
        for(int i=0; i<(numSlices+1); i++){

        }

    }



}


/*
* This is the function being used to create BSI attributes from Vector in our tests
*/
template <class uword>
BsiAttribute<uword>* BsiAttribute<uword>::buildBsiAttributeFromVector(std::vector<long> nums, double compressThreshold) const{
    uword max = std::numeric_limits<uword>::min();
    /*
    * 
    bits = 8*sizeof(uword);
    If we declare a  BsiAttribute<uint64_t> variable,unsigned long long
    each element in the vector can fit in a 64 bit word
    Therefore bits = 64
    How many such words are needed to represent the sign and non-zero property of each element ?
    If one element is represented by one bit of the sign word, the number of words needed = number of elements/number of bits per word.
    */
    
    int numberOfElements = nums.size();
    std::vector<uword> signBits(numberOfElements/(bits)+1);
    std::vector<uword> existBits(numberOfElements/(bits)+1); // keep track for non-zero values
    int countOnes =0;
    int CountZeros = 0;
    const uword one = 1;
    //int bits = 8*sizeof(uword);
    //find max, min, and zeros.
    //Setting sign bits and existence bits for the array of numbers 
    for (int i=0; i<nums.size(); i++){
        int offset = i%(bits);
        if(nums[i] < 0){
            nums[i] = 0 - nums[i];
            signBits[i / (bits)] |= (one << offset); // seting sign bit
            countOnes++;
        }
        if(nums[i] != 0){
            existBits[i / (bits)] |= (one << offset); // seting one at position
        }else{
            CountZeros++;
        }
        if(nums[i] > max){
            max = nums[i];
        }
    }
    //Finding the maximum length of the bit representation of the numbers
    int slices = sliceLengthFinder(max);
    //finding bits used in bsi to store values
//    size_t bits_used = getBitsUsedBSI(max);
//    std::cout << "Bits used by bsi: " << bits_used << std::endl;
    BsiUnsigned<uword>* res = new BsiUnsigned<uword>(slices+1);
    res->sign.reset();
    res->sign.verbatim = true;
    
    for (typename std::vector<uword>::iterator it=signBits.begin(); it != signBits.end(); it++){
        res->sign.addVerbatim(*it,numberOfElements);
    }
    res->sign.setSizeInBits(numberOfElements);
    res->sign.density = countOnes/(double)numberOfElements;
    
    double existBitDensity = (CountZeros/(double)nums.size()); // to decide whether to compress or not
    double existCompressRatio = 1-pow((1-existBitDensity), (2*bits))-pow(existBitDensity, (2*bits));
    if(existCompressRatio >= compressThreshold){
        HybridBitmap<uword> bitmap;
        for(int j=0; j<existBits.size(); j++){
            bitmap.addWord(existBits[j]);
        }
        //bitmap.setSizeInBits(numberOfElements);
        bitmap.density=existBitDensity;
        res->setExistenceBitmap(bitmap);
    }else{
        HybridBitmap<uword> bitmap(true,existBits.size());
        for(int j=0; j<existBits.size(); j++){
            bitmap.buffer[j] = existBits[j];
        }
        //bitmap.setSizeInBits(numberOfElements);
        bitmap.density=existBitDensity;
        res->setExistenceBitmap(bitmap);
    }
    
    //The method to put the elements in the input vector nums to the bsi property of BSIAttribute result
    std::vector< std::vector< uword > > bitSlices = bringTheBits(nums,slices,numberOfElements);
    
    for(int i=0; i<slices; i++){
        double bitDensity = bitSlices[i][0]/(double)numberOfElements; // the bit density for this slice
        double compressRatio = 1-pow((1-bitDensity), (2*bits))-pow(bitDensity, (2*bits));
        if(compressRatio<compressThreshold && compressRatio!=0 ){
            //build compressed bitmap
            HybridBitmap<uword> bitmap;
            for(int j=1; j<bitSlices[i].size(); j++){
                bitmap.addWord(bitSlices[i][j]);
            }
            //bitmap.setSizeInBits(numberOfElements);
            bitmap.density=bitDensity;
            res->addSlice(bitmap);

        }else{
            //build verbatim Bitmap
            HybridBitmap<uword> bitmap(true);
            bitmap.reset();
            bitmap.verbatim = true;
            //                std::copy(bitSlices[i].begin(), bitSlices[i].end(), bitmap.buffer.begin());
            for (typename std::vector<uword>::iterator it=bitSlices[i].begin()+1; it != bitSlices[i].end(); it++){
                bitmap.addVerbatim(*it,numberOfElements);
            }
            // bitmap.buffer=Arrays.copyOfRange(bitSlices[i], 1, bitSlices[i].length);
            //bitmap.actualsizeinwords=bitSlices[i].length-1;
            bitmap.setSizeInBits(numberOfElements);
            bitmap.density=bitDensity;
            res->addSlice(bitmap);
            
        }
    }
    res->existenceBitmap.setSizeInBits(numberOfElements,true);
    res->existenceBitmap.density=1;
    res->lastSlice=true;
    res->firstSlice=true;
    res->twosComplement = false;
    res->rows = numberOfElements;
    res->is_signed = true;
    return res;
};

/*
Build bsi attribute without compression
*/
template <class uword>
BsiAttribute<uword>* BsiAttribute<uword>::buildBsiAttributeFromVector_without_compression(std::vector<long> nums) const{
    uword max = std::numeric_limits<uword>::min();
    /*
    * 
    bits = 8*sizeof(uword);
    If we declare a  BsiAttribute<uint64_t> variable,unsigned long long
    each element in the vector can fit in a 64 bit word
    Therefore bits = 64
    How many such words are needed to represent the sign and non-zero property of each element ?
    If one element is represented by one bit of the sign word, the number of words needed = number of elements/number of bits per word.
    */
    
    int numberOfElements = nums.size();
    std::vector<uword> signBits(numberOfElements/(bits)+1);
    std::vector<uword> existBits(numberOfElements/(bits)+1); // keep track for non-zero values
    int countOnes =0;
    int CountZeros = 0;
    const uword one = 1;
    //int bits = 8*sizeof(uword);
    //find max, min, and zeros.
    //Setting sign bits and existence bits for the array of numbers 
    for (int i=0; i<nums.size(); i++){
        int offset = i%(bits);
        if(nums[i] < 0){
            nums[i] = 0 - nums[i];
            signBits[i / (bits)] |= (one << offset); // seting sign bit
            countOnes++;
        }
        if(nums[i] != 0){
            existBits[i / (bits)] |= (one << offset); // seting one at position
        }else{
            CountZeros++;
        }
        if(nums[i] > max){
            max = nums[i];
        }
    }
    //Finding the maximum length of the bit representation of the numbers
    int slices = sliceLengthFinder(max);
    BsiUnsigned<uword>* res = new BsiUnsigned<uword>(slices+1);
    res->sign.reset();
    res->sign.verbatim = true;
    
    for (typename std::vector<uword>::iterator it=signBits.begin(); it != signBits.end(); it++){
        res->sign.addVerbatim(*it,numberOfElements);
    }
    res->sign.setSizeInBits(numberOfElements);
    res->sign.density = countOnes/(double)numberOfElements;
    
    double existBitDensity = (CountZeros/(double)nums.size());
    
    HybridBitmap<uword> bitmap;
    for(int j=0; j<existBits.size(); j++){
        bitmap.addWord(existBits[j]);
    }
    //bitmap.setSizeInBits(numberOfElements);
    bitmap.density=existBitDensity;
    res->setExistenceBitmap(bitmap);
    
    //The method to put the elements in the input vector nums to the bsi property of BSIAttribute result
    std::vector< std::vector< uword > > bitSlices = bringTheBits(nums,slices,numberOfElements);
    
    for(int i=0; i<slices; i++){
        //build verbatim Bitmap
        double bitDensity = bitSlices[i][0]/(double)numberOfElements;
        HybridBitmap<uword> bitmap(true);
        bitmap.reset();
        bitmap.verbatim = true;
        //                std::copy(bitSlices[i].begin(), bitSlices[i].end(), bitmap.buffer.begin());
        for (typename std::vector<uword>::iterator it=bitSlices[i].begin()+1; it != bitSlices[i].end(); it++){
            bitmap.addVerbatim(*it,numberOfElements);
        }
        // bitmap.buffer=Arrays.copyOfRange(bitSlices[i], 1, bitSlices[i].length);
        //bitmap.actualsizeinwords=bitSlices[i].length-1;
        bitmap.setSizeInBits(numberOfElements);
        bitmap.density=bitDensity;
        res->addSlice(bitmap);
    }
    res->existenceBitmap.setSizeInBits(numberOfElements,true);
    res->existenceBitmap.density=1;
    res->lastSlice=true;
    res->firstSlice=true;
    res->twosComplement = false;
    res->rows = numberOfElements;
    res->is_signed = true;
    return res;
};

template <class uword>
BsiAttribute<uword>* BsiAttribute<uword>::createRandomBsi(int vectorLength, int range, double compressThreshold) const {
    // Calculate slices needed
    int slices = 0;
    uword maxValue = range - 1;
    while (maxValue > 0) {
        slices++;
        maxValue >>= 1;
    }

    // Calculate word counts and prepare result structure first
    int wordsPerSlice = (vectorLength + bits - 1) / bits;

    // BSI first to minimize allocations
    BsiUnsigned<uword>* result = new BsiUnsigned<uword>(slices);
    result->setFirstSliceFlag(true);
    result->setLastSliceFlag(true);
    result->setPartitionID(0);
    result->twosComplement = false;
    result->rows = vectorLength;

    // existence bitmap
    HybridBitmap<uword> existBitmap;
    existBitmap.setSizeInBits(vectorLength, true);
    existBitmap.density = 1.0;
    result->setExistenceBitmap(existBitmap);

    std::vector<HybridBitmap<uword>> sliceBitmaps(slices);
    std::vector<int> bitCounts(slices, 0);

    for (int s = 0; s < slices; s++) {
        sliceBitmaps[s].verbatim = true;
        sliceBitmaps[s].buffer.resize(wordsPerSlice, 0);
    }

    //larger chunks for better cache utilization
    const int CHUNK_SIZE = 4096;  // Process 4K elements at a time for cache efficiency

    for (int chunk = 0; chunk < vectorLength; chunk += CHUNK_SIZE) {
        int chunkEnd = std::min(chunk + CHUNK_SIZE, vectorLength);

        for (int i = chunk; i < chunkEnd; i++) {
            uword randomValue = std::rand() % range;
            int wordIdx = i / bits;
            int bitPos = i % bits;

            //Set bits directly in pre-allocated buffers
            for (int s = 0; s < slices; s++) {
                if (randomValue & (static_cast<uword>(1) << s)) {
                    sliceBitmaps[s].buffer[wordIdx] |= (static_cast<uword>(1) << bitPos);
                    bitCounts[s]++;
                }
            }
        }
    }

    for (int s = 0; s < slices; s++) {
        double density = static_cast<double>(bitCounts[s]) / vectorLength;
        sliceBitmaps[s].density = density;
        sliceBitmaps[s].setSizeInBits(vectorLength);

        double compressRatio = 1 - pow((1 - density), (2 * bits)) - pow(density, (2 * bits));

        if (compressRatio < compressThreshold && compressRatio != 0) {
            HybridBitmap<uword> compressedBitmap;
            for (int w = 0; w < wordsPerSlice; w++) {
                compressedBitmap.addWord(sliceBitmaps[s].buffer[w]);
            }
            compressedBitmap.setSizeInBits(vectorLength);
            compressedBitmap.density = density;
            result->addSlice(compressedBitmap);
        } else {
            result->addSlice(sliceBitmaps[s]);
        }
    }

    return result;
}

/*
 * Check if the array has any signed numbers to know whether to build BsiSigned or BsiUnsigned
 * */

template <class uword>
BsiAttribute<uword>* BsiAttribute<uword>::buildBsiAttributeFromVectorSigned(std::vector<long> nums, double compressThreshold) const{
    const int MAXLONGLENGTH = 64;
    int slices = 0;
    /*
    *
    bits = 8*sizeof(uword);
    If we declare a  BsiAttribute<uint64_t> variable,unsigned long long
    each element in the vector can fit in a 64 bit word
    Therefore bits = 64
    How many such words are needed to represent the sign and non-zero property of each element ?
    If one element is represented by one bit of the sign word, the number of words needed = number of elements/number of bits per word.
    */
    long min = 0;
    int numberOfElements = nums.size();
    std::vector<uword> signBits(numberOfElements/(bits)+1);
    std::vector<uword> existBits(numberOfElements/(bits)+1); // keep track for non-zero values
    int countOnes = 0;
    int CountZeros = 0;
    const uword one = 1;
    //int bits = 8*sizeof(uword);
    //find max, min, and zeros.
    //Setting sign bits and existence bits for the array of numbers
    for (int i=0; i<nums.size(); i++){
        int offset = i%(bits);
        min = std::min(min,nums[i]);
        if(nums[i] < 0){
            nums[i] = 0 - nums[i];
            signBits[i / (bits)] |= (one << offset); // setting sign bit
            countOnes++;
        }
        existBits[i / (bits)] |= (one << offset); // seting one at position
        if(nums[i] == 0){
            CountZeros++;
        }
        slices = std::max(slices,sliceLengthFinder(nums[i])); //Finding the maximum length of the bit representation of the numbers
    }

    BsiAttribute* res;
    //Let's try to always build a BsiSigned Vector
    res = new BsiSigned<uword>(slices + 1);
    res->sign.reset();
    res->sign.verbatim = true;
    for (typename std::vector<uword>::iterator it = signBits.begin(); it != signBits.end(); it++) {
        res->sign.addVerbatim(*it, numberOfElements);
    }
    res->sign.setSizeInBits(numberOfElements);
    res->sign.density = countOnes / (double)numberOfElements;
    res->sign.density = countOnes / (double)numberOfElements;
    /*
    * if (min < 0) {
        res = new BsiSigned<uword>(slices+1);
        res->sign.reset();
        res->sign.verbatim = true;

        for (typename std::vector<uword>::iterator it=signBits.begin(); it != signBits.end(); it++){
            res->sign.addVerbatim(*it,numberOfElements);
        }
        res->sign.setSizeInBits(numberOfElements);
        res->sign.density = countOnes/(double)numberOfElements;
    } else {
        res = new BsiUnsigned<uword>(slices+1);
    }
    */
    

    double existBitDensity = (CountZeros/(double)nums.size()); // to decide whether to compress or not
    double existCompressRatio = 1-pow((1-existBitDensity), (2*bits))-pow(existBitDensity, (2*bits));
    if(existCompressRatio >= compressThreshold){
        HybridBitmap<uword> bitmap;
        for(int j=0; j<existBits.size(); j++){
            bitmap.addWord(existBits[j]);
        }
        //bitmap.setSizeInBits(numberOfElements);
        bitmap.density=existBitDensity;
        res->setExistenceBitmap(bitmap);
    }else{
        HybridBitmap<uword> bitmap(true,existBits.size());
        for(int j=0; j<existBits.size(); j++){
            bitmap.buffer[j] = existBits[j];
        }
        //bitmap.setSizeInBits(numberOfElements);
        bitmap.density=existBitDensity;
        res->setExistenceBitmap(bitmap);
    }

    //The method to put the elements in the input vector nums to the bsi property of BSIAttribute result
    std::vector< std::vector< uword > > bitSlices = bringTheBits(nums,slices,numberOfElements);

    for(int i=0; i<slices; i++){
        double bitDensity = bitSlices[i][0]/(double)numberOfElements; // the bit density for this slice
        double compressRatio = 1-pow((1-bitDensity), (2*bits))-pow(bitDensity, (2*bits));
        if(compressRatio<compressThreshold && compressRatio!=0 ){
            //build compressed bitmap
            HybridBitmap<uword> bitmap;
            for(int j=1; j<bitSlices[i].size(); j++){
                bitmap.addWord(bitSlices[i][j]);
            }
            //bitmap.setSizeInBits(numberOfElements);
            bitmap.density=bitDensity;
            res->addSlice(bitmap);

        }else{
            //build verbatim Bitmap
            HybridBitmap<uword> bitmap(true);
            bitmap.reset();
            bitmap.verbatim = true;
            //                std::copy(bitSlices[i].begin(), bitSlices[i].end(), bitmap.buffer.begin());
            for (typename std::vector<uword>::iterator it=bitSlices[i].begin()+1; it != bitSlices[i].end(); it++){
                bitmap.addVerbatim(*it,numberOfElements);
            }
            // bitmap.buffer=Arrays.copyOfRange(bitSlices[i], 1, bitSlices[i].length);
            //bitmap.actualsizeinwords=bitSlices[i].length-1;
            bitmap.setSizeInBits(numberOfElements);
            bitmap.density=bitDensity;
            res->addSlice(bitmap);

        }
    }
    res->existenceBitmap.setSizeInBits(numberOfElements,true);
    res->existenceBitmap.density=1;
    res->lastSlice=true;
    res->firstSlice=true;
    //res->twosComplement = false;
    res->rows = numberOfElements;
    //res->is_signed = true;
    return res;
};

/**
 * function to be parallelised for bringTheBits
 * 
 * For each slice we do the following:
 * Create offsetter, which is a number with only one bit as 1 and rest are all 0. The bit that is set to 1 represents the slice being worked on
 * 
 * For each element in the array, we calculate which word we are working on (w) and which position the element is in the array (offset)
 * if the element has the bit at the slice set to 1 we OR the 1 in the result and increment the count of 1s.
*/
template <class uword> 
void BsiAttribute<uword>::bringTheBitsHelper(const std::vector<long> &array, int slice, int numberOfElements,
                                            std::vector< std::vector< uword > > &bitmapDataRaw) const{
    const uword one = 1;
    uword offsetter = (one << slice);

    for(int seq=0; seq<numberOfElements; seq++) {
        uword thisBin = array[seq];

        // the following calculations are done over and over and see if the results can be saved somewhere
        int w = (seq / (bits) + 1);
        int offset = seq % (bits);

        //update if one
        //ToDo confirm the AND operation is effecient enough
        if( (thisBin & offsetter) == offsetter) {
            bitmapDataRaw[slice][w] |= (one << offset);
            bitmapDataRaw[slice][0]++;
        }
    }
}

/*
 * Private Function

    Example:
    If input array is {4, 5, 6}
    In binary it would be: {100, 101, 110}
    For each slice (column-wise we want the binary representation and the number of 1s present)
                Therefore: {1 0 0,
                            1 0 1,   ^
                            1 1 0}   |
Representation (upwards):   7 4 2
                    1s:     3 1 1
    Binary representation is taken from last element to first, the rightmost-bit is the first row of the result

    For input {4, 5, 6} we get the result {{1, 2}, {1, 4}, {3, 7}} (each row is {number of 1s, number from binary counting from down to up})


    This function works by creating the result matrix (as we already know the size) and then working on each slice in threads (use -pthread while compiling if it doesnt work for you)
 */
template <class uword>
std::vector< std::vector< uword > > BsiAttribute<uword>::bringTheBits(const std::vector<long> &array, int slices, int numberOfElements) const{
    //The number of words needed to represent the elements in the array
    int wordsNeeded = ceil( numberOfElements / (double)(bits));
    //output
    std::vector< std::vector< uword > > bitmapDataRaw(slices,std::vector<uword>(wordsNeeded +1));

    //multithread
    std::vector<std::thread> helperThreads;
    for(int slice=0; slice<slices; slice++) {
        //std::ref is a wrapper around reference variables to make the compiler happy
        helperThreads.push_back( std::thread(&BsiAttribute::bringTheBitsHelper, this, std::ref(array), slice, numberOfElements, std::ref(bitmapDataRaw)) );
    }

    for(int slice=0; slice<slices; slice++) {
        helperThreads[slice].join();
    }

    return bitmapDataRaw;
};

/*
 * maj perform c = a&b | b&c | a&c
 */
template <class uword>
HybridBitmap<uword> BsiAttribute<uword>::maj(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const{
    
    if(a.verbatim && b.verbatim && c.verbatim){
        return a.maj(b, c);
    }else{
        
        return a.logicaland(b).logicalor(b.logicaland(c)).logicalor(a.logicaland(c));
    }
};

/*
* XOR operation
*/


template <class uword>
HybridBitmap<uword> BsiAttribute<uword>::XOR(const HybridBitmap<uword> &a,const HybridBitmap<uword> &b,const HybridBitmap<uword> &c)const{
    if(a.verbatim && b.verbatim && c.verbatim){
        return a.Xor(b.Xor(c));
    }else{
        return a.Xor(b).Xor(c);
    }
};

/*
 * perform  a | b & ~c
 */
template <class uword>
HybridBitmap<uword> BsiAttribute<uword>::orAndNot(const HybridBitmap<uword> &a,const HybridBitmap<uword> &b,const HybridBitmap<uword> &c)const{
    if(a.verbatim && b.verbatim && c.verbatim){
        return a.orAndNotV(b,c);
    }else{
        return a.logicalor(b.andNot(c));
    }
};


/*
 * perform  a | b & c
 */
template <class uword>
HybridBitmap<uword> BsiAttribute<uword>::orAnd(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const{
    if(a.verbatim && b.verbatim && c.verbatim){
        return a.orAndV(b,c);
    }else{
        return a.logicalor(b.logicaland(c));
    }
};

template <class uword>
HybridBitmap<uword> BsiAttribute<uword>::And(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const{
    if(a.verbatim && b.verbatim && c.verbatim){
        return a.andVerbatim(b,c);
    }else{
        return a.logicaland(b.logicaland(c));
    }
};



/*
 *
 */
template <class uword>
void BsiAttribute<uword>::signMagnitudeToTwos(int bits){
    int i=0;
    for(i=0; i<getNumberOfSlices(); i++){
        bsi[i]=bsi[i].Xor(sign);
    }
    while(i<bits){ // sign extension
        
        addSlice(sign);
        i++;
    }
    if(this->firstSlice){
        addOneSliceSameOffset(sign);
    }
    
    setTwosFlag(true);
};


template <class uword>
BsiAttribute<uword>* BsiAttribute<uword>::signMagnToTwos(int bit_limit){
    BsiAttribute* res = new BsiSigned<uword>();
    res->twosComplement=true;
    int i=0;
    for(i=0; i<getNumberOfSlices(); i++){
        res->bsi[i]=bsi[i].Xor(sign);
    }
    while(i<bit_limit){
        res->addSlice(sign);
        i++;}
    if(firstSlice){
        res->addOneSliceSameOffset(sign);
    }
    return res;
};

template <class uword>
BsiAttribute<uword>* BsiAttribute<uword>::TwosToSignMagnitue(){
    BsiAttribute* res = new BsiSigned<uword>();
    for (int i=0; i<size; i++){
        res->bsi[i]=bsi[i].logicalxor(bsi[size-1]);
    }if(firstSlice){
        res->addOneSliceSameOffset(bsi[size-1]);
    }
    return res;
};


template <class uword>
void BsiAttribute<uword>::addOneSliceSameOffset(const HybridBitmap<uword> &slice){
    HybridBitmap<uword> C,S;
    S=bsi[0].Xor(slice);
    C=bsi[0].And(slice);
    bsi[0]=S;
    int curPos =1;
    while(C.numberOfOnes()>0){
        if(curPos<size){
            S=C.Xor(bsi[curPos]);
            C=C.And(bsi[curPos]);
            bsi[curPos]=S;
            curPos++;
        }else{
            addSlice(C);
            return;
        }
    }
};


template <class uword>
void BsiAttribute<uword>::addOneSliceDiscardCarry(const HybridBitmap<uword> &slice){
    
    HybridBitmap<uword> C,S;
    S=bsi[0].Xor(slice);
    C=bsi[0].And(slice);
    bsi[0]=S;
    int curPos =1;
    while(C.numberOfOnes()>0){
        if(curPos<size){
            S=C.Xor(bsi[curPos]);
            C=C.And(bsi[curPos]);
            bsi[curPos]=S;
            curPos++;
        }
    }
};

template <class uword>
void BsiAttribute<uword>::addOneSliceNoSignExt(const HybridBitmap<uword> &slice){
    
    
    HybridBitmap<uword> C,S;
    S=bsi[0].Xor(slice);
    C=bsi[0].And(slice);
    bsi[0]=S;
    int curPos =1;
    while(C.numberOfOnes()>0){
        if(curPos<size){
            S=C.Xor(bsi[curPos]);
            C=C.And(bsi[curPos]);
            bsi[curPos]=S;
            curPos++;
        }else return;
    }
};

template <class uword>
void BsiAttribute<uword>::applyExsistenceBitmap(const HybridBitmap<uword> &ex){
    existenceBitmap = ex;
    for(int i=0; i< size; i++){
        this->bsi[i] = bsi[i].And(ex);
    }
//    addSlice(ex.Not());
};


/*
 * buildCompressedBsiFromVector is used for making synchronised compressed bsi
 * every bitmap is compressed by same positions
 */


template <class uword>
BsiAttribute<uword>* BsiAttribute<uword>::buildCompressedBsiFromVector(std::vector<long> nums, double compressThreshold) const{
    uword max = std::numeric_limits<uword>::min();
    
    int attRows = nums.size();
    //    int slices = 3*digits + (int)std::log2(digits);
    std::vector<uword> signBits(attRows/(bits)+1);
    std::vector<uword> existBits(attRows/(bits)+1);
    int countOnes =0;
    int CountZeros = 0;
    for (int i=0; i<nums.size(); i++){
        int offset = i%(bits);
        if(nums[i] < 0){
            nums[i] = 0 - nums[i];
            signBits[i / (bits)] |= (1L << offset);
            countOnes++;
        }
        if(nums[i] != 0){
            existBits[i / (bits)] |= (1L << offset);
        }else{
            CountZeros++;
        }
        if(nums[i] > max){
            max = nums[i];
        }
    }
    int slices = sliceLengthFinder(max);
    BsiSigned<uword>* res = new BsiSigned<uword>(slices+1);
    res->sign.reset();
    res->sign.verbatim = true;
    
    for (typename std::vector<uword>::iterator it=signBits.begin(); it != signBits.end(); it++){
        res->sign.addVerbatim(*it,attRows);
    }
    res->sign.setSizeInBits(attRows);
    res->sign.density = countOnes/(double)attRows;
    
    double existBitDensity = 1- (CountZeros/(double)nums.size());
    double existCompressRatio = pow((1-existBitDensity), (2*bits))+pow(existBitDensity, (2*bits));
    if(existCompressRatio >= compressThreshold){
        HybridBitmap<uword> bitmap;
        for(int j=0; j<existBits.size(); j++){
            bitmap.addWord(existBits[j]);
        }
        bitmap.setSizeInBits(nums.size());
        bitmap.density=existBitDensity;
        res->setExistenceBitmap(bitmap);
    }else{
        HybridBitmap<uword> bitmap(true,existBits.size());
        for(int j=0; j<existBits.size(); j++){
            bitmap.buffer[j] = existBits[j];
        }
        bitmap.density=existBitDensity;
        res->setExistenceBitmap(bitmap);
    }
    std::vector<std::vector<uword> > bitSlices = bringTheBits(nums,slices,attRows);
    for(int i=0; i<slices; i++){
        double bitDensity = bitSlices[i][0]/(double)attRows; // the bit density for this slice
        double compressRatio = 1-pow((1-bitDensity), (2*bits))-pow(bitDensity, (2*bits));
        if(!res->existenceBitmap.isVerbatim()){
            //build compressed bitmap
            HybridBitmap<uword> bitmap;
            HybridBitmapRawIterator<uword> ii = res->existenceBitmap.raw_iterator();
            BufferedRunningLengthWord<uword> &rlwi = ii.next();
            int position = 1;
            while ( rlwi.size() > 0) {
                while (rlwi.getRunningLength() > 0) {
                    bitmap.addStreamOfEmptyWords(0, rlwi.getRunningLength());
                    position += rlwi.getRunningLength();
                    rlwi.discardRunningWordsWithReload();
                }
                const size_t nbre_literal = rlwi.getNumberOfLiteralWords();
                if (nbre_literal > 0) {
                    for (size_t k = 0; k < nbre_literal; ++k) {
                        bitmap.addLiteralWord(bitSlices[i][position]);
                        position++;
                    }
                }
                rlwi.discardLiteralWordsWithReload(nbre_literal);
            }
            bitmap.density=bitDensity;
            bitmap.setSizeInBits(nums.size());
            res->addSlice(bitmap);
            
        }else{
            //build verbatim Bitmap
            HybridBitmap<uword> bitmap(true);
            bitmap.reset();
            bitmap.verbatim = true;
            for (typename std::vector<uword>::iterator it=bitSlices[i].begin()+1; it != bitSlices[i].end(); it++){
                bitmap.addVerbatim(*it,attRows);
            }
            bitmap.setSizeInBits(attRows);
            bitmap.density=bitDensity;
            res->addSlice(bitmap);
            
        }
    }
    res->existenceBitmap.setSizeInBits(attRows,true);
    res->existenceBitmap.density=1;
    res->lastSlice=true;
    res->firstSlice=true;
    res->twosComplement=false;
    res->rows = attRows;
    res->is_signed = true;
    return res;
};

#endif /* BsiAttribute_hpp */
