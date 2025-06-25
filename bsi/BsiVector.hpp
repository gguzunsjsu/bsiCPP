/**
 * BsiVector.hpp
 * @author gguzun
 * This defines the methods for a BsiVector.
 * The least significant bit/slice is slice 0.
 */

#ifndef BsiVector_hpp
#define BsiVector_hpp

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


template <class uword = uint64_t> class BsiUnsigned;
template <class uword = uint64_t> class BsiSigned;
//template <class uword = uint32_t>
template <class uword = uint64_t> class BsiVector{
public:
    int numSlices; //holds number of slices
    int offset =0;
    int decimals = 0;
    int bits = 8*sizeof(uword);

    std::vector<HybridBitmap<uword> > bsi ;
    HybridBitmap<uword> existenceBitmap;
    HybridBitmap<uword> sign; // sign bitslice

    long rows;  // number of elements
    long index; // if split horizontally

    bool is_signed; // sign flag

    bool firstSlice; //contains first slice(least significant)
    bool lastSlice; //contains last slice(most significant)
    
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

    virtual BsiVector* SUM(BsiVector* a)=0;
    virtual BsiVector* SUMsigned(BsiVector* a)=0;
    virtual BsiVector* SUM(long a)const=0;
    virtual BsiVector<uword>* sum_Horizontal(const BsiVector<uword> *a) const=0;

    virtual BsiVector* convertToTwos(int bits)=0;

    virtual BsiUnsigned<uword>* abs()=0;
    virtual BsiUnsigned<uword>* abs(int resultSlices,const HybridBitmap<uword> &EB)=0;
    virtual BsiUnsigned<uword>* absScale(double range)=0;
    virtual int compareTo(BsiVector<uword> *a, int index)=0;

    virtual long getValue(int pos)const=0;

    virtual HybridBitmap<uword> rangeBetween(long lowerBound, long upperBound)=0;
    virtual BsiVector<uword>* multiplyByConstantNew(int number)const=0;
    virtual BsiVector<uword>* multiplyByConstant(int number)const=0;
    virtual BsiVector<uword>* multiplication(BsiVector<uword> *a)const=0;
    virtual BsiVector<uword>* multiplication_array(BsiVector<uword> *a)const=0;
    virtual BsiVector<uword>* multiplyBSI(BsiVector<uword> *a) const=0;
    virtual BsiVector<uword>*  multiplyWithBsiHorizontal(const BsiVector<uword> *unbsi, int precision) const=0;
    virtual BsiVector<uword>*  multiplyWithBsiHorizontal(const BsiVector<uword> *unbsi) const=0;
    virtual BsiVector<uword>* multiplication_Horizontal(const BsiVector<uword> *a) const=0;
    virtual BsiVector<uword>* multiplication_Horizontal_Hybrid(const BsiVector<uword> *a) const=0;
    virtual BsiVector<uword>* multiplication_Horizontal_Verbatim(const BsiVector<uword> *a) const=0;
    virtual BsiVector<uword>* multiplication_Horizontal_compressed(const BsiVector<uword> *a) const=0;
    virtual BsiVector<uword>* multiplication_Horizontal_Hybrid_other(const BsiVector<uword> *a) const=0;
    virtual long dotProduct(BsiVector<uword>* a) const = 0;
    virtual long long int dot(BsiVector<uword>* a) const = 0;
    virtual long long int dot_withoutCompression(BsiVector<uword>* a) const = 0;
    virtual void multiplicationInPlace(BsiVector<uword> *a)=0;

    virtual BsiVector<uword>* negate()=0;

    virtual long sumOfBsi()const=0;

    virtual bool append(long value)=0;
    
    BsiVector* buildQueryAttribute(long query, int rows, long partitionID);

    BsiVector<uword>* buildBsiVector(std::vector<long> decimalVector, double compressThreshold) const;
    BsiVector<uword>* buildBsiVector(std::vector<double> decimalVector, int decimalPrecision, double compressThreshold) const;
    
    HybridBitmap<uword> maj(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const;

    HybridBitmap<uword> XOR(const HybridBitmap<uword> &a,const HybridBitmap<uword> &b,const HybridBitmap<uword> &c)const;
    HybridBitmap<uword> orAndNot(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const;
    HybridBitmap<uword> orAnd(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const;
    HybridBitmap<uword> And(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const;
   
    BsiVector* signMagnToTwos(int bits);
    BsiVector* TwosToSignMagnitue();
    void signMagnitudeToTwos(int bits);

    void addOneSliceSameOffset(const HybridBitmap<uword> &slice);
    void addOneSliceDiscardCarry(const HybridBitmap<uword> &slice);
    void addOneSliceNoSignExt(const HybridBitmap<uword> &slice);
    void applyExsistenceBitmap(const HybridBitmap<uword> &ex);    
    virtual ~BsiVector();
//    size_t getSizeInMemory() const {
//        size_t size_in_memory = sizeof(*this);
//
//        // Add the numSlices of dynamically allocated vectors (assuming they are vectors)
//        size_in_memory += bsi.numSlices() * sizeof(HybridBitmap<>);  // adjust as needed
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
    void bringTheBitsHelper(const std::vector<long> &array, int slice, long numberOfElements, std::vector<std::vector<uword>> &bitmapDataRaw) const;
    std::vector< std::vector< uword > > bringTheBits(const std::vector<long> &array, int slices, long vectorElements) const;
    std::vector< std::vector< uword > > bringTheBits(const std::vector<long> &array, int slices) const;
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
* buildBsiVectorFromVector
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
BsiVector<uword>::~BsiVector(){
};


template <class uword>
bool BsiVector<uword>::isLastSlice() const{
    return lastSlice;
};

/*
 *
 * @param flag if the attribute contains the most significant slice then set it to true. Otherwise false.
 */
template <class uword>
void BsiVector<uword>::setLastSliceFlag(bool flag){
    lastSlice=flag;
};
/*
 *
 * @return if this attribute contains the first slice(least significant). For internal purposes (when splitting into sub attributes)
 */
template <class uword>
bool BsiVector<uword>::isFirstSlice()const{
    return firstSlice;
};
/*
 *
 * @param flag if the attribute contains the least significant slice then set it to true. Otherwise false.
 */
template <class uword>
void BsiVector<uword>::setFirstSliceFlag(bool flag){
    firstSlice=flag;
};

/*
 * Returns false if contains only positive numbers
 */
template <class uword>
bool BsiVector<uword>::isSigned()const{
    return is_signed;
};

template <class uword>
void BsiVector<uword>::addSlice(const HybridBitmap<uword> &slice){
    bsi.push_back(slice);
    numSlices++;
};



/*
 * Don't use for already built bsi
 */

template <class uword>
void BsiVector<uword>::setNumberOfSlices(int s){
    numSlices = s;
}

/**
 * Returns the numSlices of the bsi (how many slices are non zeros) ----> Really ? This just returns the number of slices, right ?
 */
template <class uword>
int BsiVector<uword>::getNumberOfSlices()const{
    return numSlices;
}

/**
 * Returns the slice number i
 */
template <class uword>
HybridBitmap<uword> BsiVector<uword>::getSlice(int i) const{
    return bsi[i];
}


/**
 * Returns the offset of the bsi (the first "offset" slices are zero, thus not encoding)
 */
template <class uword>
int BsiVector<uword>::getOffset() const{
    return offset;
}


/**
 * Sets the offset of the bsi (the first "offset" slices are zero, thus not encoding)
 */

template <class uword>
void BsiVector<uword>::setOffset(int offset){
    BsiVector::offset=offset;
}

/**
 * Returns the number of rows for this attribute
 */

template <class uword>
long BsiVector<uword>::getNumberOfRows() const{
    return rows;
}


/**
 * Sets the number of rows for this attribute
 */
template <class uword>
void BsiVector<uword>::setNumberOfRows(long rows){
    BsiVector::rows=rows;
}

/**
 * Returns the index(partition id if horizontally partitioned) for this attribute
 */
template <class uword>
long BsiVector<uword>::getPartitionID() const{
    return index;
}

/**
 * Sets the index(partition id if horizontally partitioned) for this attribute
 */
template <class uword>
void BsiVector<uword>::setPartitionID(long index){
    BsiVector::index=index;
}

/**
 * Returns the Existence bitmap of the bsi attribute
 */
template <class uword>
HybridBitmap<uword> BsiVector<uword>::getExistenceBitmap(){
    return existenceBitmap;
}

/**
 * Sets the existence bitmap of the bsi attribute
 */
template <class uword>
void BsiVector<uword>::setExistenceBitmap(const HybridBitmap<uword> &exBitmap){
    BsiVector::existenceBitmap=exBitmap;
}

/**
 * flag is true when bsi contain data into two's complement form
 */
template <class uword>
void BsiVector<uword>::setTwosFlag(bool flag){
    twosComplement=flag;
}

/**
 * builds a BSI attribute with all rows identical given one number (row)
 * @param query
 * @param rows
 * @return the BSI attribute with all rows identical
 */

template <class uword>
BsiVector<uword>* BsiVector<uword>::buildQueryAttribute(long query, int rows, long partitionID){
    if(query<0){
        uword q = std::abs(query);
        int maxsize = sliceLengthFinder(q);
        BsiVector* res = new BsiUnsigned<uword>(maxsize);
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
        BsiVector* res = new BsiUnsigned<uword>(maxsize);
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
int BsiVector<uword>::sliceLengthFinder(uword value) const{
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
 * @param compressThreshold determined whether to compress the bit vector or not. Set it to zero to NOT compress.
 * @param compressThreshold Set it to one to compress
 */

template <class uword>
BsiVector<uword>* BsiVector<uword>::buildBsiVector(std::vector<long> nums, double compressThreshold) const{
    long max = INT64_MIN;
    long min = INT64_MAX;
    int numberOfElements = nums.size();
    int count = 0;

    for (int it = 0; it  < numberOfElements; it++, count++) {
        max = std::max(max, nums[count]);
        min = std::min(min, nums[count]);
    }

    int slices =  std::__bit_width(std::max(std::abs(min), std::abs(max)));

    if (min < 0) {
        BsiVector<uword>* res = new BsiSigned<uword>(slices+1);
        std::vector< std::vector< uword > > bitSlices = bringTheBits(nums,slices+1,numberOfElements);
        for(int i=0; i<=slices; i++){
            double bitDensity = bitSlices[i][0]/(double)numberOfElements; // the bit density for this slice
            double compressRatio = 1-pow((1-bitDensity), (2*bits))-pow(bitDensity, (2*bits));
            if(compressRatio<compressThreshold && compressRatio!=0 ){
                //build compressed bitmap
                HybridBitmap<uword> bitmap;
                for(int j=1; j<bitSlices[i].size()-1; j++){
                    bitmap.addWord(bitSlices[i][j]);
                }

                int bitsthatmatter = bits-(bits*(bitSlices[i].size()-1)-numberOfElements);
                bitmap.addWord(bitSlices[i][bitSlices[i].size()-1],bitsthatmatter);
                bitmap.density=bitDensity;
                res->addSlice(bitmap);
            }else {
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
        res->sign = res->bsi[res->numSlices - 1];
        res->lastSlice = true;
        res->firstSlice = true;
        res->twosComplement = true;
        res->rows = numberOfElements;
        res->is_signed = true;
        int wholeWords = floor(numberOfElements/(float)bits);
        res->existenceBitmap.addStreamOfEmptyWords(true,wholeWords);
        res->existenceBitmap.addVerbatim(~(uword)0, numberOfElements-(wholeWords*bits)); res->lastSlice=true;
        res->existenceBitmap.density = 1;
        return res;

    }else {
        //int slices = std::__bit_width(max);
        BsiUnsigned<uword>* res = new BsiUnsigned<uword>(slices);


        //The method to put the elements in the input vector nums to the bsi property of BSIAttribute result
        std::vector< std::vector< uword > > bitSlices = bringTheBits(nums,slices,numberOfElements);

        for(int i=0; i<slices; i++){
            double bitDensity = bitSlices[i][0]/(double)numberOfElements; // the bit density for this slice
            double compressRatio = 1-pow((1-bitDensity), (2*bits))-pow(bitDensity, (2*bits));
            if(compressRatio<compressThreshold && compressRatio!=0 ){
                //build compressed bitmap
                HybridBitmap<uword> bitmap;
                for(int j=1; j<bitSlices[i].size()-1; j++){
                    bitmap.addWord(bitSlices[i][j]);
                }

                int bitsthatmatter = bits-(bits*(bitSlices[i].size()-1)-numberOfElements);
                bitmap.addWord(bitSlices[i][bitSlices[i].size()-1],bitsthatmatter);
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
        int wholeWords = floor(numberOfElements/(float)bits);
        res->sign.addStreamOfEmptyWords(false,wholeWords);
        res->sign.addVerbatim(0, numberOfElements-(wholeWords*bits));
        res->sign.density = 0;

        //this existence bitmap causing issues with SUM
        res->existenceBitmap.addStreamOfEmptyWords(true,wholeWords);
        res->existenceBitmap.addVerbatim(~(uword)0, numberOfElements-(wholeWords*bits)); res->lastSlice=true;
        res->existenceBitmap.density = 1;
        res->firstSlice=true;
        //res->twosComplement = false;
        res->rows = numberOfElements;
        //res->is_signed = true;
        return res;
    }
};

/*
 * Used for converting vector into BSI
 * @param compressThreshold determined whether to compress the bit vector or not. Set it to zero to NOT compress.
 * @param compressThreshold Set it to one to compress
 * @param decimalPlaces: precision as number of digits to the right of the decimal point in a decimal number.
 */

template <class uword>
BsiVector<uword>* BsiVector<uword>::buildBsiVector(std::vector<double> nums, int decimalPlaces, double compressThreshold) const{
    long max = INT64_MIN;
    long min = INT64_MAX;
    int numberOfElements = nums.size();
    int count = 0;
    std::vector<long> nums_long(numberOfElements);



    for (int it = 0; it  < numberOfElements; it++, count++) {
        nums_long[it]=(long)round(nums[count]*pow(10,decimalPlaces));
        max = std::max(max, nums_long[it] );
        min = std::min(min,  nums_long[it]);
    }

    int slices =  std::__bit_width(std::max(std::abs(min), std::abs(max)));

    if (min < 0) {
        BsiVector<uword>* res = new BsiSigned<uword>(slices+1);
        std::vector< std::vector< uword > > bitSlices = bringTheBits(nums_long,slices+1,numberOfElements);
        for(int i=0; i<=slices; i++){
            double bitDensity = bitSlices[i][0]/(double)numberOfElements; // the bit density for this slice
            double compressRatio = 1-pow((1-bitDensity), (2*bits))-pow(bitDensity, (2*bits));
            if(compressRatio<compressThreshold && compressRatio!=0 ){
                //build compressed bitmap
                HybridBitmap<uword> bitmap;
                for(int j=1; j<bitSlices[i].size()-1; j++){
                    bitmap.addWord(bitSlices[i][j]);
                }

                int bitsthatmatter = bits-(bits*(bitSlices[i].size()-1)-numberOfElements);
                bitmap.addWord(bitSlices[i][bitSlices[i].size()-1],bitsthatmatter);
                bitmap.density=bitDensity;
                res->addSlice(bitmap);
            }else {
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
        res->sign = res->bsi[res->numSlices - 1];
        res->lastSlice = true;
        res->firstSlice = true;
        res->twosComplement = true;
        res->rows = numberOfElements;
        res->is_signed = true;
        int wholeWords = floor(numberOfElements/(float)bits);
        res->existenceBitmap.addStreamOfEmptyWords(true,wholeWords);
        res->existenceBitmap.addVerbatim(~(uword)0, numberOfElements-(wholeWords*bits)); res->lastSlice=true;
        res->existenceBitmap.density = 1;
        return res;

    }else {
        //int slices = std::__bit_width(max);
        BsiUnsigned<uword>* res = new BsiUnsigned<uword>(slices);


        //The method to put the elements in the input vector nums to the bsi property of BSIAttribute result
        std::vector< std::vector< uword > > bitSlices = bringTheBits(nums_long,slices,numberOfElements);

        for(int i=0; i<slices; i++){
            double bitDensity = bitSlices[i][0]/(double)numberOfElements; // the bit density for this slice
            double compressRatio = 1-pow((1-bitDensity), (2*bits))-pow(bitDensity, (2*bits));
            if(compressRatio<compressThreshold && compressRatio!=0 ){
                //build compressed bitmap
                HybridBitmap<uword> bitmap;
                for(int j=1; j<bitSlices[i].size()-1; j++){
                    bitmap.addWord(bitSlices[i][j]);
                }

                int bitsthatmatter = bits-(bits*(bitSlices[i].size()-1)-numberOfElements);
                bitmap.addWord(bitSlices[i][bitSlices[i].size()-1],bitsthatmatter);
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
        int wholeWords = floor(numberOfElements/(float)bits);
        res->sign.addStreamOfEmptyWords(false,wholeWords);
        res->sign.addVerbatim(0, numberOfElements-(wholeWords*bits));
        res->sign.density = 0;

        //this existence bitmap causing issues with SUM
        res->existenceBitmap.addStreamOfEmptyWords(true,wholeWords);
        res->existenceBitmap.addVerbatim(~(uword)0, numberOfElements-(wholeWords*bits)); res->lastSlice=true;
        res->existenceBitmap.density = 1;
        res->firstSlice=true;
        //res->twosComplement = false;
        res->rows = numberOfElements;
        //res->is_signed = true;
        return res;
    }
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
void BsiVector<uword>::bringTheBitsHelper(const std::vector<long> &array, int slice, long numberOfElements,
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


    This function works by creating the result matrix (as we already know the numSlices) and then working on each slice in threads (use -pthread while compiling if it doesnt work for you)
 */
template <class uword>
std::vector< std::vector< uword > > BsiVector<uword>::bringTheBits(const std::vector<long> &array, int slices, long numberOfElements) const{
    //The number of words needed to represent the elements in the array
    int wordsNeeded = ceil( numberOfElements / (double)(bits));
    //output
    std::vector< std::vector< uword > > bitmapDataRaw(slices,std::vector<uword>(wordsNeeded +1));

    //multithread
    std::vector<std::thread> helperThreads;
    for(int slice=0; slice<slices; slice++) {
        //std::ref is a wrapper around reference variables to make the compiler happy
        helperThreads.push_back( std::thread(&BsiVector::bringTheBitsHelper, this, std::ref(array), slice, numberOfElements, std::ref(bitmapDataRaw)) );
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
HybridBitmap<uword> BsiVector<uword>::maj(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const{
    
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
HybridBitmap<uword> BsiVector<uword>::XOR(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const{
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
HybridBitmap<uword> BsiVector<uword>::orAndNot(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const{
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
HybridBitmap<uword> BsiVector<uword>::orAnd(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const{
    if(a.verbatim && b.verbatim && c.verbatim){
        return a.orAndV(b,c);
    }else{
        return a.logicalor(b.logicaland(c));
    }
};

template <class uword>
HybridBitmap<uword> BsiVector<uword>::And(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const{
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
void BsiVector<uword>::signMagnitudeToTwos(int bits){
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
BsiVector<uword>* BsiVector<uword>::signMagnToTwos(int bit_limit){
    BsiVector* res = new BsiSigned<uword>();
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
BsiVector<uword>* BsiVector<uword>::TwosToSignMagnitue(){
    BsiVector* res = new BsiSigned<uword>();
    for (int i=0; i < numSlices; i++){
        res->bsi[i]=bsi[i].logicalxor(bsi[numSlices - 1]);
    }if(firstSlice){
        res->addOneSliceSameOffset(bsi[numSlices - 1]);
    }
    return res;
};


template <class uword>
void BsiVector<uword>::addOneSliceSameOffset(const HybridBitmap<uword> &slice){
    HybridBitmap<uword> C,S;
    S=bsi[0].Xor(slice);
    C=bsi[0].And(slice);
    bsi[0]=S;
    int curPos =1;
    while(C.numberOfOnes()>0){
        if(curPos < numSlices){
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
void BsiVector<uword>::addOneSliceDiscardCarry(const HybridBitmap<uword> &slice){
    
    HybridBitmap<uword> C,S;
    S=bsi[0].Xor(slice);
    C=bsi[0].And(slice);
    bsi[0]=S;
    int curPos =1;
    while(C.numberOfOnes()>0){
        if(curPos < numSlices){
            S=C.Xor(bsi[curPos]);
            C=C.And(bsi[curPos]);
            bsi[curPos]=S;
            curPos++;
        }
    }
};

template <class uword>
void BsiVector<uword>::addOneSliceNoSignExt(const HybridBitmap<uword> &slice){
    
    
    HybridBitmap<uword> C,S;
    S=bsi[0].Xor(slice);
    C=bsi[0].And(slice);
    bsi[0]=S;
    int curPos =1;
    while(C.numberOfOnes()>0){
        if(curPos < numSlices){
            S=C.Xor(bsi[curPos]);
            C=C.And(bsi[curPos]);
            bsi[curPos]=S;
            curPos++;
        }else return;
    }
};

template <class uword>
void BsiVector<uword>::applyExsistenceBitmap(const HybridBitmap<uword> &ex){
    existenceBitmap = ex;
    for(int i=0; i < numSlices; i++){
        this->bsi[i] = bsi[i].And(ex);
    }
//    addSlice(ex.Not());
};


#endif /* BsiAttribute_hpp */
