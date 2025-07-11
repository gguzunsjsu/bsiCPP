//
//  BsiSigned.hpp
//

#ifndef BsiSigned_hpp
#define BsiSigned_hpp

#include <stdio.h>
#include <stdlib.h>     /* abs */
#include "BsiVector.hpp"
template <class uword>
class BsiSigned: public BsiVector<uword>{
public:
    /*
     Declaring Constructors
     */
    BsiSigned();
    BsiSigned(int maxSize);
    BsiSigned(int maxSize, int numOfRows);
    BsiSigned(int maxSize, int numOfRows, long partitionID);
    BsiSigned(int maxSize, long numOfRows, long partitionID, HybridBitmap<uword> ex);
    
    /*
     Declaring Override Functions
     */
    
    HybridBitmap<uword> topKMax(int k) override;
    HybridBitmap<uword> topKMin(int k) override;
    BsiVector<uword>* SUM(BsiVector<uword>* a) override;
    BsiVector<uword>* SUM(long a)const override;
    BsiVector<uword>* convertToTwos(int bits) override;
    long getValue(int pos) const override;
    HybridBitmap<uword> rangeBetween(long lowerBound, long upperBound) override;
    BsiUnsigned<uword>* abs() override;
    BsiUnsigned<uword>* abs(int resultSlices,const HybridBitmap<uword> &EB) override;
    BsiUnsigned<uword>* absScale(double range) override;
    BsiVector<uword>* negate() override;
    BsiVector<uword>* multiplyByConstant(int number)const override;
    BsiVector<uword>* multiplyByConstantNew(int number) const override;
    BsiVector<uword>* multiplication(BsiVector<uword> *a)const override;
    BsiVector<uword>* multiplyWithBsiHorizontal(const BsiVector<uword> *a, int precision) const;
    void multiplicationInPlace(BsiVector<uword> *a) override;
    long sumOfBsi()const override;
    bool append(long value) override;
    int compareTo(BsiVector<uword> *a, int index) override;
    
    /*
     Declaring Other Functions
     */
    void addSliceWithOffset(HybridBitmap<uword> slice, int sliceOffset);
    BsiVector<uword>* SUMunsigned(BsiVector<uword>* a)const;
    BsiVector<uword>* SUMsigned(BsiVector<uword>* a);
    BsiVector<uword>* SUMsignToMagnitude(BsiVector<uword>* a)const;
    BsiVector<uword>* SUMsignMagnitude(BsiVector<uword>* a)const;
    BsiVector<uword>* SUMtwosComplement(BsiVector<uword>* a);
    void twosToSignMagnitude(BsiVector<uword>* a)const;
    BsiVector<uword>* multiplyWithBsiHorizontal(const BsiVector<uword> *a) const;
    void multiply(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans) const;
    void BitWords(std::vector<uword> &bitWords, long value, int offset);
    void appendBitWords(long value);
    void multiply_array(uword a[],int size_a, uword b[], int size_b, uword ans[], int size_ans)const;
    BsiVector<uword>* multiplyWithBsiHorizontal_array(const BsiVector<uword> *a) const;
    BsiVector<uword>* multiplication_Horizontal_Hybrid(const BsiVector<uword> *a) const;
    BsiVector<uword>* multiplication_Horizontal_Verbatim(const BsiVector<uword> *a) const;
    BsiVector<uword>* multiplication_Horizontal_compressed(const BsiVector<uword> *a) const;
    BsiVector<uword>* multiplication_Horizontal_Hybrid_other(const BsiVector<uword> *a) const;
    BsiVector<uword>* multiplication_Horizontal(const BsiVector<uword> *a) const;
    BsiVector<uword>* multiplication_array(BsiVector<uword> *a)const override;
    BsiVector<uword>* multiplyBSI(BsiVector<uword> *unbsi)const override;
    long dotProduct(BsiVector<uword>* unbsi) const override;
    long long int dot(BsiVector<uword>* unbsi) const override;
    long long int dot_withoutCompression(BsiVector<uword>* unbsi) const override;
    
    
    BsiVector<uword>* sum_Horizontal_Hybrid(const BsiVector<uword> *a) const;
    BsiVector<uword>* sum_Horizontal_Verbatim(const BsiVector<uword> *a) const;
    BsiVector<uword>* sum_Horizontal_compressed(const BsiVector<uword> *a) const;
    BsiVector<uword>* sum_Horizontal_Hybrid_other(const BsiVector<uword> *a) const;
    BsiVector<uword>* sum_Horizontal(const BsiVector<uword> *a) const;
    void sum(uword a[],int size_a, uword b[], int size_b, uword ans[], int size_ans)const;
    
    
    
    void multiplyKaratsuba(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const;
    void sumOfWordsKaratsuba(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const;
    void subtractionOfWordsKaratsuba(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const;
    void twosComplimentKaratsuba(std::vector<uword> &a)const;
    void combineWordsKaratsuba(std::vector<uword> &a, std::vector<uword> &b,std::vector<uword> &c, std::vector<uword> &d, std::vector<uword> &ac, std::vector<uword> &bd, std::vector<uword> &ans)const;
    void shiftLeftKaratsuba(std::vector<uword> &a, int offset)const;
    void makeEqualLengthKaratsuba(std::vector<uword> &a, std::vector<uword> &b)const;
    void removeZeros(std::vector<uword> &a)const;
    ~BsiSigned();
};



template <class uword>
BsiSigned<uword>::~BsiSigned(){
    
};


//------------------------------------------------------------------------------------------------------

/*
 Defining Constructors
 */

template <class uword>
BsiSigned<uword>::BsiSigned() {
    this->numSlices = 0;
    this->bsi.reserve(BsiVector<uword>::bits);
    this->is_signed =true;
}

template <class uword>
BsiSigned<uword>::BsiSigned(int maxSize) {
    this->numSlices = 0;
    this->bsi.reserve(maxSize);
    this->is_signed =true;
}

/**
 *
 * @param maxSize - maximum number of slices allowed for this attribute
 * @param numOfRows - The number of rows (tuples) in the attribute
 */
template <class uword>
BsiSigned<uword>::BsiSigned(int maxSize, int numOfRows) {
    this->numSlices = 0;
    this->is_signed =true;
    this->bsi.reserve(maxSize);
    this->rows = numOfRows;
}

/**
 *
 * @param maxSize - maximum number of slices allowed for this attribute
 * @param numOfRows - The number of rows (tuples) in the attribute
 */
template <class uword>
BsiSigned<uword>::BsiSigned(int maxSize, int numOfRows, long partitionID) {
    this->numSlices = 0;
    this->is_signed =true;
    this->bsi.reserve(maxSize);
    this->index=partitionID;
    this->rows = numOfRows;
}

/**
 *
 * @param maxSize - maximum number of slices allowed for this attribute
 * @param numOfRows - The number of rows (tuples) in the attribute
 * @param partitionID - the id of the partition
 * @param ex - existence bitmap
 */

template <class uword>
BsiSigned<uword>::BsiSigned(int maxSize, long numOfRows, long partitionID, HybridBitmap<uword> ex) {
    this->numSlices = 0;
    this->is_signed =true;
    this->bsi.reserve(maxSize);
    this->existenceBitmap = ex;
    this->index=partitionID;
    
    this->rows = numOfRows;
}


/*
 Defining Override Functions --------------------------------------------------------------------------------------
 */


/**
 * Computes the top-K tuples in a bsi-attribute.
 * @param k - the number in top-k
 * @return a bitArray containing the top-k tuples
 *
 * TokMAx is Compatible with bsi's SignMagnitude Form not Two'sComplement form
 */
template <class uword>
HybridBitmap<uword> BsiSigned<uword>::topKMax(int k){
    
    HybridBitmap<uword> topK, SE, X;
    HybridBitmap<uword> G;
    G.addStreamOfEmptyWords(false, this->existenceBitmap.sizeInBits()/64);
    HybridBitmap<uword> E = this->existenceBitmap.andNot(this->sign); //considers only positive values
    int n = 0;
    for (int i = this->numSlices - 1; i >= 0; i--) {
        SE = E.And(this->bsi[i]);
        X = SE.Or(G);
        n = X.numberOfOnes();
        if (n > k) {
            E = SE;
        }
        if (n < k) {
            G = X;
            E = E.andNot(this->bsi[i]);
        }
        if (n == k) {
            E = SE;
            break;
        }
    }
    if(n<k){
        //todo add negative numbers here (topKMin abs)
    }
    n = G.numberOfOnes();
    topK = G.Or(E);
    return topK;
};

/*
 * topKMin used for find k min values from bsi and return postions bitmap. NOT IMPLEMENTED YET
 */

template <class uword>
HybridBitmap<uword> BsiSigned<uword>::topKMin(int k){

    
    HybridBitmap<uword> h;
    std::cout<<k<<std::endl;
    return h;
};

/*
 * sumOfBsi perform sum vertically on bsi
 */

template <class uword>
long BsiSigned<uword>::sumOfBsi() const{
    long sum =0, minusSum=0;
    //    int power = 1;
    for (int i=0; i< this->getNumberOfSlices(); i++){
        sum += this->getSlice(i).numberOfOnes()<<(i);
    }
    for (int i=0; i< this->getNumberOfSlices(); i++){
        minusSum += this->getSlice(i).And(this->sign).numberOfOnes()<<(i);
    }
    return sum - 2*minusSum;
}


template <class uword>
BsiVector<uword>* BsiSigned<uword>::SUM(BsiVector<uword>* a) {
    //return sum_Horizontal(a);
    if (a->is_signed and a->twosComplement){
        return this->SUMtwosComplement(a);
    }else if(a->is_signed){
        return this->SUMsignToMagnitude(a);
    }
    else{
        return this->SUMunsigned(a);
    }
};


/*
 * add value to every number in BSI
 */


template <class uword>
BsiVector<uword>* BsiSigned<uword>::SUM(long a)const{
    
    uword abs_a = std::abs(a);
    int intSize =  BsiVector<uword>::sliceLengthFinder(abs_a);
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.addStreamOfEmptyWords(false,this->existenceBitmap.bufferSize());
    BsiVector<uword>* res=new BsiSigned<uword>(std::max((int)this->numSlices, intSize) + 1);
    
    HybridBitmap<uword> C;
    //int minSP = std::min(this->numSlices, intSize); was not used
    HybridBitmap<uword> allOnes;
    allOnes.setSizeInBits(this->bsi[0].sizeInBits());
    allOnes.density=1;
    if ((a&1)==0){
        res->bsi[0]=this->bsi[0];
        C = zeroBitmap;
    }
    else{
        res->bsi[0]=this->bsi[0].Not();
        C=this->bsi[0];
    }
    res->numSlices++;
    int i;
    for(i=1;i<this->numSlices; i++){
        if((a&(1<<i))!=0){
            res->bsi[i]=C.logicalxornot(this->bsi[i]);
            //res.bsi[i] = C.xor(this.bsi[i].NOT());
            C=this->bsi[i].logicalor(C);
        }else{
            res->bsi[i]=this->bsi[i].logicalxor(C);
            C=this->bsi[i].logicaland(C);
        }
        res->numSlices++;
    }
    if(intSize>this->numSlices){
        while (i<intSize){
            if((a&(1<<i))!=0){
                res->bsi[i]=C.logicalxornot(this->bsi[this->numSlices - 1]);
                C=this->bsi[this->numSlices - 1].logicalor(C);
            }else{
                res->bsi[i]=C.logicalxor(this->bsi[this->numSlices - 1]);
                C=this->bsi[this->numSlices - 1].logicaland(C);
            }
            res->numSlices++;
            i++;
        }
    }
    if(this->lastSlice && C.numberOfOnes()>0 ){
        if(a>0){
            res->addSlice(this->sign.logicalandnot(C));
        }else{
            res->addSlice(this->XOR(C,allOnes,this->sign));
        }
    }else{
        res->addSlice(C);
    }
    res->sign = res->bsi[res->numSlices - 1];
    res->firstSlice=this->firstSlice;
    res->lastSlice=this->lastSlice;
    res->existenceBitmap = this->existenceBitmap;
    res->twosComplement=false;
    return res;
};


/*
 * convertToTwos converting SignMagnitude to Two'sComplement
 */

template <class uword>
BsiVector<uword>* BsiSigned<uword>::convertToTwos(int bits){
    BsiSigned res(bits);
    
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.addStreamOfEmptyWords(false,this->existenceBitmap.bufferSize());
    int i=0;
    for(i=0; i<this->getNumberOfSlices(); i++){
        res.addSlice(this->bsi[i].logicalxor(this->sign));
    }
    while(i<bits){
        res.addSlice(this->sign);
        i++;
    }
    res.addSliceWithOffset(this->sign,0);
    res.setTwosFlag(true);
    
    BsiVector<uword>* ans = &res;
    return ans;
};

/*
 * getValue: fetches the decimal value on position @pos of a bsiVector
 */
template <class uword>
long BsiSigned<uword>::getValue(int pos) const{
    if(this->twosComplement){ // not implemented for compressed
        bool sign = this->bsi[this->numSlices - 1].get(pos);
        long sum=0;
        //HybridBitmap<uword> B_i;
        for (int j = 0; j < this->numSlices - 1; j++) {
            //B_i = this->bsi[j];
            if(this->bsi[j].get(pos)^sign)
                sum =sum|( 1<<(this->offset + j));
        }
        return (sum+((sign)?1:0))*((sign)?-1:1);
    }else{
        long sign = (this->sign.get(pos))?-1:1;
        long sum = 0;
        for (int j = 0; j < this->numSlices; j++) {
            if(this->bsi[j].get(pos))
                sum += 1<<(this->offset + j);
            /*if (this->bsi[j].isVerbatim()) {
                if(this->bsi[j].get(i))
                    sum += 1<<(this->offset + j);
            } else {
                ConstRunningLengthWord<uword> rlwa(this->bsi[j].buffer[0]);
                int bit = 0;
                int word = 0;
                while (word*sizeof(uword) < i) {
                    word += (int) rlwa.getRunningLength();
                    if (word*sizeof(uword) > i) {
                        bit = rlwa.getRunningBit();
                    } else {
                        if (i/sizeof(uword) >= word+rlwa.getNumberOfLiteralWords()) {
                            word += rlwa.getNumberOfLiteralWords();
                        } else {
                            bit = this->bsi[j].buffer[i/sizeof(uword)+1] & (1 << (i%sizeof(uword)));
                            word += rlwa.getNumberOfLiteralWords();
                        }
                    }
                }
                if(bit)
                    sum += 1<<(this->offset + j);
            }*/
        }
        return sum*sign;
    }
};

/*
 * Provides values between range in position bitmap: - not implemented yet
 */

template <class uword>
HybridBitmap<uword> BsiSigned<uword>::rangeBetween(long lowerBound, long upperBound){
    //this needs to be implemented
    HybridBitmap<uword> h;
    std::cout<<"lower bound is: "<< lowerBound <<"  uper bound is: "<< upperBound << std::endl;
    return h;
};

/*
 * abs is for converting bsi to two'sComplement to magnitude
 */
template <class uword>
BsiUnsigned<uword>* BsiSigned<uword>::abs(){
    BsiUnsigned<uword>* res = new BsiUnsigned<uword>(this->numSlices);
    if(this->twosComplement){
        for (int i=0; i< this->numSlices - 1; i++){
            res->bsi[i]=this->bsi[i].logicalxor(this->sign);
            res->numSlices++;
        }
        if(this->firstSlice){
            res->addOneSliceSameOffset(this->sign);
        }
    }else{
        res->bsi=this->bsi;
        res->numSlices=this->numSlices;
    }
    res->existenceBitmap=this->existenceBitmap;
    res->setPartitionID(this->getPartitionID());
    res->firstSlice=this->firstSlice;
    res->lastSlice=this->lastSlice;
    return res;
};


template <class uword>
BsiUnsigned<uword>* BsiSigned<uword>::abs(int resultSlices,const HybridBitmap<uword> &EB){
    //number of slices allocated for the result; Existence bitmap
    //    HybridBitmap zeroBitmap = new HybridBitmap();
    //    zeroBitmap.setSizeInBits(this.bsi[0].sizeInBits(),false);
    int min = std::min(this->numSlices - 1, resultSlices);
    BsiUnsigned<uword>* res = new BsiUnsigned<uword>(min+1);
    
    if(this->twosComplement){
        for (int i=0; i<min; i++){
            res->bsi[i]=this->bsi[i].Xor(this->sign);
        }
        res->numSlices=min;
        if(this->firstSlice){
            res->addOneSliceDiscardCarry(this->sign);
        }
    }else{
        for(int i=0;i<min; i++){
            res->bsi[i]=this->bsi[i];
        }
        res->numSlices=min;
    }
    res->addSlice(EB.Not()); // this is for KNN to add one slice
    res->existenceBitmap=this->existenceBitmap;
    res->setPartitionID(this->getPartitionID());
    res->firstSlice=this->firstSlice;
    res->lastSlice=this->lastSlice;
    return res;
};


template <class uword>
BsiUnsigned<uword>* BsiSigned<uword>::absScale(double range){
    HybridBitmap<uword> penalty = this->bsi[this->numSlices - 2].Xor(this->sign);
    
    int resSize=0;
    for (int i= this->numSlices - 2; i >= 0; i--){
        penalty=penalty.logicalor(this->bsi[i].Xor(this->sign));
        if(penalty.numberOfOnes()>=(this->bsi[0].sizeInBits()*range)){
            resSize=i;
            break;
        }
    }
    
    BsiUnsigned<uword>* res = new BsiUnsigned<uword>(2);
    
    res->addSlice(penalty);
    res->existenceBitmap=this->existenceBitmap;
    res->setPartitionID(this->getPartitionID());
    res->firstSlice=this->firstSlice;
    res->lastSlice=this->lastSlice;
    return res;
};

/*
 * Compares values stored in slice "index".
 * Returns -1 if this is less than a, 1 if this is greater than a, 0 otherwise
*/
template <class uword>
int BsiSigned<uword>::compareTo(BsiVector<uword> *a, int index) {
    if (this->sign.get(index) == 1 && a->sign.get(index) == 0) return -1;
    if (this->sign.get(index) == 0 && a->sign.get(index) == 1) return 1;
    if (this->twosComplement && this->sign.get(index) == 1) {
        if (this->numSlices < a->numSlices) {
            for (int i= a->numSlices - 1; i >= this->numSlices; i--) {
                if (a->bsi[i].get(index) == 1) {
                    return -1;
                }
            }
        } else if (this->numSlices > a->numSlices) {
            for (int i= this->numSlices - 1; i >= a->numSlices; i--) {
                if (this->bsi[i].get(index) == 1) {
                    return 1;
                }
            }
        }
        for (int i= std::min(this->numSlices, a->numSlices) - 1; i >= 0; i--) {
            if (this->bsi[i].get(index) != a->bsi[i].get(index)) {
                if (this->sign.get(index) == this->bsi[i].get(index)) return 1;
                else return -1;
            }
        }
    } else {
        if (this->numSlices < a->numSlices) {
            for (int i= a->numSlices - 1; i >= this->numSlices; i--) {
                if (a->bsi[i].get(index) == 1) {
                    if (a->sign.get(index) == 0) return -1;
                    else return 1;
                }
            }
        } else if (this->numSlices > a->numSlices) {
            for (int i= this->numSlices - 1; i >= a->numSlices; i--) {
                if (this->bsi[i].get(index) == 1) {
                    if (this->sign.get(index) == 0) return 1;
                    else return -1;
                }
            }
        }
        for (int i= std::min(this->numSlices, a->numSlices) - 1; i >= 0; i--) {
            if (this->bsi[i].get(index) != a->bsi[i].get(index)) {
                if (this->sign.get(index) == this->bsi[i].get(index)) return -1;
                else return 1;
            }
        }
    }
    return 0;
}


/*
 Defining Other Functions -----------------------------------------------------------------------------------------
 */



template <class uword>
void BsiSigned<uword>::addSliceWithOffset(HybridBitmap<uword> slice, int sliceOffset){
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
    HybridBitmap<uword> A = this->bsi[sliceOffset-this->offset];
    HybridBitmap<uword> C,S;
    
    S=A.Xor(slice);
    C=A.And(slice);
    
    this->bsi[sliceOffset-this->offset]=S;
    int curPos = sliceOffset-this->offset+1;
    
    while(C.numberOfOnes()>0){
        if(curPos<this->numSlices){
            A=this->bsi[curPos];
            S=C.Xor(A);
            C=C.And(A);
            this->bsi[curPos]=S;
            curPos++;
        }else{
            this->addSlice(C);
        }
    }
}


/*
 * SUMsigned was designed for performing sum without sign bits, which is replaced with
 * SUMsignToMagnitude
 */

template <class uword>
BsiVector<uword>* BsiSigned<uword>::SUMunsigned(BsiVector<uword>* a)const{
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
    BsiVector<uword> *res = new BsiSigned();
    res->twosComplement=true;
    res->setPartitionID(a->getPartitionID());
//    if(!this->twosComplement)
//        this->signMagnitudeToTwos(this->numSlices+1);
//    
    int i = 0, s = a->numSlices, p = this->numSlices, aIndex=0, thisIndex=0;
    int minOffset = std::min(a->offset, this->offset);
    res->offset = minOffset;
    
    if(a->offset>this->offset){
        for(int j=0;j<a->offset-minOffset; j++){
            if(j<this->numSlices)
                res->bsi[res->numSlices]=this->bsi[thisIndex];
            else if(this->lastSlice)
                res->bsi[res->numSlices]=this->sign; //sign extend if contains the sign slice
            else
                res->bsi[res->numSlices]=zeroBitmap;
            thisIndex++;
            res->numSlices++;
        }
    }else if(this->offset>a->offset){
        for(int j=0;j<this->offset-minOffset;j++){
            if(j<a->numSlices)
                res->bsi[res->numSlices]=a->bsi[aIndex];
            else
                res->bsi[res->numSlices]=zeroBitmap;
            res->numSlices++;
            aIndex++;
        }
    }
    //adjust the remaining sizes for s and p
    s=s-aIndex;
    p=p-thisIndex;
    int minSP = std::min(s, p);
    
    if(minSP<=0){ // one of the BSI attributes is exausted
        for(int j=thisIndex; j<this->numSlices; j++){
            res->bsi[res->numSlices]=this->bsi[j];
            res->numSlices++;
        }
        HybridBitmap<uword> CC;
        for(int j=aIndex; j<a->numSlices; j++){
            if(this->lastSlice){ // operate with the sign slice if contains the last slice
                if(j==aIndex){
                    res->bsi[res->numSlices]=a->bsi[j].logicalxor(this->sign);
                    CC=a->bsi[j].logicaland(this->sign);
                    res->lastSlice=true;
                }else{
                    res->bsi[res->numSlices]=this->XOR(a->bsi[j], this->sign, CC);
                    CC=this->maj(a->bsi[j],this->sign,CC);
                }
                res->numSlices++;
            }else{
                res->bsi[res->numSlices]=a->bsi[j];
                res->numSlices++;}
        }
        res->lastSlice=this->lastSlice;
        res->firstSlice=this->firstSlice|a->firstSlice;
        res->existenceBitmap = a->existenceBitmap.logicalor(this->existenceBitmap);
        res->sign = &res->bsi[res->numSlices - 1];
        return res;
    }else {
        res->bsi[res->numSlices] = a->bsi[aIndex].logicalxor(this->bsi[thisIndex]);
        HybridBitmap<uword> C = a->bsi[aIndex].logicaland(this->bsi[thisIndex]);
        res->numSlices++;
        thisIndex++;
        aIndex++;
        
        for(i=1; i<minSP; i++){
            //res.bsi[i] = this.bsi[i].xor(a.bsi[i].xor(C));
            res->bsi[res->numSlices] = this->XOR(a->bsi[aIndex], this->bsi[thisIndex], C);
            //res.bsi[i] = this.bsi[i].xor(this.bsi[i], a.bsi[i], C);
            C= this->maj(a->bsi[aIndex], this->bsi[thisIndex], C);
            res->numSlices++;
            thisIndex++;
            aIndex++;
        }
        
        if(s>p){
            for(i=p; i<s;i++){
                res->bsi[res->numSlices] = this->bsi[thisIndex].Xor(C);
                C=this->bsi[thisIndex].logicaland(C);
                res->numSlices++;
                thisIndex++;
            }
        }else{
            for(i=s; i<p;i++){
                if(this->lastSlice){
                    res->bsi[res->numSlices] = this->XOR(a->bsi[aIndex], this->sign, C);
                    C = this->maj(a->bsi[aIndex], this->sign, C);
                    res->numSlices++;
                    aIndex++;}
                else{
                    res->bsi[res->numSlices] = a->bsi[aIndex].Xor(C);
                    C = a->bsi[aIndex].logicaland(C);
                    res->numSlices++;
                    aIndex++;}
            }
        }
        if(!this->lastSlice && C.numberOfOnes()>0){
            res->bsi[res->numSlices]= C;
            res->numSlices++;
        }
        
        res->lastSlice=this->lastSlice;
        res->firstSlice=this->firstSlice|a->firstSlice;
        res->existenceBitmap = a->existenceBitmap.logicalor(this->existenceBitmap);
        res->sign = &res->bsi[res->numSlices - 1];
        return res;
    }
};

/*
 * SUMsigned was designed for performing sum with sign bits, which is replaced with SUMsignToMagnitude
 */

template <class uword>
BsiVector<uword>* BsiSigned<uword>::SUMsigned(BsiVector<uword>* a){
    
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
    BsiVector<uword>* res = new BsiSigned();
    res->twosComplement=true;
    res->setPartitionID(a->getPartitionID());
    
    if (!a->twosComplement)
        a->signMagnitudeToTwos(a->numSlices + 1); //plus one for the sign
    if (!this->twosComplement)
        this->signMagnitudeToTwos(this->numSlices + 1); //plus one for the sign
    
    int i = 0, s = a->numSlices, p = this->numSlices, aIndex=0, thisIndex=0;
    int minOffset = std::min(a->offset, this->offset);
    res->offset = minOffset;
    
    if(this->offset>a->offset){
        for(int j=0;j<this->offset-minOffset; j++){
            if(j<a->numSlices)
                res->bsi[res->numSlices]=a->bsi[aIndex];
            else if(a->lastSlice)
                res->bsi[res->numSlices]=a->sign; //sign extend if contains the sign slice
            else
                res->bsi[res->numSlices]=zeroBitmap;
            aIndex++;
            res->numSlices++;
        }
    }else if(a->offset>this->offset){
        for(int j=0;j<a->offset-minOffset;j++){
            if(j<this->numSlices)
                res->bsi[res->numSlices]=this->bsi[thisIndex];
            else if(this->lastSlice)
                res->bsi[res->numSlices]=this->sign;
            else
                res->bsi[res->numSlices]=zeroBitmap;
            res->numSlices++;
            thisIndex++;
        }
    }
    //adjust the remaining sizes for s and p
    s=s-aIndex;
    p=p-thisIndex;
    int minSP = std::min(s, p);
    
    if(minSP<=0){ // one of the BSI attributes is exausted
        HybridBitmap<uword> CC;
        for(int j=aIndex; j<a->numSlices; j++){
            if(this->lastSlice){ // operate with the sign slice if contains the last slice
                if(j==aIndex){
                    res->bsi[res->numSlices]=a->bsi[j].logicalxor(this->sign);
                    CC=a->bsi[j].logicaland(this->sign);
                    res->lastSlice=true;
                }else{
                    res->bsi[res->numSlices]=this->XOR(a->bsi[j], this->sign, CC);
                    CC=this->maj(a->bsi[j],this->sign,CC);
                }
                res->numSlices++;
            }else{
                res->bsi[res->numSlices]=a->bsi[j];
                res->numSlices++;}
        }
        //CC = NULL;
        for(int j=thisIndex; j<this->numSlices; j++){
            if(a->lastSlice){ // operate with the sign slice if contains the last slice
                if(j==thisIndex){
                    res->bsi[res->numSlices]=this->bsi[j].Xor(a->sign);
                    CC=this->bsi[j].logicaland(a->sign);
                    res->lastSlice=true;
                }else{
                    res->bsi[res->numSlices]=this->XOR(this->bsi[j], a->sign, CC);
                    CC=this->maj(this->bsi[j],a->sign,CC);
                }
                res->numSlices++;
            }else{
                res->bsi[res->numSlices]=this->bsi[j];
                res->numSlices++;}
        }
        
        res->lastSlice=this->lastSlice;
        res->firstSlice=this->firstSlice|a->firstSlice;
        res->existenceBitmap = a->existenceBitmap.logicalor(this->existenceBitmap);
        res->sign = &res->bsi[res->numSlices - 1];
        return res;
    }else {
        
        res->bsi[res->numSlices] = this->bsi[thisIndex].logicalxor(a->bsi[aIndex]);
        HybridBitmap<uword> C = this->bsi[thisIndex].logicaland(a->bsi[aIndex]);
        res->numSlices++;
        thisIndex++;
        aIndex++;
        
        for(i=1; i<minSP; i++){
            //res.bsi[i] = this.bsi[i].xor(a.bsi[i].xor(C));
            res->bsi[res->numSlices] = this->XOR(this->bsi[thisIndex], a->bsi[aIndex], C);
            //res.bsi[i] = this.bsi[i].xor(this.bsi[i], a.bsi[i], C);
            C= this->maj(this->bsi[thisIndex], a->bsi[aIndex], C);
            res->numSlices++;
            thisIndex++;
            aIndex++;
        }
        
        if(s>p){
            for(i=p; i<s;i++){
                if(this->lastSlice){
                    res->bsi[res->numSlices] = this->XOR(a->bsi[aIndex], this->sign, C);
                    C = this->maj(a->bsi[aIndex], this->sign, C);
                    res->numSlices++;
                    aIndex++;}
                res->bsi[res->numSlices] = a->bsi[aIndex].logicalxor(C);
                C=a->bsi[aIndex].logicaland(C);
                res->numSlices++;
                aIndex++;
            }
        }else{
            for(i=s; i<p;i++){
                if(a->lastSlice){
                    res->bsi[res->numSlices] = this->XOR(this->bsi[thisIndex], a->sign, C);
                    C = this->maj(this->bsi[thisIndex], a->sign, C);
                    res->numSlices++;
                    thisIndex++;}
                else{
                    res->bsi[res->numSlices] = this->bsi[thisIndex].Xor(C);
                    C = this->bsi[thisIndex].logicaland(C);
                    res->numSlices++;
                    thisIndex++;}
            }
        }
        
        if(!this->lastSlice&&!a->lastSlice && C.numberOfOnes()>0){
            res->bsi[res->numSlices]= C;
            res->numSlices++;
        }
        res->sign = this->sign;
        res->existenceBitmap = this->existenceBitmap.logicalor(a->existenceBitmap);
        res->lastSlice=a->lastSlice;
        res->firstSlice=this->firstSlice|a->firstSlice;
        res->setNumberOfRows(this->getNumberOfRows());
        return res;
    }
};



/*
 *  SUMsignToMagnitude takes bsiAttribute as signeMagnitude form perform sumation operation and
 *  return bsiAttribute as signeMagnitude.
 *  signeMagnitude: sign bit is stored separate and only magnitude is stored in bsi.
 */

template <class uword>
BsiVector<uword>* BsiSigned<uword>::SUMsignToMagnitude(BsiVector<uword>* a) const{
    BsiVector<uword> *res = new BsiSigned<uword>();
    if(a->twosComplement or this->twosComplement){
        return res;
    }
    if(this->getNumberOfRows() != a->getNumberOfRows()){
        return res;
    }
    
    int maxSlices = this->getNumberOfSlices() > a->getNumberOfSlices() ? this->getNumberOfSlices() : a->getNumberOfSlices();
    int minSlices = this->getNumberOfSlices() < a->getNumberOfSlices() ? this->getNumberOfSlices() : a->getNumberOfSlices();
    
    HybridBitmap<uword> S;
    HybridBitmap<uword> C = this->sign.xorVerbatim(a->sign);   // Initialize carry with 1 where sign bit is one
    HybridBitmap<uword> signAndBitmap = this->sign.andVerbatim(a->sign);
    HybridBitmap<uword> slice, aSlice,thisSign, aSign;
    
    thisSign = this->sign.xorVerbatim(signAndBitmap);   // Calculating sign bit for Two's compliment
    aSign = a->sign.xorVerbatim(signAndBitmap);
    
    for(int i=0; i<minSlices; i++){
        slice = thisSign.Xor(this->bsi[i]); // converting slice into two's compliment
        aSlice = aSign.Xor(a->bsi[i]);      // converting slice into two's compliment
        S = this->XOR(slice, aSlice, C);
        HybridBitmap<uword> temp = aSlice.And(C);
        HybridBitmap<uword> temp2 = aSlice.And(slice);
        C = slice.And(aSlice).Or(slice.And(C)).Or(aSlice.And(C));
        res->addSlice(S);
    }
    if(this->getNumberOfSlices() == minSlices){
        for(int i=minSlices; i<maxSlices; i++){
            slice = thisSign;
            aSlice = aSign.Xor(a->bsi[i]);
            S = this->XOR(slice, aSlice, C);
            C = slice.And(aSlice).Or(slice.And(C)).Or(aSlice.And(C));
            res->addSlice(S);
        }
    
    }else{
        for(int i=minSlices; i<maxSlices; i++){
            slice =  thisSign.Xor(this->bsi[i]);
            aSlice = aSign;
            S = this->XOR(slice, aSlice, C);
            C = slice.And(aSlice).Or(slice.And(C)).Or(aSlice.And(C));
            res->addSlice(S);
        }
    }
    HybridBitmap<uword> signOrBitmap = thisSign.orVerbatim(aSign);
    
//    if(onlyPositiveNumbersCarry.numberOfOnes() >0){
//        res->addSlice(onlyPositiveNumbersCarry);
//    }
    res->addSlice(this->XOR(thisSign, aSign, C));
    //res->sign = res->bsi[res->bsi.numSlices()-1].andVerbatim(signOrBitmap);
    res->is_signed = true;
    res->setNumberOfRows(this->getNumberOfRows());
    //res->twosComplement = true;
    res->twosComplement = false;
    twosToSignMagnitude(res);   // Converting back to Sign to magnitude form
    //res->sign = res->sign.Or(signAndBitmap);
    return res;
};

/*
 *  SUMsignToMagnitude takes bsiAttribute as signeMagnitude form perform sumation operation and
 *  return bsiAttribute as signeMagnitude.
 *  signeMagnitude: sign bit is stored separate and only magnitude is stored in bsi.
 */

template <class uword>
BsiVector<uword>* BsiSigned<uword>::SUMsignMagnitude(BsiVector<uword>* a) const{
    BsiVector<uword> *res = new BsiSigned<uword>();
    if(a->twosComplement or this->twosComplement){
        return res;
    }
    if(this->getNumberOfRows() != a->getNumberOfRows()){
        return res;
    }
    
    int maxSlices = this->getNumberOfSlices() > a->getNumberOfSlices() ? this->getNumberOfSlices() : a->getNumberOfSlices();
    int minSlices = this->getNumberOfSlices() < a->getNumberOfSlices() ? this->getNumberOfSlices() : a->getNumberOfSlices();
    
    HybridBitmap<uword> S;
    HybridBitmap<uword> C = this->sign.Xor(a->sign);
    HybridBitmap<uword> slice, aSlice;
    
    for(int i=0; i<minSlices; i++){
        slice = this->sign.Xor(this->bsi[i]); // converting slice into two's compliment
        aSlice = a->sign.Xor(a->bsi[i]);      // converting slice into two's compliment
        S = this->XOR(slice, aSlice, C);
        C = slice.And(aSlice).Or(slice.And(C)).Or(aSlice.And(C));
        res->addSlice(S);
    }
    if(this->getNumberOfSlices() == minSlices){
        for(int i=minSlices; i<maxSlices; i++){
            slice = this->sign;
            aSlice = a->sign.Xor(a->bsi[i]);
            S = this->XOR(slice, aSlice, C);
            C = slice.And(aSlice).Or(slice.And(C)).Or(aSlice.And(C));
            res->addSlice(S);
        }
    
    }else{
        for(int i=minSlices; i<maxSlices; i++){
            slice =  this->sign.Xor(this->bsi[i]);
            aSlice = a->sign;
            S = this->XOR(slice, aSlice, C);
            C = slice.And(aSlice);
            res->addSlice(S);
        }
    }
    
    res->addSlice(this->XOR(this->sign, a->sign, C));
    res->is_signed = true;
    res->setNumberOfRows(this->getNumberOfRows());
    twosToSignMagnitude(res);   // Converting back to Sign to magnitude form
    res->twosComplement = false;
    return res;
};

/*
 *  SUMsignToMagnitude takes bsiAttribute as twos complement form perform sumation operation and
 *  return bsiAttribute as twos complement.
 */

template <class uword>
BsiVector<uword>* BsiSigned<uword>::SUMtwosComplement(BsiVector<uword>* a) {
    BsiVector<uword> *res = new BsiSigned<uword>();
    if(this->getNumberOfRows() != a->getNumberOfRows()){
        return res;
    }
    if(!a->twosComplement) {
        a->signMagnitudeToTwos(a->getNumberOfSlices()+1);
    }
    if (!this->twosComplement){
        this->signMagnitudeToTwos(a->getNumberOfSlices()+1);
    }
    
    int maxSlices = this->getNumberOfSlices() > a->getNumberOfSlices() ? this->getNumberOfSlices() : a->getNumberOfSlices();
    int minSlices = this->getNumberOfSlices() < a->getNumberOfSlices() ? this->getNumberOfSlices() : a->getNumberOfSlices();
    
    HybridBitmap<uword> S;
    HybridBitmap<uword> C;
    C.addStreamOfEmptyWords(false, this->existenceBitmap.sizeInBits()/64);
    
    for(int i=0; i<minSlices; i++){
        S = this->XOR(this->bsi[i], a->bsi[i], C);
        C = this->bsi[i].And(a->bsi[i]).Or(this->bsi[i].And(C)).Or(a->bsi[i].And(C));
        res->addSlice(S);
    }
    if(this->getNumberOfSlices() == minSlices){
        for(int i=minSlices; i<maxSlices; i++){
            S = this->XOR(this->sign, a->bsi[i], C);
            C = this->sign.And(a->bsi[i]).Or(this->sign.And(C)).Or(a->bsi[i].And(C));
            res->addSlice(S);
        }
    
    }else{
        for(int i=minSlices; i<maxSlices; i++){
            S = this->XOR(this->bsi[i], a->sign, C);
            C = this->bsi[i].And(a->sign).Or(this->bsi[i].And(C)).Or(a->sign.And(C));
            res->addSlice(S);
        }
    }
    
    res->is_signed = true;
    res->setNumberOfRows(this->getNumberOfRows());
    res->twosComplement = true;
    return res;
};

/*
 *  twosToSignMagnitude is converting Two'sCompliment form into signeMagnitude form
 */
template <class uword>
void  BsiSigned<uword>::twosToSignMagnitude(BsiVector<uword>* a) const{
    a->sign = a->bsi[a->getNumberOfSlices()-1];
    HybridBitmap<uword> C = a->sign;
    HybridBitmap<uword> S,slice;
    for(size_t i=0; i< a->bsi.size(); i++){
        slice = a->sign.Xor(a->bsi[i]);
        S = slice.Xor(C);//a->sign.Xor(a->bsi[i]).Xor(C);
        C = slice.And(C);
        a->bsi[i] = S;
    }
    
    a->bsi.pop_back();
    a->setNumberOfSlices(a->bsi.size());
};

/*
 *  negate the sign bit
 */
template <class uword>
BsiVector<uword>* BsiSigned<uword>::negate(){
    BsiVector<uword>* res = new BsiSigned<uword>();
    res->bsi = this->bsi;
    res->sign = this->sign.Not();
    res->is_signed = true;
    res->twosComplement = false;
    res->setNumberOfRows(this->getNumberOfRows());
    return res;
};


/**
 * Multiplies the BsiVector by a constant(Booth's Algorithm)
 * @param number - the constant number
 * @return - the result of the multiplication
 */

template <class uword>
BsiVector<uword>* BsiSigned<uword>::multiplyByConstantNew(int number)const {
    BsiSigned<uword>* res = nullptr;
    /*
    *  HybridBitmap<uword> C, S;
    return res;
    int k = 0;
    while (number > 0) {
        if ((number & 1) == 1) {
            //If the last bit is set
            if (res == nullptr) {
                //Initialize the result
                res = new BsiSigned();
                res->offset = k;
                for (int i = 0; i < this.numSlices; i++) {
                    res.bsi[i] = this.bsi[i];
                }
                res->numSlices = this.numSlices;
                k = 0;
            }
            else {
            }
        }
        number >>= 1;
        k++;    
    }
    */
    return res;
   
};


template <class uword>
BsiVector<uword>* BsiSigned<uword>::multiplyByConstant(int number) const {
    //The result
    BsiSigned<uword>* res = nullptr;
    HybridBitmap<uword> C, S;
    bool isNegative = false;
    int k = 0;
    if(number < 0){
        isNegative = true;
        number = 0-number;
    }else if(number == 0){
        res = new BsiSigned<uword>();
        HybridBitmap<uword> zeroBitmap;
        zeroBitmap.reset();
        zeroBitmap.verbatim = true;
        int bufferLength = (this->getNumberOfRows()/(this->bits))+1;
        for (int i=0; i<bufferLength; i++){
            zeroBitmap.buffer.push_back(0);
        }
        zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
        res->offset = 0;
        for (int i = 0; i < this->numSlices; i++) {
            res->bsi.push_back(zeroBitmap);
        }
        res->numSlices = this->numSlices;
        res->sign = zeroBitmap;
        res->rows = this->rows;
        res->index = this->index;
        res->is_signed = true;
        res->twosComplement = false;
        res->setNumberOfRows(this->getNumberOfRows());
        return res;
    }
    
    
    while (number > 0) {
        if ((number & 1) == 1) {
            if (res == nullptr) {
                res = new BsiSigned<uword>();
                res->offset = k;
                for (int i = 0; i < this->numSlices; i++) {
                    res->bsi.push_back(this->bsi[i]);
                }
                res->numSlices = this->numSlices;
                k = 0;
            } else {
                /* Move the slices of res k positions */
                HybridBitmap<uword> A, B;
                if (k >= res->numSlices) {
                    A = new HybridBitmap<uword>(1,false);
                } else {
                    A = res->bsi[k];
                }
                B = this->bsi[0];
                S = A.Xor(B);
                C = A.And(B);
                if (k >= res->numSlices) {
                    HybridBitmap<uword> zeroBitmap;
                    zeroBitmap.addStreamOfEmptyWords(false, this->existenceBitmap.bufferSize());
                    //zeroBitmap.reset();
                    //zeroBitmap.verbatim = true;
                    zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits(), false);
                    for (int i = res->numSlices; i < k; i++) {
                        res->bsi.push_back(zeroBitmap);
                        res->numSlices ++;
                    }
                    res->numSlices ++;
                    res->bsi.push_back(S);
                } else {
                    res->bsi[k] = S;
                }
                for (int i = 1; i < this->numSlices; i++) {// Add the slices of this to the current res
                    B = this->bsi[i];
                    if ((i + k) >=res->numSlices){
                        S = B.Xor(C);
                        C = B.And(C);
                        res->numSlices++;
                        res->bsi.push_back(S);
                        continue;
                    } else {
                        A = res->bsi[i + k];
                        S = A.Xor(B).Xor(C);
                        C = A.And(B).Or(B.And(C)).Or(A.And(C));
                    }
                    res->bsi[i + k] = S;
                }
                for (int i = this->numSlices + k; i < res->numSlices; i++) {// Add the remaining slices of res with the Carry C
                    A = res->bsi[i];
                    S = A.Xor(C);
                    C = A.And(C);
                    res->bsi[i] = S;
                }
                if (C.numberOfOnes() > 0) {
                    res->bsi.push_back(C); // Carry bit
                    res->numSlices++;
                }
            }
        }/*else{
            if (res == nullptr) {
                res = new BsiSigned<uword>();
                HybridBitmap<uword> zeroBitmap;
                zeroBitmap.addStreamOfEmptyWords(false, this->existenceBitmap.bufferSize());
                //zeroBitmap.reset();
                //zeroBitmap.verbatim = true;
                zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits(), false);
                for (int i = 0; i < this->numSlices; i++) {
                    res->bsi.push_back(zeroBitmap);
                }
                res->numSlices = this->numSlices;
                k = 0;
            }
        }*/
        number >>= 1;
        k++;
    }
    res->existenceBitmap = this->existenceBitmap;
    if(isNegative){
        res->sign = this->sign.Not();
        
    }else{
        res->sign = this->sign;
    }
    res->rows = this->rows;
    res->index = this->index;
    res->is_signed = true;
    res->twosComplement = false;
    res->setNumberOfRows(this->getNumberOfRows());
    return res;
};

/*
 */

template <class uword>
BsiVector<uword>*  BsiSigned<uword>::multiplyWithBsiHorizontal(const BsiVector<uword> *bsi, int precision) const{
    int precisionInBits = 3*precision +1;
    BsiSigned<uword>* res = nullptr;
    res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.reset();
    hybridBitmap.verbatim = true;
    for(int j=0; j< this->numSlices + bsi->numSlices; j++){
        res->addSlice(hybridBitmap);
    }
    int size_a = this->numSlices;
    int size_b = bsi->numSlices;
    std::vector<uword> a(size_a);
    std::vector<uword> b(size_b);
    std::vector<uword> answer(size_a + size_b);

    for(int i=0; i< this->bsi[0].bufferSize(); i++){
        for(int j=0; j< this->numSlices; j++){
            a[j] = this->bsi[j].getWord(i); //fetching one word
        }
        for(int j=0; j< bsi->numSlices; j++){
            b[j] = bsi->bsi[j].getWord(i);
        }
        this->multiply(a,b,answer);         //perform multiplication on one word
//        this->multiplyBSI(a);         //perform multiplication on one word
//        this->multiplyWithBSI(b);         //perform multiplication on one word

        for(int j=0; j< answer.size() ; j++){
            res->bsi[j].addVerbatim(answer[j]);
        }
    }
    for(int j=0; j< res->numSlices; j++){
        int temp = res->bsi[j].numberOfOnes();
        res->bsi[j].density = res->bsi[j].numberOfOnes()*1.0/res->bsi[j].sizeInBits();
    }
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->is_signed = true;
    res->sign = this->sign.Xor(bsi->sign);
    return res;
};

/*
 * multiply_array perform multiplication at word level
 * word from every bitmap of Bsi is multiplied with other bsi's word
 * it is modified version of Booth's Algorithm
 */

template <class uword>
void BsiSigned<uword>:: multiply_array(uword a[],int size_a, uword b[], int size_b, uword ans[], int size_ans)const{
    uword S=0,C=0,FS;

    int k=0, ansSize=0;
    for(int i=0; i<size_a; i++){  // Initialing with first bit operation
        ans[i] = a[i] & b[0];
    }
    for(int i = size_a; i< size_a + size_b; i++){ // Initializing rest of bits to zero
        ans[i] = 0;
    }
    k=1;
    ansSize = size_a;
    for(int it=1; it<size_b; it++){
        S = ans[k]^a[0];
        C = ans[k]&a[0];
        FS = S & b[it];
        ans[k] = (~b[it] & ans[k]) | (b[it] & FS); // shifting Operation
        
        for(int i=1; i<size_a; i++){
            int t = i+k;
            if(t < ansSize){
                S = ans[t] ^ a[i] ^ C;
                C = (ans[t]&a[i]) | (a[i]&C) | (ans[t]&C);
            }else{
                S = a[i] ^ C;
                C = a[i] & C;
                FS = S & b[it];
                ansSize++;
                ans[ansSize - 1] = FS;
            }
            FS = b[it] & S;
            ans[i + k ] =(~b[it] & ans[t]) | (b[it] & FS); // shifting Operation
        }
        for(int i=size_a + k; i< ansSize; i++){
            S = ans[i] ^ C;
            C = ans[i] & C;
            FS = b[it] & S;
            ans[k] = (~b[it] & ans[k]) | (b[it] & FS); // shifting Operation
        }
        if(C>0){
            ansSize++;
            ans[ansSize-1] = b[it] & C;
        }
        k++;
    }
    for(int t=ansSize; t<size_ans; t++){
        ans[t] = 0;
    }
};


/*
 * multiplyWithBsiHorizontal_array perform multiplication betwwen bsi using multiply_array
 * only support verbatim Bsi
 */

template <class uword>
BsiVector<uword>*  BsiSigned<uword>::multiplyWithBsiHorizontal_array(const BsiVector<uword> *a) const{
    //    int precisionInBits = 3*precision + (int)std::log2(precision);
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        res->sign = this->sign.Xor(a->sign);
        res->is_signed = true;
        res->twosComplement = false;
        return res;
    }
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.reset();
    hybridBitmap.verbatim = true;
    for(int j=0; j< this->numSlices + a->numSlices; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->numSlices;
    int size_y = a->numSlices;
    int size_ans = size_y +size_x;
    uword x[size_x];
    uword y[size_y];
    uword answer[size_ans];
    
    for(int i=0; i< this->bsi[0].bufferSize(); i++){
        for(int j=0; j< size_x; j++){
            x[j] = this->bsi[j].getWord(i);
        }
        for(int j=0; j< size_y; j++){
            y[j] = a->bsi[j].getWord(i);
        }
        this->multiply_array(x,size_x,y, size_y,answer, size_ans);
        for(int j=0; j< size_ans ; j++){
            res->bsi[j].addVerbatim(answer[j]);
        }
    }
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
};


/*
 * multiplyWithBsiHorizontal_array perform multiplication betwwen bsi using multiply_array
 * support both verbatim and compressed Bsi(using existenceBitmap)
 */

template <class uword>
BsiVector<uword>* BsiSigned<uword>::multiplication_Horizontal(const BsiVector<uword> *a) const{
    if(!this->existenceBitmap.isVerbatim() and !a->existenceBitmap.isVerbatim()){
       return this->multiplication_Horizontal_compressed(a);
    }else if (this->existenceBitmap.isVerbatim() or a->existenceBitmap.isVerbatim()){
        if(this->existenceBitmap.verbatim){
            return this->multiplication_Horizontal_Hybrid_other(a);
        }else{
            return this->multiplication_Horizontal_Hybrid(a);
        }
    }else{
        return this->multiplication_Horizontal_Verbatim(a);
    }
    
}

/*
 * multiplication_Horizontal_compressed perform multiplication betwwen bsi using multiply_array
 * only support compressed Bsi(using existenceBitmap)
 */


template <class uword>
BsiVector<uword>* BsiSigned<uword>::multiplication_Horizontal_compressed(const BsiVector<uword> *a) const{
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        res->sign = this->sign.Xor(a->sign);
        res->is_signed = true;
        res->twosComplement = false;
        return res;
    }
    
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.setSizeInBits(this->existenceBitmap.sizeInBits());
    for(int j=0; j< this->numSlices + a->numSlices; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->numSlices;
    int size_y = a->numSlices;
    int size_ans = size_y +size_x;
    uword* x = new uword[size_x];
    uword* y = new uword[size_y];
    uword* answer = new uword[size_ans];
    //uword x[size_x];
    //uword y[size_y];
    //uword answer[size_ans];
    
    HybridBitmapRawIterator<uword> iterator = this->existenceBitmap.raw_iterator();
    HybridBitmapRawIterator<uword> a_iterator = a->existenceBitmap.raw_iterator();
    BufferedRunningLengthWord<uword> &rlwi = iterator.next();
    BufferedRunningLengthWord<uword> &rlwa = a_iterator.next();
    
    int position = 0;
    int literal_counter = 1;
    int positionNext = 0;
    int a_literal_counter = 1;
    int a_position = 0;
    int a_positionNext = 0;
    while (rlwi.size() > 0 and rlwa.size() > 0) {
        position = positionNext;
        a_position = a_positionNext;
        while ((rlwi.getRunningLength() > 0) || (rlwa.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwa.getRunningLength();
            BufferedRunningLengthWord<uword> &prey(i_is_prey ? rlwi : rlwa);
            BufferedRunningLengthWord<uword> &predator(i_is_prey ? rlwa : rlwi);
            if (!predator.getRunningBit()) {
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].fastaddStreamOfEmptyWords(false, predator.getRunningLength());
                }
                if(i_is_prey){
                    if(rlwi.getNumberOfLiteralWords() < rlwa.getRunningLength()){
                        position = position+rlwi.getNumberOfLiteralWords()+1;
                        literal_counter =  1;
                    }else{
                        literal_counter += rlwa.getRunningLength() - rlwi.getRunningLength();
                    }
                }else{
                    if(rlwa.getNumberOfLiteralWords() < rlwi.getRunningLength()){
                        a_position = a_position+rlwa.getNumberOfLiteralWords()+1;
                        a_literal_counter = 1;
                    }else{
                        a_literal_counter += rlwi.getRunningLength() - rlwa.getRunningLength();
                    }
                }
                 prey.discardFirstWordsWithReload(predator.getRunningLength());
            }
            predator.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwa.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                for(int j=0; j< size_x; j++){
                    x[j] = this->bsi[j].getWord(position+literal_counter);
                }
                for(int j=0; j< size_y; j++){
                    y[j] = a->bsi[j].getWord(a_position+a_literal_counter);
                }
                this->multiply_array(x,size_x,y, size_y,answer, size_ans);
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].addLiteralWord(answer[j]);
                }
                literal_counter++;
                a_literal_counter++;
            }
            if(rlwi.getNumberOfLiteralWords() == nbre_literal){
                positionNext = position+nbre_literal+1;
                literal_counter = 1;
            }
            if(rlwa.getNumberOfLiteralWords() == nbre_literal){
                a_positionNext = a_position + nbre_literal+1;
                a_literal_counter = 1;
            }
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwa.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
}


/*
 * multiplication_Horizontal_Verbatim perform multiplication betwwen bsi using multiply_array
 * only support verbatim Bsi(using existenceBitmap)
 */


template <class uword>
BsiVector<uword>* BsiSigned<uword>::multiplication_Horizontal_Verbatim(const BsiVector<uword> *a) const{
    
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        res->sign = this->sign.Xor(a->sign);
        res->is_signed = true;
        res->twosComplement = false;
        return res;
    }
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.reset();
    hybridBitmap.verbatim = true;
    for(int j=0; j< this->numSlices + a->numSlices; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->numSlices;
    int size_y = a->numSlices;
    int size_ans = size_y +size_x;
    uword* x = new uword[size_x];
    uword* y = new uword[size_y];
    uword* answer = new uword[size_ans];
    //uword x[size_x];
    //uword y[size_y];
    //uword answer[size_ans];
    
    for(size_t i=0; i< this->bsi[0].bufferSize(); i++){
        for(int j=0; j< size_x; j++){
            x[j] = this->bsi[j].getWord(i);
        }
        for(int j=0; j< size_y; j++){
            y[j] = a->bsi[j].getWord(i);
        }
        this->multiply_array(x,size_x,y, size_y,answer, size_ans);
        for(int j=0; j< size_ans ; j++){
            res->bsi[j].addVerbatim(answer[j]);
        }
    }
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
}


/*
 * multiplication_Horizontal_Hybrid perform multiplication betwwen bsi using multiply_array
 * only support hybrid Bsis(one is verbatim and one is compressed)
 */

template <class uword>
BsiVector<uword>* BsiSigned<uword>::multiplication_Horizontal_Hybrid(const BsiVector<uword> *a) const{

    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        res->sign = this->sign.Xor(a->sign);
        res->is_signed = true;
        res->twosComplement = false;
        return res;
    }

    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.setSizeInBits(this->existenceBitmap.sizeInBits());
    for(int j=0; j< this->numSlices + a->numSlices; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->numSlices;
    int size_y = a->numSlices;
    int size_ans = size_y +size_x;
    uword* x = new uword[size_x];
    uword* y = new uword[size_y];
    uword* answer = new uword[size_ans];
    //uword x[size_x];
    //uword y[size_y];
    //uword answer[size_ans];
    
    HybridBitmapRawIterator<uword> i = this->existenceBitmap.raw_iterator();
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    
    int positionOfCompressed = 1;
    int positionOfVerbatim = 0;
    while ( rlwi.size() > 0) {
        while (rlwi.getRunningLength() > 0) {
            positionOfVerbatim += rlwi.getRunningLength();
            for(int j=0; j< size_ans ; j++){
                res->bsi[j].addStreamOfEmptyWords(0,rlwi.getRunningLength());
            }
            if(rlwi.getNumberOfLiteralWords() == 0){
                positionOfCompressed++;
            }
            rlwi.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = rlwi.getNumberOfLiteralWords();
        if (nbre_literal > 0) {
            for (size_t k = 1; k <= nbre_literal; ++k) {
                for(int j=0; j< size_x; j++){
                    x[j] = this->bsi[j].getWord(positionOfCompressed +k);
                }
                for(int j=0; j< size_y; j++){
                    y[j] = a->bsi[j].getWord(positionOfVerbatim);
                }
                this->multiply_array(x,size_x,y, size_y,answer, size_ans);
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].addLiteralWord(answer[j]);
                }
                positionOfVerbatim++;
            }
        }
        positionOfCompressed += rlwi.getNumberOfLiteralWords()+1;
        rlwi.discardLiteralWordsWithReload(nbre_literal);
        
    }

    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
};



template <class uword>
BsiVector<uword>* BsiSigned<uword>::multiplication_Horizontal_Hybrid_other(const BsiVector<uword> *a) const{
    
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        res->sign = this->sign.Xor(a->sign);
        res->is_signed = true;
        res->twosComplement = false;
        return res;
    }
    
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    for(int j=0; j< this->numSlices + a->numSlices; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->numSlices;
    int size_y = a->numSlices;
    int size_ans = size_y +size_x;
    uword* x = new uword[size_x];
    uword* y = new uword[size_y];
    uword* answer = new uword[size_ans];
    //uword x[size_x];
    //uword y[size_y];
    //uword answer[size_ans];
    
    HybridBitmapRawIterator<uword> i = a->existenceBitmap.raw_iterator();
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    
    int positionOfCompressed = 0;
    int positionOfVerbatim = 0;
    while ( rlwi.size() > 0) {
        while (rlwi.getRunningLength() > 0) {
            positionOfVerbatim += rlwi.getRunningLength();
            for(int j=0; j< size_ans ; j++){
                res->bsi[j].addStreamOfEmptyWords(0,rlwi.getRunningLength());
            }
            if(rlwi.getNumberOfLiteralWords() == 0){
                positionOfCompressed++;
            }
            rlwi.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = rlwi.getNumberOfLiteralWords();
        if (nbre_literal > 0) {
            for (size_t k = 1; k <= nbre_literal; ++k) {
                for(int j=0; j< size_x; j++){
                    x[j] = this->bsi[j].getWord(positionOfVerbatim);
                }
                for(int j=0; j< size_y; j++){
                    y[j] = a->bsi[j].getWord(positionOfCompressed + k);
                }
                this->multiply_array(x,size_x,y, size_y,answer, size_ans);
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].addLiteralWord(answer[j]);
                }
                positionOfVerbatim++;
            }
        }
        positionOfCompressed += rlwi.getNumberOfLiteralWords()+1;
        rlwi.discardLiteralWordsWithReload(nbre_literal);
    }
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
};


/*
 * multiply is modified Booth's algorithm for multiplication design for
 * vertical multiplication of 64-bits at a time same as multiply_array
 */

template <class uword>
void BsiSigned<uword>:: multiply(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const{
    uword S=0,C=0,FS;
    //int startPosition = b.numSlices() + a.numSlices() - ans.numSlices();
    int k=0, ansSize=0;
    for(size_t i=0; i<a.size(); i++){  // Initialing with first bit operation
        ans[i] = a[i] & b[0];
    }
    for(size_t i = a.size(); i< b.size() + a.size(); i++){ // Initializing rest of bits to zero
        ans[i] = 0;
    }
    k=1;
    ansSize = a.size();
    for(size_t it=1; it<b.size(); it++){
        S = ans[k]^a[0];
        C = ans[k]&a[0];
        FS = S & b[it];
        ans[k] = (~b[it] & ans[k]) | (b[it] & FS); // shifting Operation
        
        for(size_t i=1; i<a.size(); i++){
            int t = i+k;
            if(t < ansSize){
                S = ans[t] ^ a[i] ^ C;
                C = (ans[t]&a[i]) | (a[i]&C) | (ans[t]&C);
            }else{
                S = a[i] ^ C;
                C = a[i] & C;
                FS = S & b[it];
                ansSize++;
                ans[ansSize - 1] = FS;
            }
            FS = b[it] & S;
            ans[i + k ] =(~b[it] & ans[t]) | (b[it] & FS); // shifting Operation
        }
        for(int i=a.size() + k; i< ansSize; i++){
            S = ans[i] ^ C;
            C = ans[i] & C;
            FS = b[it] & S;
            ans[k] = (~b[it] & ans[k]) | (b[it] & FS); // shifting Operation
        }
        if(C>0){
            ansSize++;
            ans[ansSize-1] = b[it] & C;
        }
        k++;
    }
};



/*
 * multiplyWithBsiHorizontal perform multiplication betwwen bsi using multiply
 * only support verbatim Bsi
 */


template <class uword>
BsiVector<uword>*  BsiSigned<uword>::multiplyWithBsiHorizontal(const BsiVector<uword> *a) const{
//    int precisionInBits = 3*precision + (int)std::log2(precision);
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        res->sign = this->sign.Xor(a->sign);
        res->is_signed = true;
        res->twosComplement = false;
        return res;
    }
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.reset();
    hybridBitmap.verbatim = true;
    for(int j=0; j< this->numSlices + a->numSlices; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->numSlices;
    int size_y = a->numSlices;
    std::vector<uword> x(size_x);
    std::vector<uword> y(size_y);
    std::vector<uword> answer(size_x+size_y);
    
    for(size_t i=0; i< this->bsi[0].bufferSize(); i++){
        for(int j=0; j< size_x; j++){
            x[j] = this->bsi[j].getWord(i);
        }
        for(int j=0; j< size_y; j++){
            y[j] = a->bsi[j].getWord(i);
        }
        
        this->multiply(x,y,answer);
        for(size_t j=0; j< answer.size() ; j++){
            res->bsi[j].addVerbatim(answer[j]);
        }
//        for(int k=answer.numSlices(); k < res->bsi.numSlices(); k++){
//            res->bsi[k].addVerbatim(0);
//        }
        answer.clear();
    }
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed= true;
    res->twosComplement = false;
    return res;
};




/**
 * multiplication the BsiVector by another BsiVector
 * @param a - the other BsiVector
 * @return - the result of the multiplication
 * Only Compatible with verbatim Bitmaps
 */


template <class uword>
BsiVector<uword>* BsiSigned<uword>::multiplication(BsiVector<uword> *a)const{
    
    BsiVector<uword>* res = multiplyWithBsiHorizontal(a);
    int size = res->bsi.size();
    for(int i=0; i< size; i++){
        res->bsi[i].density = res->bsi[i].numberOfOnes()/(double)res->getNumberOfRows();
    }
    return res;
}



template <class uword>
BsiVector<uword>* BsiSigned<uword>::multiplication_array(BsiVector<uword> *a)const{
    
    BsiVector<uword>* res = multiplication_Horizontal(a);
    int size = res->bsi.size();
    for(int i=0; i< size; i++){
        res->bsi[i].density = res->bsi[i].numberOfOnes()/(double)res->getNumberOfRows();
    }
    return res;
}

/*
 * appendBitWords
 */

template <class uword>
void BsiSigned<uword>::appendBitWords(long value){
    const uword one = 1;
    if (this->bsi.size() == 0){
        int i = 0;
        if(value <0){
            value = std::abs(value);
            this->sign.reset();
            this->sign.verbatim = true;
            this->sign.buffer.push_back((uword)1);
            this->sign.setSizeInBits(1);
        }else{
            this->sign.reset();
            this->sign.verbatim = true;
            this->sign.buffer.push_back((uword)0);
            this->sign.setSizeInBits(1);
        }
        if(value == 0){
            HybridBitmap<uword> zeroBitmap(true,1);
            zeroBitmap.buffer[0] = (0);
            this->bsi.push_back(zeroBitmap);
            this->numSlices++;
        }
        while (value > 0){
            HybridBitmap<uword> zeroBitmap(true,1);
            zeroBitmap.buffer[0] = (value & 1);
            this->bsi.push_back(zeroBitmap);
            this->numSlices++;
            value = value/2;
            i++;
        }
        this->rows = 1;
        this->is_signed = true;
        this->twosComplement =false;

    }else{
        int i = 0;
        int size = this->bsi[0].buffer.size() - 1;
        int offset = this->getNumberOfRows()%(BsiVector<uword>::bits);
        if(value <0){
            value = std::abs(value);
            this->sign.buffer[size] = this->sign.buffer.back() | (one << offset);
            this->sign.setSizeInBits(this->sign.sizeInBits()+1);
        }else{
            this->sign.setSizeInBits(this->sign.sizeInBits()+1);
        }
        for(int i=0; i<this->numSlices; i++){
            this->bsi[i].buffer[size] = this->bsi[i].buffer.back() | ((value & one) << offset);
            this->bsi[i].setSizeInBits(this->bsi[i].sizeInBits()+1);
            value = value/2;
        }
        while (value > 0){
            HybridBitmap<uword> zeroBitmap(true,size+1);
            zeroBitmap.buffer[0] = (value & one)<< offset;
            zeroBitmap.setSizeInBits(this->rows+1);
            this->bsi.push_back(zeroBitmap);
            value = value/2;
            i++;
            this->numSlices++;
        }
        this->rows++;
       
    }
}

/*
 * Only Use for verbatim bitslices
 */

template <class uword>
bool BsiSigned<uword>::append(long value){
    /*
     * If bitslices are not verbatime compitable
     */

    appendBitWords(value);
    return true;
    }

/*
 * multiplyKaratsuba performs multiplication using Karatsuba algorithm
 * it is lot slower than multiplication_array
 */
template <class uword>
void BsiSigned<uword>::multiplyKaratsuba(std::vector<uword> &A, std::vector<uword> &B, std::vector<uword> &ans)const{
    makeEqualLengthKaratsuba(A,B);
    if(A.size() <= 1){
        for(int i=0; i<A.size(); i++){
            ans.push_back(A[i] & B[i]);
        }
//        multiply(A, B, ans);
    }else{
        int mid = A.size()/2;
        std::vector<uword> b(A.cbegin(),A.cbegin()+mid);
        std::vector<uword> d(B.cbegin(),B.cbegin()+mid);
        std::vector<uword> c(B.cbegin()+mid,B.cend());
        std::vector<uword> a(A.cbegin()+mid,A.cend());
        std::vector<uword> ac(A.size());
        multiply(a, c, ac);
        removeZeros(ac);
        std::vector<uword> bd(B.size());
        multiply(b, d, bd);
        removeZeros(bd);
        combineWordsKaratsuba(a, b, c, d, ac, bd, ans);
        
    }
    
};


/*
 * Calculate Ans = Ac + (a + b)*(c+d)(1<<2*sh) - (ac + bd)(1<<sh) + bd operation.
 */

template <class uword>
void BsiSigned<uword>::combineWordsKaratsuba(std::vector<uword> &a, std::vector<uword> &b,std::vector<uword> &c, std::vector<uword> &d, std::vector<uword> &ac, std::vector<uword> &bd, std::vector<uword> &ans)const{
    std::vector<uword> a_plus_b;
    sumOfWordsKaratsuba(a,b, a_plus_b);
    std::vector<uword> c_plus_d;
    sumOfWordsKaratsuba(c, d, c_plus_d);
    std::vector<uword> a_plus_b_c_plus_d(a_plus_b.size()+c_plus_d.size());
    multiply(a_plus_b, c_plus_d, a_plus_b_c_plus_d);
    removeZeros(a_plus_b_c_plus_d);
    std::vector<uword> ac_plus_bd;
    sumOfWordsKaratsuba(ac, bd, ac_plus_bd);
    makeEqualLengthKaratsuba(a_plus_b_c_plus_d, ac_plus_bd);
    twosComplimentKaratsuba(ac_plus_bd);
    std::vector<uword> middle_word; // (a + b)*(c+d) - (ac + bd)
    subtractionOfWordsKaratsuba(a_plus_b_c_plus_d, ac_plus_bd, middle_word);
    shiftLeftKaratsuba(ac, a.size()+b.size());
    shiftLeftKaratsuba(middle_word, b.size());
    std::vector<uword> firsthalf;
    sumOfWordsKaratsuba(ac, middle_word, firsthalf);
    sumOfWordsKaratsuba(firsthalf, bd, ans);
    removeZeros(ans);
};

template <class uword>
void BsiSigned<uword>::makeEqualLengthKaratsuba(std::vector<uword> &a, std::vector<uword> &b)const{
    if(a.size() > b.size()){
        for (int i=b.size();i<a.size(); i++){
            b.push_back(0);
        }
    }else if (b.size() > a.size()){
        for (int i=a.size();i<b.size(); i++){
            a.push_back(0);
        }
    }
};

template <class uword>
void BsiSigned<uword>::removeZeros(std::vector<uword> &a)const{
    while (a.back() == 0) {
        a.pop_back();
    }
}

template <class uword>
void BsiSigned<uword>::sumOfWordsKaratsuba(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const{
    makeEqualLengthKaratsuba(a,b);
    uword carry = 0;
    for(int i=0; i<a.size(); i++){
        ans.push_back(a[i] ^ b[i] ^ carry);
        carry = (a[i] & b[i]) | (a[i] & carry) | (b[i] & carry);
    }
    if(carry != 0){
        ans.push_back(carry);
    }
};

template <class uword>
void BsiSigned<uword>::subtractionOfWordsKaratsuba(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const{
    uword carry = 0;
    for(int i=0; i<a.size(); i++){
        ans.push_back(a[i] ^ b[i] ^ carry);
        carry = (a[i] & b[i]) | (a[i] & carry) | (b[i] & carry);
    }
};

template <class uword>
void BsiSigned<uword>::twosComplimentKaratsuba(std::vector<uword> &a)const{
    uword carry = ~0;
    for(int i=0; i<a.size(); i++){
        uword ans = ~a[i] ^ carry;
        carry = (~a[i] & carry);
        a[i] = ans;
    }

};

template <class uword>
void BsiSigned<uword>::shiftLeftKaratsuba(std::vector<uword> &a, int offset)const{
    std::vector<uword> ans;
    
    for(int i=0; i<offset; i++){
        ans.push_back(0);
        a.push_back(0);
    }
    for(int i=0; i < a.size() - offset; i++){
        ans.push_back(a[i]);
    }
    for (int i=0; i< ans.size(); i++){
        a[i] = ans[i];
    }
};


/*
 * multiplicationInPlace perfom a *= b using modified booth's algorithm
 */


template <class uword>
void BsiSigned<uword>::multiplicationInPlace(BsiVector<uword> *a){
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        return;
    }
    HybridBitmap<uword> hybridBitmap(true,this->bsi[0].buffer.size());
    hybridBitmap.verbatim = true;
    int size = this->numSlices;
    for(int j=size; j< size + a->numSlices; j++){
        this->addSlice(hybridBitmap);
    }
    int size_x = size;
    int size_y = a->numSlices;
    int size_ans = size_y +size_x;
    uword   * x = new uword[size_x];
    uword   * y = new uword[size_y];
    uword   * answer = new uword[size_ans];
    //uword x[size_x];
    //uword y[size_y];
    //uword answer[size_ans];
    
    for(size_t i=0; i< this->bsi[0].bufferSize(); i++){
        for(int j=0; j< size_x; j++){
            x[j] = this->bsi[j].getWord(i);
        }
        for(int j=0; j< size_y; j++){
            y[j] = a->bsi[j].getWord(i);
        }
        this->multiply_array(x,size_x,y, size_y,answer, size_ans);
        for(int j=0; j< size_ans ; j++){
            this->bsi[j].setWord(i,answer[j]);
        }
    }
    this->numSlices = this->bsi.size();
    for(int i=0; i<this->numSlices; i++){
        this->bsi[i].density = this->bsi[i].numberOfOnes()/(double)this->getNumberOfRows();
    }
    while(this->bsi.back().numberOfOnes() == 0){
        this->bsi.pop_back();
    }
    this->numSlices = this->bsi.size();
    
}



/*
 * multiplicationInPlace perfom a = b * c using modified booth's algorithm
 */
template <class uword>
long BsiSigned<uword>::dotProduct(BsiVector<uword>* unbsi) const {
    return 1;
}

/*
 * dot perfom a = b dot c
 */
template <class uword>
long long int BsiSigned<uword>::dot(BsiVector<uword>* unbsi) const {
    long long int res =0;
    HybridBitmap<uword> signNegative;
    signNegative = this->sign.Xor(unbsi->sign);
    HybridBitmap<uword> signPositive;
    signPositive = signNegative.Not();
    for(int j=0; j<unbsi->numSlices; j++){
        for (int i = 0; i < this->numSlices; i++) {
            if(j==0 && i==0) {  //first iteration
                    res = res - unbsi->bsi[j].And(this->bsi[i]).And(signNegative).numberOfOnes();
                    res = res + unbsi->bsi[j].And(this->bsi[i]).And(signPositive).numberOfOnes();
            }
            else {
                res = res - unbsi->bsi[j].And(this->bsi[i]).And(signNegative).numberOfOnes() * (2 << (j + i - 1));
                res = res + unbsi->bsi[j].And(this->bsi[i]).And(signPositive).numberOfOnes() * (2 << (j + i - 1));
            }
            }
        }
    return res;
    }

template <class uword>
long long int BsiSigned<uword>::dot_withoutCompression(BsiVector<uword>* unbsi) const {
    long long int res =0;
    HybridBitmap<uword> signNegative;
    signNegative = this->sign.xorVerbatim(unbsi->sign);
    HybridBitmap<uword> signPositive;
    signPositive = signNegative.Not();
    for(int j=0; j<unbsi->numSlices; j++){
        for (int i = 0; i < this->numSlices; i++) {
            if(j==0 && i==0) {  //first iteration
                    res = res - unbsi->bsi[j].andVerbatim(this->bsi[i]).andVerbatim(signNegative).numberOfOnes();
                    res = res + unbsi->bsi[j].andVerbatim(this->bsi[i]).andVerbatim(signPositive).numberOfOnes();
            }
            else {
                res = res - unbsi->bsi[j].andVerbatim(this->bsi[i]).andVerbatim(signNegative).numberOfOnes() * (2 << (j + i - 1));
                res = res + unbsi->bsi[j].andVerbatim(this->bsi[i]).andVerbatim(signPositive).numberOfOnes() * (2 << (j + i - 1));
            }
            }
        }
    return res;
    }


template <class uword>
BsiVector<uword>* BsiSigned<uword>::multiplyBSI(BsiVector<uword> *a) const{
    BsiVector<uword>* res = nullptr;
    HybridBitmap<uword> C, S, FS, DS;
    int k = 0;
    res = new BsiSigned<uword>();
    res->offset = k;
    for (int i = 0; i < this->numSlices; i++) {
        res->bsi.push_back(a->bsi[0].And(this->bsi[i]));
    }
    res->numSlices = this->numSlices;
    k = 1;
    for (int it=1; it<a->numSlices; it++) {
        /* Move the slices of res k positions */
        S=res->bsi[k];
        //S = S.Xor(this->bsi[0]);
        S.XorInPlace(this->bsi[0]);
        C = res->bsi[k].And(this->bsi[0]);
        FS = a->bsi[it].And(S);
        //res->bsi[k] = a->bsi[it].Not().And(res->bsi[k]).Or(a->bsi[it].And(FS)); // shifting operation
        res->bsi[k].selectMultiplicationInPlace(a->bsi[it],FS);
        
        for (int i = 1; i < this->numSlices; i++) {// Add the slices of this to the current res
            if ((i + k) < res->numSlices){
                //A = res->bsi[i + k];
                S = res->bsi[i + k];
                //S = S.Xor(this->bsi[i]);
                //S = S.Xor(C);
                S.XorInPlace(this->bsi[i]);
                S.XorInPlace(C);
                //C = res->bsi[i + k].And(this->bsi[i]).Or(this->bsi[i].And(C)).Or(res->bsi[i + k].And(C));
                C.majInPlace(res->bsi[i + k],this->bsi[i]);
                
            } else {
                S=this->bsi[i];
                //S = S.Xor(C);
                //C = C.And(this->bsi[i]);
                S.XorInPlace(C);
                C.AndInPlace(this->bsi[i]);
                res->numSlices++;
                FS = a->bsi[it].And(S);
                res->bsi.push_back(FS);
            }
            FS = a->bsi[it].And(S);
            //res->bsi[i + k] = res->bsi[i + k].andNot(a->bsi[it]).Or(a->bsi[it].And(FS)); // shifting operation
            res->bsi[i+k].selectMultiplicationInPlace(a->bsi[it],FS);
        }
        for (int i = this->numSlices + k; i < res->numSlices; i++) {// Add the remaining slices of res with the Carry C
            S = res->bsi[i];
            //S = S.Xor(C);
            //C = C.And(res->bsi[i]);
            S.XorInPlace(C);
            C.AndInPlace(res->bsi[i]);
            FS = a->bsi[it].And(S);
            //res->bsi[k] = a->bsi[it].Not().And(res->bsi[k]).Or(a->bsi[it].And(FS)); // shifting operation
            res->bsi[k].selectMultiplicationInPlace(a->bsi[it],FS);
        }
        if (C.numberOfOnes() > 0) {
            res->bsi.push_back(a->bsi[it].And(C)); // Carry bit
            res->numSlices++;
        }
        k++;
    }
    
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
};

/*
 * sum perform sum at word level
 * word from bitmap of one bsi is added to others
 */

template <class uword>
void BsiSigned<uword>::sum(uword a[],int size_a, uword b[], int size_b, uword ans[], int size_ans)const{
    uword carry = 0;
    bool is_a_big = size_a > size_b;
    if(is_a_big){
        for(int i=0; i<size_b; i++){
            ans[i] = (a[i] ^ b[i] ^ carry);
            carry = (a[i] & b[i]) | (a[i] & carry) | (b[i] & carry);
        }
        for(int i=size_b; i<size_a; i++){
            ans[i] = (a[i] ^ carry);
            carry = (a[i] & carry);
        }
            ans[size_ans - 1] = carry;
    }else{
        for(int i=0; i<size_a; i++){
            ans[i] = (a[i] ^ b[i] ^ carry);
            carry = (a[i] & b[i]) | (a[i] & carry) | (b[i] & carry);
        }
        for(int i=size_a; i<size_b; i++){
            ans[i] = (b[i] ^ carry);
            carry = (b[i] & carry);
        }
            ans[size_ans - 1] = carry;
    }
}


/*
 * sum_Horizontal_Hybrid perform summation between two bsi using sum method
 * only support hybrid bsi (one verbatim and one compressed)
 * a is compressed
 */



template <class uword>
BsiVector<uword>* BsiSigned<uword>::sum_Horizontal_Hybrid(const BsiVector<uword> *a) const{
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        if(this->bsi.size() ==0){
            for(size_t i=0; i<a->bsi.size(); i++){
                res->addSlice(a->bsi[i]);
            }
            res->existenceBitmap = a->existenceBitmap;
            res->rows = a->rows;
            res->index = a->index;
            res->sign = a->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }else{
            for(size_t i=0; i<this->bsi.size(); i++){
                res->addSlice(this->bsi[i]);
            }
            res->existenceBitmap = this->existenceBitmap;
            res->rows = this->rows;
            res->index = this->index;
            res->sign = this->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }
    }
    
    int size_x = this->numSlices;
    int size_y = a->numSlices;
    int size_ans = std::max(size_y,size_x) + 1;
    uword* x = new uword[size_x];
    uword* y = new uword[size_y];
    uword* answer = new uword[size_ans];
    //uword x[size_x];
    //uword y[size_y];
    //uword answer[size_ans];
    
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap(true,0);
    hybridBitmap.setSizeInBits(a->existenceBitmap.sizeInBits());
    for(int j=0; j< size_ans; j++){
        res->addSlice(hybridBitmap);
    }
    HybridBitmapRawIterator<uword> i = this->existenceBitmap.raw_iterator();
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    
    int positionOfCompressed = 1;
    int positionOfVerbatim = 0;
    while ( rlwi.size() > 0) {
        while (rlwi.getRunningLength() > 0) {
            for(size_t k=0; k<rlwi.getRunningLength(); k++){
                /*
                 * directly adding non zero values to ans if one has zeros
                 */
                for(int j=0; j< size_y; j++){
                    res->bsi[j].addVerbatim(a->bsi[j].buffer[positionOfVerbatim]);
                }
                for(int j=size_y; j< size_ans; j++){
                    res->bsi[j].addVerbatim(0L);
                }
                positionOfVerbatim++;
            }
            
            rlwi.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = rlwi.getNumberOfLiteralWords();
        if (nbre_literal > 0) {
            /*
             * if both have non-zero values
             */
            for (size_t k = 0; k < nbre_literal; ++k) {
                for(int j=0; j< size_x; j++){
                    x[j] = this->bsi[j].getWord(positionOfCompressed);
                }
                for(int j=0; j< size_y; j++){
                    y[j] = a->bsi[j].getWord(positionOfVerbatim);
                }
                this->sum(x,size_x,y, size_y,answer, size_ans);
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].addVerbatim(answer[j]);
                }
                positionOfCompressed++;
                positionOfVerbatim++;
            }
        }
        rlwi.discardLiteralWordsWithReload(nbre_literal);
    }
    
    res->existenceBitmap = a->existenceBitmap;
    res->rows = a->rows;
    res->index = a->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
}


/*
 * sum_Horizontal_Verbatim perform summation between two bsi using sum method
 * only support verbatim bsis
 */

template <class uword>
BsiVector<uword>* BsiSigned<uword>::sum_Horizontal_Verbatim(const BsiVector<uword> *a) const{

    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        if(this->bsi.size() ==0){
            for(size_t i=0; i<a->bsi.size(); i++){
                res->addSlice(a->bsi[i]);
            }
            res->existenceBitmap = a->existenceBitmap;
            res->rows = a->rows;
            res->index = a->index;
            res->sign = a->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }else{
            for(size_t i=0; i<this->bsi.size(); i++){
                res->addSlice(this->bsi[i]);
            }
            res->existenceBitmap = this->existenceBitmap;
            res->rows = this->rows;
            res->index = this->index;
            res->sign = this->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }
    }
    
    int size_x = this->numSlices;
    int size_y = a->numSlices;
    int size_ans = std::max(size_y,size_x) + 1;
    uword* x = new uword[size_x];
    uword* y = new uword[size_y];
    uword* answer = new uword[size_ans];
    //uword x[size_x];
    //uword y[size_y];
    //uword answer[size_ans];
    
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap(true,0);
    hybridBitmap.setSizeInBits(a->existenceBitmap.sizeInBits());
    for(int j=0; j< size_ans; j++){
        res->addSlice(hybridBitmap);
    }
    for(size_t i=0; i< this->bsi[0].bufferSize(); i++){
        for(int j=0; j< size_x; j++){
            x[j] = this->bsi[j].getWord(i);
        }
        for(int j=0; j< size_y; j++){
            y[j] = a->bsi[j].getWord(i);
        }
        this->sum(x,size_x,y, size_y,answer, size_ans);
        for(int j=0; j< size_ans ; j++){
            res->bsi[j].addVerbatim(answer[j]);
        }
    }
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
}


/*
 * sum_Horizontal_compressed perform summation between two bsi using sum method
 * only support compressed bsis
 */

template <class uword>
BsiVector<uword>* BsiSigned<uword>::sum_Horizontal_compressed(const BsiVector<uword> *a) const{
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        if(this->bsi.size() ==0){
            for(size_t i=0; i<a->bsi.size(); i++){
                res->addSlice(a->bsi[i]);
            }
            res->existenceBitmap = a->existenceBitmap;
            res->rows = a->rows;
            res->index = a->index;
            res->sign = a->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }else{
            for(size_t i=0; i<this->bsi.size(); i++){
                res->addSlice(this->bsi[i]);
            }
            res->existenceBitmap = this->existenceBitmap;
            res->rows = this->rows;
            res->index = this->index;
            res->sign = this->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }
    }
    
    int size_x = this->numSlices;
    int size_y = a->numSlices;
    int size_ans = std::max(size_y,size_x) + 1;
    uword* x = new uword[size_x];
    uword* y = new uword[size_y];
    uword* answer = new uword[size_ans];
    //uword x[size_x];
    //uword y[size_y];
    //uword answer[size_ans];
    
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.setSizeInBits(this->existenceBitmap.sizeInBits());
    for(int j=0; j< size_ans; j++){
        res->addSlice(hybridBitmap);
    }
  
    
    HybridBitmapRawIterator<uword> iterator = this->existenceBitmap.raw_iterator();
    HybridBitmapRawIterator<uword> a_iterator = a->existenceBitmap.raw_iterator();
    BufferedRunningLengthWord<uword> &rlwi = iterator.next();
    BufferedRunningLengthWord<uword> &rlwa = a_iterator.next();
    
   
    
    int position = 0;
    int literal_counter = 1;
    //int positionNext = 0;
    int a_literal_counter = 1;
    int a_position = 0;
    //int a_positionNext = 0;
    while (rlwi.size() > 0 and rlwa.size() > 0) {
        while ((rlwi.getRunningLength() > 0) || (rlwa.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwa.getRunningLength();
            BufferedRunningLengthWord<uword> &prey(i_is_prey ? rlwi : rlwa);
            BufferedRunningLengthWord<uword> &predator(i_is_prey ? rlwa : rlwi);
            if (!prey.getRunningBit() && prey.getRunningLength() >0) {
                // Filling Zeros
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].fastaddStreamOfEmptyWords(false, prey.getRunningLength());
                }
            }
            predator.discardFirstWordsWithReload(prey.getRunningLength());
            prey.discardRunningWordsWithReload();
            if(prey.getNumberOfLiteralWords() >= predator.getRunningLength()){
                for(size_t k=0; k < predator.getRunningLength(); k++){
                    if(i_is_prey){
                        for(int j=0; j< size_x; j++){
                            res->bsi[j].addLiteralWord(this->bsi[j].getWord(position+literal_counter));
                        }
                        for(int j=size_x; j< size_ans; j++){
                            res->bsi[j].addLiteralWord(0);
                        }
                        literal_counter++;
                    }else{
                        for(int j=0; j< size_y; j++){
                            res->bsi[j].addLiteralWord(a->bsi[j].getWord(a_position+a_literal_counter));
                        }
                        for(int j=size_y; j< size_ans; j++){
                            res->bsi[j].addLiteralWord(0);
                        }
                        a_literal_counter++;
                    }
                }
                if(i_is_prey){
                    if(prey.getNumberOfLiteralWords() == predator.getRunningLength()){
                        position +=  literal_counter;
                        literal_counter = 1;
                    }
                }else{
                    if(prey.getNumberOfLiteralWords() == predator.getRunningLength()){
                        a_position +=  a_literal_counter;
                        a_literal_counter = 1;
                    }
                }
                prey.discardFirstWordsWithReload(predator.getRunningLength());
                predator.discardRunningWordsWithReload();
            }else{
                for(size_t k=0; k < prey.getNumberOfLiteralWords(); k++){
                    if(i_is_prey){
                        for(int j=0; j< size_x; j++){
                            res->bsi[j].addLiteralWord(this->bsi[j].getWord(position+literal_counter));
                        }
                        for(int j=size_x; j< size_ans; j++){
                            res->bsi[j].addLiteralWord(0);
                        }
                        literal_counter++;
                    }else{
                        for(int j=0; j< size_y; j++){
                            res->bsi[j].addLiteralWord(a->bsi[j].getWord(a_position+a_literal_counter));
                        }
                        for(int j=size_y; j< size_ans; j++){
                            res->bsi[j].addLiteralWord(0);
                        }
                        a_literal_counter++;
                    }
                }
                
                if(i_is_prey){
                    position += literal_counter;
                    literal_counter = 1;
                }else{
                    a_position += a_literal_counter;
                    a_literal_counter = 1;
                }
                predator.discardFirstWordsWithReload(prey.getNumberOfLiteralWords());
                prey.discardLiteralWordsWithReload(prey.getNumberOfLiteralWords());
            }
        }
        
        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwa.getNumberOfLiteralWords());
        //To test for 10 numbers addition. All having only one buffer.
        literal_counter = 0;
        a_literal_counter = 0;

        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                for(int j=0; j< size_x; j++){
                    x[j] = this->bsi[j].getWord(position+literal_counter);
                }
                for(int j=0; j< size_y; j++){
                    y[j] = a->bsi[j].getWord(a_position+a_literal_counter);
                }
                this->sum(x,size_x,y, size_y,answer, size_ans);
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].addLiteralWord(answer[j]);
                }
                literal_counter++;
                a_literal_counter++;
            }
            if(rlwi.getNumberOfLiteralWords() == nbre_literal){
                position += nbre_literal+1;
                literal_counter = 1;
            }
            if(rlwa.getNumberOfLiteralWords() == nbre_literal){
                a_position += nbre_literal+1;
                a_literal_counter = 1;
            }
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwa.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
}

/*
 * support method for sum_Horizontal_hybrid
 * a is verbatim
 */

template <class uword>
BsiVector<uword>* BsiSigned<uword>::sum_Horizontal_Hybrid_other(const BsiVector<uword> *a) const{
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        if(this->bsi.size() ==0){
            for(size_t i=0; i<a->bsi.size(); i++){
                res->bsi.push_back(a->bsi[i]);
            }
            res->existenceBitmap = a->existenceBitmap;
            res->rows = a->rows;
            res->index = a->index;
            res->sign = a->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }else{
            for(size_t i=0; i<this->bsi.size(); i++){
                res->bsi.push_back(this->bsi[i]);
            }
            res->existenceBitmap = this->existenceBitmap;
            res->rows = this->rows;
            res->index = this->index;
            res->sign = this->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }
    }
    
    int size_x = this->numSlices;
    int size_y = a->numSlices;
    int size_ans = std::max(size_y,size_x) + 1;
    uword* x = new uword[size_x];
    uword* y = new uword[size_y];
    uword* answer = new uword[size_ans];
    //uword x[size_x];
    //uword y[size_y];
    //uword answer[size_ans];
    
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap(true,0);
    hybridBitmap.setSizeInBits(this->existenceBitmap.sizeInBits());
    for(int j=0; j< size_ans; j++){
        res->addSlice(hybridBitmap);
    }
    
    
    HybridBitmapRawIterator<uword> i = a->existenceBitmap.raw_iterator();
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    
    int positionOfCompressed = 1;
    int positionOfVerbatim = 0;
    while ( rlwi.size() > 0) {
        while (rlwi.getRunningLength() > 0) {
            //            positionOfCompressed ++;
            //            positionOfVerbatim += rlwi.getRunningLength();
            for(size_t k=0; k<rlwi.getRunningLength(); k++){
                for(int j=0; j< size_x; j++){
                    res->bsi[j].addVerbatim(this->bsi[j].buffer[positionOfVerbatim]);
                }
                for(int j=size_x; j< size_ans; j++){
                    res->bsi[j].addVerbatim(0L);
                }
                positionOfVerbatim++;
            }
            
            rlwi.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = rlwi.getNumberOfLiteralWords();
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                //                container.addWord(rlwi.getLiteralWordAt(k));
                for(int j=0; j< size_x; j++){
                    x[j] = this->bsi[j].getWord(positionOfCompressed);
                }
                for(int j=0; j< size_y; j++){
                    y[j] = a->bsi[j].getWord(positionOfVerbatim);
                }
                this->sum(x,size_x,y, size_y,answer, size_ans);
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].addVerbatim(answer[j]);
                }
                positionOfCompressed++;
                positionOfVerbatim++;
            }
        }
        rlwi.discardLiteralWordsWithReload(nbre_literal);
    }
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
}

/*
 * sum_Horizontal perform summation of two bsi using sum method
 */

template <class uword>
BsiVector<uword>* BsiSigned<uword>::sum_Horizontal(const BsiVector<uword> *a) const{
    if(!this->existenceBitmap.isVerbatim() and !a->existenceBitmap.isVerbatim()){
        return this->sum_Horizontal_compressed(a);
    }else if (this->existenceBitmap.isVerbatim() and a->existenceBitmap.isVerbatim()){
         return this->sum_Horizontal_Verbatim(a);

    }else{
        if(this->existenceBitmap.verbatim){
            return this->sum_Horizontal_Hybrid_other(a);
        }else{
            return this->sum_Horizontal_Hybrid(a);
        }
    }
}
#endif /* BsiSigned_hpp */
