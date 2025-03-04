//
//  BsiUnsigned.hpp
//

#ifndef BsiUnsigned_hpp
#define BsiUnsigned_hpp

#include <stdio.h>
#include "BsiAttribute.hpp"
#include <cmath>
#include <cstdint>


template <class uword>
class BsiUnsigned : public BsiAttribute<uword>{
public:
    /*
    Declaring Constructors
     */
    
    BsiUnsigned();
    BsiUnsigned(int maxSize);
    BsiUnsigned(int maxSize, int numOfRows);
    BsiUnsigned(int maxSize, int numOfRows, long partitionID);
    BsiUnsigned(int maxSize, long numOfRows, long partitionID, HybridBitmap<uword> ex);
    
    /*
     Declaring Override Functions
     */
    
    HybridBitmap<uword> topKMax(int k) override;
    HybridBitmap<uword> topKMin(int k) override;
    BsiAttribute<uword>* SUM(BsiAttribute<uword>* a) override;
    BsiAttribute<uword>* SUM(long a)const override;
    BsiAttribute<uword>* convertToTwos(int bits) override;
    long getValue(int pos) const override;
    HybridBitmap<uword> rangeBetween(long lowerBound, long upperBound) override;
    BsiUnsigned<uword>* abs() override;
    BsiUnsigned<uword>* abs(int resultSlices,const HybridBitmap<uword> &EB) override;
    BsiUnsigned<uword>* absScale(double range) override;
    BsiAttribute<uword>* negate() override;
    BsiAttribute<uword>* multiplyByConstant(int number)const override;
    BsiAttribute<uword>* multiplyByConstantNew(int number) const override;
    long dotProduct(BsiAttribute<uword>* unbsi) const override;
    long long int dot(BsiAttribute<uword>* unbsi) const override;
    long long int dot_withoutCompression(BsiAttribute<uword>* unbsi) const override;
    bool append(long value) override;
    int compareTo(BsiAttribute<uword> *a, int index) override;
    
    /*
     * multiplication is only compatible with Verbatim Bitmap
     */
    BsiAttribute<uword>* multiplication(BsiAttribute<uword> *a)const override;
    BsiAttribute<uword>* multiplication_array(BsiAttribute<uword> *a)const override;
    void multiplicationInPlace(BsiAttribute<uword> *a) override;
    void multiply(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const;
    long sumOfBsi()const override;
    HybridBitmap<uword> getExistenceBitmap();

    /*
     * unsigned division implementation
     */
    std::pair<BsiAttribute<uword>*, BsiAttribute<uword>*> divide(
            const BsiAttribute<uword>& dividend,
            const BsiAttribute<uword>& divisor
    ) const override;

    
    /*
     Declaring Other Functions
     */
    
    BsiAttribute<uword>* SUMunsigned(BsiAttribute<uword>* a)const;
    BsiAttribute<uword>* SUMsigned(BsiAttribute<uword>* a);
    BsiAttribute<uword>* SUM(long a, HybridBitmap<uword> EB, int rangeSlices)const;
    BsiAttribute<uword>* sum_Horizontal(const BsiAttribute<uword> *a) const;
    
//    BsiAttribute<uword>* negate();
    BsiAttribute<uword>* multiplyWithBSI(BsiUnsigned &unbsi) const;
    BsiAttribute<uword>* multiplyBSI(BsiAttribute<uword> *unbsi)const override;
    BsiUnsigned<uword>& multiplyWithKaratsuba(BsiUnsigned &unbsi) const;
    BsiAttribute<uword>* multiplyWithBsiHorizontal(const BsiAttribute<uword> *unbsi, int precision) const;
    BsiAttribute<uword>* multiplyWithBsiHorizontal(const BsiAttribute<uword> *unbsi) const;
    BsiUnsigned<uword>* multiplyBSIWithPrecision(const BsiUnsigned<uword> &unbsi, int precision) const;
    void multiply_array(uword a[],int size_a, uword b[], int size_b, uword ans[], int size_ans)const;
    BsiAttribute<uword>* multiplication_Horizontal(const BsiAttribute<uword> *a) const;
    BsiAttribute<uword>* multiplication_Horizontal_Hybrid(const BsiAttribute<uword> *a) const;
    BsiAttribute<uword>* multiplication_Horizontal_Verbatim(const BsiAttribute<uword> *a) const;
    BsiAttribute<uword>* multiplication_Horizontal_compressed(const BsiAttribute<uword> *a) const;
    BsiAttribute<uword>* multiplication_Horizontal_Hybrid_other(const BsiAttribute<uword> *a) const;
    
    //BsiUnsigned<uword>* twosComplement() const;
//    uword sumOfBsi();
    void reset();
    BsiAttribute<uword>* peasantMultiply(BsiUnsigned &unbsi) const;
    int sliceLengthFinder(uword value)const;
    void BitWords(std::vector<uword> &bitWords, long value, int offset);
    
    ~BsiUnsigned();
};



template <class uword>
BsiUnsigned<uword>::~BsiUnsigned(){
    
};

//------------------------------------------------------------------------------------------------------

/*
 Defining Constructors
 */

template <class uword>
BsiUnsigned<uword>::BsiUnsigned() {
    this->size = 0;
    this->bsi.reserve(32);
}

template <class uword>
BsiUnsigned<uword>::BsiUnsigned(int maxSize) {
    this->size = 0;
    this->bsi.reserve(maxSize);
}
/**
 *
 * @param maxSize - maximum number of slices allowed for this attribute
 * @param numOfRows - The number of rows (tuples) in the attribute
 */
template <class uword>
BsiUnsigned<uword>::BsiUnsigned(int maxSize, int numOfRows) {
    this->size = 0;
    this->bsi.reserve(maxSize);
    this->existenceBitmap.setSizeInBits(numOfRows);
    //        if(existenceBitmap.sizeInBits()%64>0)
    //            existenceBitmap.setSizeInBits(existenceBitmap.sizeInBits()+64-existenceBitmap.sizeInBits()%64, false);
    //        existenceBitmap.density = (double)numOfRows/(existenceBitmap.sizeInBits()+64-existenceBitmap.sizeInBits()%64);
    this->existenceBitmap.density=1;
    this->rows = numOfRows;
}

/**
 *
 * @param maxSize - maximum number of slices allowed for this attribute
 * @param numOfRows - The number of rows (tuples) in the attribute
 */
template <class uword>
BsiUnsigned<uword>::BsiUnsigned(int maxSize, int numOfRows, long partitionID) {
    this->size = 0;
    this->bsi.reserve(maxSize);
    this->existenceBitmap.setSizeInBits(numOfRows);
    //        if(existenceBitmap.sizeInBits()%64>0)
    //            existenceBitmap.setSizeInBits(existenceBitmap.sizeInBits()+64-existenceBitmap.sizeInBits()%64, false);
    //        existenceBitmap.density = (double)numOfRows/(existenceBitmap.sizeInBits()+64-existenceBitmap.sizeInBits()%64);
    this->existenceBitmap.density=1;
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
BsiUnsigned<uword>::BsiUnsigned(int maxSize, long numOfRows, long partitionID, HybridBitmap<uword> ex) {
    this->size = 0;
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
 */
template <class uword>
HybridBitmap<uword> BsiUnsigned<uword>::topKMax(int k){
    HybridBitmap<uword> topK, SE, X;
    HybridBitmap<uword> G;
    HybridBitmap<uword> E;
    G.setSizeInBits(this->bsi[0].sizeInBits(),false);
    E.setSizeInBits(this->bsi[0].sizeInBits(),true);
    E.density=1;
    
    int n = 0;
    for (int i = this->size - 1; i >= 0; i--) {
        SE = E.And(this->bsi[i]);
        X = G.Or(SE);
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
    n = G.numberOfOnes();
    topK = G.Or(E);
    // topK = OR(G, E.first(k - n+ 1));
    
    return topK;
};

/**
 * Computes the top-K tuples in a bsi-attribute.
 * @param k - the number in top-k
 * @return a bitArray containing the top-k tuples
 */

template <class uword>
HybridBitmap<uword> BsiUnsigned<uword>::topKMin(int k){
    HybridBitmap<uword> topK, SNOT, X;
    HybridBitmap<uword> G;
    HybridBitmap<uword> E = this->existenceBitmap;
    G.setSizeInBits(this->bsi[0].sizeInBits(),false);
    //E.setSizeInBits(this.bsi[0].sizeInBits(),true);
    //E.density=1;
    int n = 0;
    
    for (int i = this->size - 1; i >= 0; i--) {
        SNOT = E.andNot(this->bsi[i]);
        X = G.Or(SNOT); //Maximum
        n = X.numberOfOnes();
        if (n > k) {
            E = SNOT;
        }
        else if (n < k) {
            G = X;
            E = E.And(this->bsi[i]);
        }
        else {
            E = SNOT;
            break;
        }
    }
    //        n = G.cardinality();
    topK = G.Or(E); //with ties
    // topK = OR(G, E.first(k - n+ 1)); //Exact number of topK
    
    return topK;
};

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::SUM(BsiAttribute<uword>* a){
//    HybridBitmap<uword> zeroBitmap;
//    zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
//    BsiAttribute<uword>* res = new BsiUnsigned<uword>(std::max(this->size+this->offset, a->size+a->offset)+1);
//    res->setPartitionID(a->getPartitionID());
//    res->index = this->index;
//    res->existenceBitmap = this->existenceBitmap.Or(a->existenceBitmap);
//    int i = 0, s = a->size, p = this->size;
//
//
//
//    int minOffset = std::min(a->offset,this->offset);
//    res->offset = minOffset;
//
//    int aIndex = 0;
//    int thisIndex =0;
//
//    if(this->offset > a->offset){
//        for(int j=0; j < this->offset-minOffset; j++){
//            if(j<a->size)
//                res->bsi[res->size]=a->bsi[aIndex];
//            else
//                res->bsi[res->size]=zeroBitmap;
//            aIndex++;
//            res->size++;
//        }
//    }else if(a->offset > this->offset){
//        for(int j=0;j<a->offset-minOffset;j++){
//            if(j<this->size)
//                res->bsi[res->size]=this->bsi[thisIndex];
//            else
//                res->bsi[res->size]=zeroBitmap;
//            res->size++;
//            thisIndex++;
//        }
//    }
//    //adjust the remaining sizes for s and p
//    s=s-aIndex;
//    p=p-thisIndex;
//    int minSP = std::min(s, p);
//
//    if(minSP<=0){ // one of the BSI attributes is exausted
//        for(int j=aIndex; j<a->size;j++){
//            res->bsi[res->size]=a->bsi[j];
//            res->size++;
//        }
//        for(int j=thisIndex; j<this->size;j++){
//            res->bsi[res->size]=this->bsi[j];
//            res->size++;
//        }
//        return res;
//    }else {
//
//        res->bsi[res->size] = this->bsi[thisIndex].Xor(a->bsi[aIndex]);
//        HybridBitmap<uword> C = this->bsi[thisIndex].And(a->bsi[aIndex]);
//        res->size++;
//        thisIndex++;
//        aIndex++;
//
//        for(i=1; i<minSP; i++){
//            //res.bsi[i] = this.bsi[i].xor(a.bsi[i].xor(C));
//            res->bsi[res->size] = this->XOR(this->bsi[thisIndex], a->bsi[aIndex], C);
//            //res.bsi[i] = this.bsi[i].xor(this.bsi[i], a.bsi[i], C);
//            C= this->maj(this->bsi[thisIndex], a->bsi[aIndex], C);
//            res->size++;
//            thisIndex++;
//            aIndex++;
//        }
//
//        if(s>p){
//            for(i=p; i<s;i++){
//                res->bsi[res->size] = a->bsi[aIndex].Xor(C);
//                C=a->bsi[aIndex].And(C);
//                res->size++;
//                aIndex++;
//            }
//        }else{
//            for(i=s; i<p;i++){
//                res->bsi[res->size] = this->bsi[thisIndex].Xor(C);
//                C = this->bsi[thisIndex].And(C);
//                res->size++;
//                thisIndex++;
//            }
//        }
//        //if(!(this.lastSlice && a.lastSlice) && (C.cardinality()>0)){
//        if(C.numberOfOnes()>0){
//            res->bsi[res->size]= C;
//            res->size++;
//        }
//        return res;
//    }
    
    /*if (a->is_signed){
        return SUMsigned(a);
    }else{
        return SUMunsigned(a);
    }*/
    return SUMunsigned(a);
};



template <class uword>
int BsiUnsigned<uword>::sliceLengthFinder(uword value) const{
    //uword mask = 1 << (sizeof(uword) * 8 - 1);
    int lengthCounter =0;
    for(int i = 0; i < sizeof(uword) * 8; i++)
    {
        uword ai = (static_cast<uword>(1) << i);
        if( ( value & (static_cast<uword>(1) << i ) ) != 0 ){
            lengthCounter = i+1;
        }
    }
    return lengthCounter;
}


template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::SUM(long a)const{
        int intSize = sliceLengthFinder(a);
        HybridBitmap<uword> zeroBitmap;
        BsiAttribute<uword>* res;
        zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits(),false);
        HybridBitmap<uword> C;
        if(a<0){
            //int minSP = Math.min(this.size, (intSize+1));
            res = new BsiSigned<uword>(std::max( (int)this->size, (intSize+1) )+1);
            res->twosComplement=true;
            if ((a&1)==0){
                res->bsi[0]=this->bsi[0];
                C = zeroBitmap;
            }
            else{
                res->bsi[0]=this->bsi[0].logicalnot();
                C=this->bsi[0];
            }
            res->size++;
            int i;
            for( i=1; i<this->size; i++ ){
                if((a&(1<<i))!=0){//xorNot(this->bsi[i])
                    res->bsi[i]=C.xorNot(this->bsi[i]);
                    //res.bsi[i] = C.xor(this.bsi[i].NOT());
                    C=this->bsi[i].Or(C);
                }else{
                    res->bsi[i]=this->bsi[i].Xor(C);
                    C=this->bsi[i].And(C);
                }
                res->size++;
                
            }
            if((intSize+1)>this->size){
                while(i<(intSize+1)){
                    if((a&(1<<i))!=0){
                        res->bsi[i]=C.logicalnot();
                        //C=this.bsi[i].or(C);
                    }else{
                        res->bsi[i]=C;
                        C=zeroBitmap;
                    }
                    i++;
                    res->size++;
                }}else{
                    res->addSlice(C.logicalnot());
                }
            //    if(C.cardinality()!=0){
            //    res.bsi[res.size]=C;
            //res.size++;}
            res->sign = res->bsi[res->size-1];
        }else{
            int minSP = std::min((int)this->size, intSize);
            res = new BsiUnsigned(std::max((int)this->size, intSize)+1);
            HybridBitmap<uword> allOnes;
            allOnes.setSizeInBits(this->bsi[0].sizeInBits(),true);
            allOnes.density=1;
            if ((a&1)==0){
                res->bsi.push_back(this->bsi[0]);
                C = zeroBitmap;
            }
            else{
                res->bsi.push_back(this->bsi[0].logicalnot());
                C=this->bsi[0];
            }
            res->size++;
            int i;
            for(i=1;i<minSP;i++){
                if((a&(1<<i))!=0){
                    res->bsi.push_back(C.xorNot(this->bsi[i]));
                    //res.bsi[i] = C.xor(this.bsi[i].NOT());
                    C=this->bsi[i].Or(C);
                }else{
                    res->bsi.push_back(this->bsi[i].Xor(C));
                    C=this->bsi[i].And(C);
                }
                res->size++;
            }
            long cCard = C.numberOfOnes();
            if(this->size>minSP){
                while(i<this->size){
                    if(cCard>0){
                        res->bsi.push_back(this->bsi[i].Xor(C));
                        C=this->bsi[i].And(C);
                        cCard=C.numberOfOnes();
                    }else{
                        res->bsi.push_back(this->bsi[i]);
                    }
                    res->size++;
                    i++;
                }
            }else{
                while (i<intSize){
                    if(cCard>0){
                        if((a&(1<<i))!=0){
                            res->bsi.push_back(C.logicalnot());
                        }else{
                            res->bsi.push_back(C);
                            C=zeroBitmap;
                            cCard=0;
                        }
                        
                    }else{
                        if((a&(1<<i))!=0){res->bsi[i]=allOnes;
                        }else {res->bsi.push_back(zeroBitmap);}
                        
                    }
                    res->size++;
                    i++;
                }
            }
            if(cCard>0){
                res->bsi.push_back(C);
                res->size++;
            }
            
        }
        res->firstSlice=this->firstSlice;
        res->lastSlice=this->lastSlice;
        res->existenceBitmap = this->existenceBitmap;
        return res;
};

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::convertToTwos(int bitsize){
    BsiSigned<uword>* res = new BsiSigned<uword>();
    res->offset=this->offset;
    res->existenceBitmap = this->existenceBitmap;
    
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.addStreamOfEmptyWords(false,this->existenceBitmap.bufferSize());
    int i=0;
    for(i=0; i<this->getNumberOfSlices(); i++){
        res->addSlice(this->bsi[i]);
    }
    while(i<bitsize){
        res->addSlice(zeroBitmap);
        i++;
    }
    //this.setNumberOfSlices(bits);
    res->setTwosFlag(true);
    
    return res;
};

template <class uword>
long BsiUnsigned<uword>::getValue(int pos) const {
    long sum = 0;

    for (int i = 0; i < this->size; i++) {
    if(this->bsi[i].get(pos))
        sum += 1l<<(this->offset + i);
    }
    return sum;
};

template <class uword>
HybridBitmap<uword> BsiUnsigned<uword>::rangeBetween(long lowerBound, long upperBound){
    HybridBitmap<uword> B_gt;
    HybridBitmap<uword> B_lt;
    HybridBitmap<uword> B_eq1;
    HybridBitmap<uword> B_eq2;
    HybridBitmap<uword> B_f = this->existenceBitmap;
    B_gt.setSizeInBits(this->bsi[0].sizeInBits());
    B_lt.setSizeInBits(this->bsi[0].sizeInBits());
    B_eq1.setSizeInBits(this->bsi[0].sizeInBits()); B_eq1.density=1;
    B_eq2.setSizeInBits(this->bsi[0].sizeInBits()); B_eq2.density=1;
    
    for(int i=this->getNumberOfSlices()-1; i>=0; i--){
        if((upperBound & (1<<i)) !=0){
            HybridBitmap<uword> ans = B_eq1.andNot(this->bsi[i]);
            //the i'th bit is set in upperBound
            B_lt = B_lt.Or(ans);
            B_eq1 = B_eq1.And(this->bsi[i]);
        }else{ //The i'th bit is not set in uppperBound
            B_eq1=B_eq1.andNot(this->bsi[i]);
        }
        if((lowerBound & (1<<i)) != 0){ // the I'th bit is set in lowerBound
            B_eq2 = B_eq2.And(this->bsi[i]);
        }else{ //the i'th bit is not set in lowerBouond
            B_gt = B_gt.logicalor(B_eq2.And(this->bsi[i]));
            B_eq2 = B_eq2.andNot(this->bsi[i]);
        }
    }
    B_lt = B_lt.Or(B_eq1);
    B_gt = B_gt.Or(B_eq2);
    B_f = B_lt.And(B_gt.And(B_f));
    return B_f;
};

template <class uword>
BsiUnsigned<uword>* BsiUnsigned<uword>::abs(){
    return this;
};

template <class uword>
BsiUnsigned<uword>* BsiUnsigned<uword>::abs(int resultSlices, const HybridBitmap<uword> &EB){
    //    HybridBitmap zeroBitmap = new HybridBitmap();
    //    zeroBitmap.setSizeInBits(this.bsi[0].sizeInBits(),false);
    int min = std::min(this->size-1, resultSlices);
    BsiUnsigned<uword>* res = new BsiUnsigned<uword>(min+1);
    for (int i=0; i<min; i++){
        res->bsi[i]=this->bsi[i];
        res->size++;
    }
    res->size=min;
    res->addSlice(EB.logicalnot()); // this is for KNN to add one slice
    res->existenceBitmap=this->existenceBitmap;
    res->setPartitionID(this->getPartitionID());
    res->firstSlice=this->firstSlice;
    res->lastSlice=this->lastSlice;
    return res;
};

template <class uword>
BsiUnsigned<uword>* BsiUnsigned<uword>::absScale(double range){
    //    HybridBitmap zeroBitmap = new HybridBitmap();
    //    zeroBitmap.setSizeInBits(this.bsi[0].sizeInBits(),false);
    
    HybridBitmap<uword> penalty = this->bsi[this->size-1];
    
    int resSize=0;
    for (int i=this->size-1;i>=0;i--){
        penalty=penalty.logicalor(this->bsi[i]);
        if(penalty.numberOfOnes()>=(this->bsi[0].sizeInBits()*range)){
            //if(penalty.density>=0.9){
            //if(i==this.size-8){
            resSize=i;
            break;
        }
    }
    
    BsiUnsigned<uword> *res = new BsiUnsigned<uword>(resSize+1);
    
    
    
    
    for (int i=0; i<resSize; i++){
        res->bsi[i]=this->bsi[i];
        res->size++;
        
        
    }
    res->addSlice(penalty);
    
    
    res->existenceBitmap=this->existenceBitmap;
    res->setPartitionID(this->getPartitionID());
    res->firstSlice=this->firstSlice;
    res->lastSlice=this->lastSlice;
    return res;
};



/*
 Defining Other Functions -----------------------------------------------------------------------------------------
 */
template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::sum_Horizontal(const BsiAttribute<uword> *a) const{
    //TODO: implement
    BsiAttribute<uword>* res;
    return res;
}

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::SUMunsigned(BsiAttribute<uword>* a)const{
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
    BsiAttribute<uword>* res = new BsiUnsigned<uword>(std::max(this->size+this->offset, a->size+a->offset)+1);
    res->setPartitionID(a->getPartitionID());
    res->existenceBitmap = this->existenceBitmap.logicalor(a->existenceBitmap);
    int i = 0, s = a->size, p = this->size;
    
    
    
    int minOffset = std::min(a->offset, this->offset);
    res->offset = minOffset;
    
    int aIndex = 0;
    int thisIndex =0;
    
    if(this->offset>a->offset){
        for(int j=0;j<this->offset-minOffset; j++){
            if(j<a->size)
                res->bsi[res->size]=a->bsi[aIndex];
            else
                res->bsi[res->size]=zeroBitmap;
            aIndex++;
            res->size++;
        }
    }else if(a->offset>this->offset){
        for(int j=0;j<a->offset-minOffset;j++){
            if(j<this->size)
                res->bsi[res->size]=this->bsi[thisIndex];
            else
                res->bsi[res->size]=zeroBitmap;
            res->size++;
            thisIndex++;
        }
    }
    //adjust the remaining sizes for s and p
    s=s-aIndex;
    p=p-thisIndex;
    int minSP = std::min(s, p);
    
    if(minSP<=0){ // one of the BSI attributes is exausted
        for(int j=aIndex; j<a->size;j++){
            res->bsi[res->size]=a->bsi[j];
            res->size++;
        }
        for(int j=thisIndex; j<this->size;j++){
            res->bsi[res->size]=this->bsi[j];
            res->size++;
        }
        return res;
    }else {
        
        res->bsi.push_back(this->bsi[thisIndex].Xor(a->bsi[aIndex]));
        HybridBitmap<uword> C = this->bsi[thisIndex].And(a->bsi[aIndex]);
        res->size++;
        thisIndex++;
        aIndex++;
        
        for(i=1; i<minSP; i++){
            //res.bsi[i] = this.bsi[i].xor(a.bsi[i].xor(C));
            res->bsi.push_back(this->XOR(this->bsi[thisIndex], a->bsi[aIndex], C));
            //res.bsi[i] = this.bsi[i].xor(this.bsi[i], a.bsi[i], C);
            C= this->maj(this->bsi[thisIndex], a->bsi[aIndex], C);
            res->size++;
            thisIndex++;
            aIndex++;
        }
        
        if(s>p){
            for(i=p; i<s;i++){
                res->bsi[res->size] = a->bsi[aIndex].Xor(C);
                C=a->bsi[aIndex].And(C);
                res->size++;
                aIndex++;
            }
        }else{
            for(i=s; i<p;i++){
                res->bsi[res->size] = this->bsi[thisIndex].Xor(C);
                C = this->bsi[thisIndex].And(C);
                res->size++;
                thisIndex++;
            }
        }
        //if(!(this.lastSlice && a.lastSlice) && (C.cardinality()>0)){
        if(C.numberOfOnes()>0){
            res->bsi.push_back( C );
            res->size++;
        }
        return res;
    }
};


template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::SUMsigned(BsiAttribute<uword>* a){
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
    BsiAttribute<uword>* res = new BsiSigned<uword>(std::max((this->size+this->offset), (a->size+a->offset))+2);
    res->twosComplement=true;
    res->index = (this->index);
    res->existenceBitmap = this->existenceBitmap.logicalor(a->existenceBitmap);
    if (!a->twosComplement)
        a->signMagnitudeToTwos(a->size+1); //plus one for the sign
    
    int i = 0, s = a->size, p = this->size;
    int minOffset = std::min(a->offset, this->offset);
    res->offset = minOffset;
    
    int aIndex = 0;
    int thisIndex =0;
    
    if(this->offset>a->offset){
        for(int j=0;j<this->offset-minOffset; j++){
            if(j<a->size)
                res->bsi[res->size]=a->bsi[aIndex];
            else if(a->lastSlice)
                res->bsi[res->size]=a->sign; //sign extend if contains the sign slice
            else
                res->bsi[res->size] = zeroBitmap;
            aIndex++;
            res->size++;
        }
    }else if(a->offset>this->offset){
        for(int j=0;j<a->offset-minOffset;j++){
            if(j<this->size)
                res->bsi[res->size]=this->bsi[thisIndex];
            else
                res->bsi[res->size]=zeroBitmap;
            res->size++;
            thisIndex++;
        }
    }
    //adjust the remaining sizes for s and p
    s=s-aIndex;
    p=p-thisIndex;
    int minSP = std::min(s, p);
    
    if(minSP<=0){ // one of the BSI attributes is exausted
        for(int j=aIndex; j<a->size;j++){
            res->bsi[res->size]=a->bsi[j];
            res->size++;
        }
        HybridBitmap<uword> CC;
        for(int j=thisIndex; j<this->size;j++){
            if(a->lastSlice){ // operate with the sign slice if contains the last slice
                if(j==thisIndex){
                    res->bsi[res->size]=this->bsi[j].logicalxor(a->sign);
                    CC=this->bsi[j].logicaland(a->sign);
                    res->lastSlice=true;
                }else{
                    res->bsi[res->size]=this->XOR(this->bsi[j],a->sign,CC);
                    CC=this->maj(this->bsi[j],a->sign,CC);
                }
                res->size++;
            }else{
                res->bsi[res->size]=this->bsi[j];
                res->size++;}
        }
        
        //res.existenceBitmap = this.existenceBitmap.or(a.existenceBitmap);
        res->sign = &res->bsi[res->size-1];
        res->lastSlice=a->lastSlice;
        res->firstSlice=this->firstSlice|a->firstSlice;
        return res;
    }
    else {
        
        res->bsi[res->size] = this->bsi[thisIndex].logicalxor(a->bsi[aIndex]);
        HybridBitmap<uword> C = this->bsi[thisIndex].logicaland(a->bsi[aIndex]);
        res->size++;
        thisIndex++;
        aIndex++;
        
        for(i=1; i<minSP; i++){
            //res.bsi[i] = this.bsi[i].xor(a.bsi[i].xor(C));
            res->bsi[res->size] =this-> XOR(this->bsi[thisIndex], a->bsi[aIndex], C);
            //res.bsi[i] = this.bsi[i].xor(this.bsi[i], a.bsi[i], C);
            C= this->maj(this->bsi[thisIndex], a->bsi[aIndex], C);
            res->size++;
            thisIndex++;
            aIndex++;
        }
        
        if(s>p){ //a has more bits (the two's complement)
            for(i=p; i<s;i++){
                res->bsi[res->size] = a->bsi[aIndex].logicalxor(C);
                C=a->bsi[aIndex].logicaland(C);
                res->size++;
                aIndex++;
            }
        }else{
            for(i=s; i<p;i++){
                if(a->lastSlice){
                    res->bsi[res->size] = this->XOR(this->bsi[thisIndex], a->sign, C);
                    C = this->maj(this->bsi[thisIndex], a->sign, C);
                    res->size++;
                    thisIndex++;}
                else{
                    res->bsi[res->size] = this->bsi[thisIndex].Xor(C);
                    C = this->bsi[thisIndex].logicaland(C);
                    res->size++;
                    thisIndex++;}
            }
            if(this->lastSlice){
                res->bsi[res->size]= C.logicalxor(a->sign);
                C=C.logicaland(a->sign); //
                res->size++;}
        }
        if(!a->lastSlice && C.numberOfOnes()>0){
            //if(!a.lastSlice){
            res->bsi[res->size]= C;
            res->size++;
        }
        
        
        res->lastSlice=a->lastSlice;
        res->firstSlice=this->firstSlice|a->firstSlice;
        res->sign = &res->bsi[res->size-1];
        return res;
    }
};

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::SUM(long a, HybridBitmap<uword> EB, int rangeSlices)const{
    if (a==0){
        return this;
    }else{
        int intSize = (int)std::bitset< 64 >(std::min(std::abs(a),(long)rangeSlices)).to_string().length();
        
        HybridBitmap<uword> zeroBitmap;
        zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
        BsiAttribute<uword>* res;
        HybridBitmap<uword> C;
        if(a<0){
            //int minSP = Math.min(this.size, (intSize+1));
            res = new BsiUnsigned<uword>(intSize+1);
            //res.twosComplement=true;
            if ((a&1)==0){
                res->bsi[0]=this->bsi[0].logicaland(EB);
                C = zeroBitmap;
            }
            else{
                res->bsi[0]=EB.logicalandnot(this->bsi[0]);
                C=this->bsi[0].logicaland(EB);
            }
            res->size++;
            int i;
            for( i=1; i<intSize; i++ ){
                if((a&(1<<i))!=0){
                    res->bsi[i]=C.logicalxornot(EB.And(this->bsi[i]));
                    //res.bsi[i] = C.xor(this.bsi[i].NOT());
                    C=EB.logicaland(this->bsi[i]).logicalor(C);
                }else{
                    res->bsi[i]=C.logicalxor(EB.logicaland(this->bsi[i]));
                    //C=this.bsi[i].and(C);
                    C=C.logicaland(EB.logicaland(this->bsi[i].logicaland(C)));
                }
                res->size++;
                
            }
            
            res->addSlice(EB.logicalnot());
            //res.addSlice(C.and(EB));
            //    if(C.cardinality()!=0){
            //    res.bsi[res.size]=C;
            //res.size++;}
            res->sign=&res->bsi[res->size-1];
            res->firstSlice=this->firstSlice;
            res->lastSlice=this->lastSlice;
        }else{
            int minSP = std::min(this->size, intSize);
            res = new BsiUnsigned<uword>(std::max(this->size, intSize)+1);
            //TODO implement this part
        }
        res->existenceBitmap = this->existenceBitmap;
        res->setPartitionID(this->getPartitionID());
        return res;
    }
};
/*
* template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplyByConstantNew(int number)const {
    BsiUnsigned<uword>* res = nullptr;  
    //The result should have this->bsi.size() + sizeInBits(number) number of slices maximum
    int slices = sliceLengthFinder(number) + this->bsi.size();
    //Declare Sum and Carry
    HybridBitmap<uword> C, S;
    //Declare the offset
    int k = 0;
    
    while (number > 0) {
        //if the last bit of the number is 1
        if ((number & 1) == 1) {
            if (res == nullptr) {
                res = new BsiUnsigned<uword>(slices + 1);
                res->offset = k;
                for (int i = 0; i < this->size; i++) {
                    res->bsi.push_back(this->bsi[i]);
                }
                res->size = this->size;            
            }
            else {
                //Move the slices of the result by k positions
                HybridBitmap<uword> A, B;                
                B = this->bsi[0]; 
                while (k >= res->bsi.size()) {
                    //If k is greater than result's bsi size, A will be undefined
                    A = new HybridBitmap<uword>();
                    //A.addStreamOfEmptyWords(false, this->bsi[0].sizeInBits() / 64);
                    res->bsi.push_back(A);                    
                }
                res->size = k + 1;
                A = res->bsi[k];               
                S = A.Xor(B);
                C = A.And(B);
                res->bsi[k] = S;
                
                //Add the slices of the current BSI to the result
                for (int i = 1; i < this->size; i++) {                    
                    B = this->bsi[i];
                    if ((i + k) >= this->size) {
                        S = B.Xor(C);
                        C = B.And(C);
                        res->size++;
                        res->bsi.push_back(S);
                        continue;
                    }
                    else {
                        A = res->bsi[i + k];
                        S = A.Xor(B).Xor(C);
                        C = A.And(B).Or(B.And(C)).Or(A.And(C));
                    }
                    res->bsi[i + k] = S;                   
                }
                //Add the remaining slices of the result with the Carry C
                for (int i = this->size + k; i < res->size; i++) {
                    A = res->bsi[i];
                    S = A.Xor(C);
                    C = A.And(C);
                    res->bsi[i] = S;
                }
                if (C.numberOfOnes() > 0) {
                    res->bsi.push_back(C); // Carry bit
                    res->size++;
                }
            }
        }
        //Check for the next bit in number
        number >>= 1;
        k++;
    }

    //Check for null slices within the result range and fill them with zeroes
    int maxNotNull = 0;
    for (int i = 0; i < res->bsi.size(); i++) {
        if (res->bsi[i] != nullptr)
            maxNotNull = i;
    }
    for (int i = 0; i < maxNotNull; i++) {
        if (res->bsi[i] == nullptr) {
            res->bsi[i] = new HybridBitmap<uword>();
            // res.bsi[i].setSizeInBits(this.bsi[0].sizeInBits(), false);
            res->bsi[i].addStreamOfEmptyWords(false, this->existenceBitmap.sizeInBits() / 64);
        }
    }
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    
    return res;
    
};
*/
template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplyByConstantNew(int number)const {
    BsiUnsigned<uword>* res = nullptr;
    //The result should have this->bsi.size() + sizeInBits(number) number of slices maximum
    int slices = sliceLengthFinder(number) + this->bsi.size();
    //Declare Sum and Carry
    HybridBitmap<uword> C, S;
    //Declare the offset
    int k = 0;
    while (number > 0) {
        if ((number & 1)== 1) {
            //Add the slices to result
            if (res == nullptr) {
                res = new BsiUnsigned<uword>(slices + 1);
                res->offset = k;
                for (int i = 0; i < this->size; i++) {
                    res->bsi.push_back(this->bsi[i]);
                }
                res->size = this->size;
                k = 0;
            }
            else {
                //Initialize S and C
                HybridBitmap<uword>* A, B;
                B = this->bsi[0];
                while (k >= res->bsi.size()) {                    
                    A = new HybridBitmap<uword>();
                    A->padWithZeroes(this->bsi[0].sizeInBits()); 
                    res->bsi.push_back(*A);
                }
                res->size = res->bsi.size();
                A = &(res->bsi[k]);
                S = (*A).Xor(B);
                C = (*A).And(B);
                res->bsi[k] = S;
                //Actual adding the slices
                int i;
                for (i = 1; i < this->bsi.size(); i++) {
                    B = this->bsi[i];
                    while ((i + k) >= res->bsi.size()) {
                        size_t buffersize = this->bsi[0].bufferSize();                       
                        A = new HybridBitmap<uword>();
                        A->padWithZeroes(this->bsi[0].sizeInBits());                        
                        res->bsi.push_back(*A);
                    }
                    res->size = res->bsi.size();
                    A = &(res->bsi[i + k]);
                    S = (*A).Xor(B).Xor(C);
                    C = (*A).And(B).Or(B.And(C)).Or((*A).And(C));
                    res->bsi[i + k] = S;
                }
                //Add C to the remianing slices
                for (int j = this->size + k; j < res->bsi.size(); j++) {
                    A = &(res->bsi[j]);
                    S = (*A).Xor(C);
                    C = (*A).And(C);
                    res->bsi[j] = S;
                }
                //Handle the last carry
                if (C.numberOfOnes() > 0) {
                    res->bsi.push_back(C);
                    res->size = res->bsi.size();
                }

            }

        }//If the current bit position of the multiplier is one
        number = number >> 1;
        k++;

    }//While loop end for number greater than zero
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    return res;
};


template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplyByConstant(int number)const{
    BsiUnsigned<uword>* res = nullptr;
    HybridBitmap<uword> C, S;
    if(number < 0){
        int k = 0;
        number = 0 - number;
        while (number > 0) {
            if ((number & 1) == 1) {
                if (res == nullptr) {
                    res = new BsiUnsigned<uword>();
                    //                res->offset = k;
                    for (int i = 0; i < this->size; i++) {
                        res->bsi.push_back(this->bsi[i]);
                    }
                    res->size = this->size;
                    k = 0;
                } else {
                    /* Move the slices of res k positions */
                    HybridBitmap<uword> A, B;
                    A = res->bsi[k];
                    B = this->bsi[0];
                    // if (A==null || B==null) {
                    // System.out.println("A or B is null");
                    // }
                    S = A.Xor(B);
                    C = A.And(B);
                    // S = XOR_AND(A, B, C);
                    res->bsi[k] = S;
                    
                    // C = Sum[1];
                    
                    for (int i = 1; i < this->size; i++) {// Add the slices of this to the current res
                        
                        B = this->bsi[i];
                        if ((i + k) >=this->size){
                            S = B.Xor(C);
                            C = B.And(C);
                            res->size++;
                            res->bsi.push_back(S);
                            continue;
                            // S = XOR_AND(B, C, C);
                        } else {
                            A = res->bsi[i + k];
                            S = A.Xor(B).Xor(C);
                            // S = XOR(A, B, C);
                            C = A.And(B).Or(B.And(C)).Or(A.And(C));
                            // C = maj(A, B, C); // OR(OR(AND(A, B), AND(A, C)),
                            // AND(C, B));
                        }
                        res->bsi[i + k] = S;
                    }
                    for (int i = this->size + k; i < res->size; i++) {// Add the remaining slices of res with the Carry C
                        A = res->bsi[i];
                        S = A.Xor(C);
                        C = A.And(C);
                        // S = XOR_AND(A, C, C);
                        res->bsi[i] = S;
                    }
                    if (C.numberOfOnes() > 0) {
                        res->bsi.push_back(C); // Carry bit
                        res->size++;
                    }
                    /**/
                }
                // System.out.println("number="+number+" k="+k+" res="+res.SUM());
            }else{
                if (res == nullptr) {
                    res = new BsiUnsigned<uword>();
                    HybridBitmap<uword> zeroBitmap;
                    zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits(), false);
                    //                res->offset = k;
                    for (int i = 0; i < this->size; i++) {
                        res->bsi.push_back(zeroBitmap);
                    }
                    res->size = this->size;
                    k = 0;
                }
            }
            number >>= 1;
            k++;
            //        HybridBitmap<uword> temp;
            //        res->bsi.push_back(temp);
        }
            res->BsiAttribute<uword>::twosComplement = false;
            res->sign.setSizeInBits(this->bsi[0].sizeInBits(), true);
            res->sign.density = 1;
            res->existenceBitmap = this->existenceBitmap;
            res->rows = this->rows;
            res->index = this->index;
        
    }else{
        //If multiplier is greater than 0.
        int k = 0;
        //number = 0 - number;
        while (number > 0) {
            if ((number & 1) == 1) {
                if (res == nullptr) {
                    res = new BsiUnsigned<uword>();
                    //                res->offset = k;
                    //Add the slices of the object to the result
                    for (int i = 0; i < this->size; i++) {
                        res->bsi.push_back(this->bsi[i]);
                    }
                    res->size = this->size;
                    k = 0;
                } else {
                    /* Move the slices of res k positions */
                    HybridBitmap<uword> A, B;
                    A = res->bsi[k];
                    B = this->bsi[0];
                    // if (A==null || B==null) {
                    // System.out.println("A or B is null");
                    // }
                    S = A.Xor(B);
                    C = A.And(B);
                    // S = XOR_AND(A, B, C);
                    res->bsi[k] = S;
                    
                    // C = Sum[1];
                    
                    for (int i = 1; i < this->size; i++) {// Add the slices of this to the current res
                        
                        B = this->bsi[i];
                        if ((i + k) >=this->size){
                            S = B.Xor(C);
                            C = B.And(C);
                            res->size++;
                            res->bsi.push_back(S);
                            continue;
                            // S = XOR_AND(B, C, C);
                        } else {
                            A = res->bsi[i + k];
                            S = A.Xor(B).Xor(C);
                            // S = XOR(A, B, C);
                            C = A.And(B).Or(B.And(C)).Or(A.And(C));
                            // C = maj(A, B, C); // OR(OR(AND(A, B), AND(A, C)),
                            // AND(C, B));
                        }
                        res->bsi[i + k] = S;
                    }
                    for (int i = this->size + k; i < res->size; i++) {// Add the remaining slices of res with the Carry C
                        A = res->bsi[i];
                        S = A.Xor(C);
                        C = A.And(C);
                        // S = XOR_AND(A, C, C);
                        res->bsi[i] = S;
                    }
                    if (C.numberOfOnes() > 0) {
                        res->bsi.push_back(C); // Carry bit
                        res->size++;
                    }
                    /**/
                }
                // System.out.println("number="+number+" k="+k+" res="+res.SUM());
            }else{
                if (res == nullptr) {
                    res = new BsiUnsigned<uword>();
                    HybridBitmap<uword> zeroBitmap;
                    zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits(), false);
                    //                res->offset = k;
                    for (int i = 0; i < this->size; i++) {
                        res->bsi.push_back(zeroBitmap);
                    }
                    res->size = this->size;
                    k = 0;
                }
            }
            number >>= 1;
            k++;
            //        HybridBitmap<uword> temp;
            //        res->bsi.push_back(temp);
        }
        
//        res->BsiAttribute<uword>::twosComplement = false;
//        res->sign.setSizeInBits(this->bsi[0].sizeInBits(), true);
        res->sign.density = 1;
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
    }
    
    return res;
};

//template <class uword>
//BsiAttribute<uword>* BsiUnsigned<uword>::negate(){
//    HybridBitmap<uword> onesBitmap;
//    onesBitmap.setSizeInBits(this->bsi[0].sizeInBits(), true);
//    onesBitmap.density=1;
//    
//    int signslicesize=1;
//    if(this->firstSlice)
//        signslicesize=2;
//    
//    BsiSigned<uword>* res = new BsiSigned<uword>(this->getNumberOfSlices()+signslicesize);
//    for(int i=0; i<this->getNumberOfSlices(); i++){
//        res->bsi[i]=this->bsi[i].Not();
//        //            try {
//        //                res.bsi[i]=(HybridBitmap) this.bsi[i].clone();
//        //            } catch (CloneNotSupportedException e) {
//        //                // TODO Auto-generated catch block
//        //                e.printStackTrace();
//        //            }
//        //            res.bsi[i].not();
//        res->size++;
//    }
//    res->addSlice(onesBitmap);
//    
//    if(this->firstSlice){
//        res->addOneSliceNoSignExt(onesBitmap);
//    }
//    res->existenceBitmap=this->existenceBitmap;
//    res->setPartitionID(this->getPartitionID());
//    res->sign=&res->bsi[res->size-1];
//    res->firstSlice=this->firstSlice;
//    res->lastSlice=this->lastSlice;
//    res->setTwosFlag(true);
//    return res;
//};




template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplyWithBSI(BsiUnsigned &unbsi) const{
    BsiUnsigned<uword>* res = nullptr;
    HybridBitmap<uword> C, S, FS, DS;
    int k = 0;
    res = new BsiUnsigned<uword>(this->bsi.size() + unbsi.bsi.size());
    res->offset = k;
    for (int i = 0; i < this->size; i++) {
        res->bsi.push_back(unbsi.bsi[0].andVerbatim(this->bsi[i]));
    }
    res->size = this->size;
    k = 1;
    for (int it=1; it<unbsi.size; it++) {
        /* Move the slices of res k positions */
        HybridBitmap<uword> A, B;
        A = res->bsi[k];
        B = this->bsi[0];
        S = A.xorVerbatim(B);
        C = A.andVerbatim(B);
        FS = unbsi.bsi[it].andVerbatim(S);
        res->bsi[k] = unbsi.bsi[it].selectMultiplication(res->bsi[k],FS);
        //                res->bsi[k] = unbsi.bsi[it].Not().And(res->bsi[k]).Or(unbsi.bsi[it].And(FS));

        for (int i = 1; i < this->size; i++) {// Add the slices of this to the current res
            B = this->bsi[i];
            if ((i + k) < res->size){
                A = res->bsi[i + k];
                S = A.xorVerbatim(B).xorVerbatim(C);
                C = A.maj(B, C);
                //C = A.And(B).Or(B.And(C)).Or(A.And(C));

            } else {
                S = B.xorVerbatim(C);
                C = B.andVerbatim(C);
                res->size++;
                FS = unbsi.bsi[it].andVerbatim(S);
                res->bsi.push_back(FS);
            }
            FS = unbsi.bsi[it].andVerbatim(S);
            res->bsi[i + k] = unbsi.bsi[it].selectMultiplication(res->bsi[i + k],FS);
            //res->bsi[i + k] = res->bsi[i + k].andNot(unbsi.bsi[it]).Or(unbsi.bsi[it].And(FS));
        }
        for (int i = this->size + k; i < res->size; i++) {// Add the remaining slices of res with the Carry C
            A = res->bsi[i];
            S = A.xorVerbatim(C);
            C = A.andVerbatim(C);
            FS = unbsi.bsi[it].andVerbatim(S);
            res->bsi[k] = unbsi.bsi[it].selectMultiplication(res->bsi[k],FS);
        }
        if (C.numberOfOnes() > 0) {
            res->bsi.push_back(unbsi.bsi[it].andVerbatim(C)); // Carry bit
            res->size++;
        }
        k++;
    }
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    return res;
};
//<<<<<<< HEAD
//template <class uword>
//BsiUnsigned<uword>* BsiUnsigned<uword>::twosComplement() const{
//    HybridBitmap<uword> onesBitmap, A,S,C;
//    //C.setSizeInBits(this->bsi[0].sizeInBits(), false);
//    //C.density = 0;
//    onesBitmap.setSizeInBits(this->bsi[0].sizeInBits(), true);
//    onesBitmap.density=1;
//
//    int signslicesize=1;
//    if(this->firstSlice)
//        signslicesize=2;
//
//    BsiUnsigned<uword>* res = new BsiUnsigned<uword>(this->getNumberOfSlices()+signslicesize);
//    A = this->bsi[0].Not();
//    S = A.Xor(onesBitmap);
//    C = A.And(onesBitmap);
//    res->bsi.push_back(S);
//    for(int i=1; i<this->getNumberOfSlices(); i++){
//        A = this->bsi[i].Not();
//        S = A.Xor(C);
//        C = A.And(C);
////        S = A.xorVerbatim(onesBitmap);
////        C = A.andVerbatim(onesBitmap);
//        res->bsi.push_back(S);
//        res->size++;
//    }
//    if (C.numberOfOnes() > 0) {
//        res->bsi.push_back(C); // Carry bit
//        res->size++;
//    }
//    //res->addSlice(onesBitmap);
//
//    if(this->firstSlice){
//        res->addOneSliceNoSignExt(onesBitmap);
//    }
//    res->existenceBitmap=this->existenceBitmap;
//    res->setPartitionID(this->getPartitionID());
//    res->sign=&res->bsi[res->size-1];
//    res->firstSlice=this->firstSlice;
//    res->lastSlice=this->lastSlice;
//    res->setTwosFlag(true);
//    return res;
//};
//
//template <class uword>
//BsiUnsigned<uword>& BsiUnsigned<uword>::multiplyWithKaratsuba(BsiUnsigned &unbsi)const{
//
//=======


template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplyBSI(BsiAttribute<uword> *unbsi) const{
    BsiUnsigned<uword>* res = nullptr;
    HybridBitmap<uword> C, S, FS, DS;
    int k = 0;
    res = new BsiUnsigned<uword>();
    res->offset = k;
    for (int i = 0; i < this->size; i++) {
        res->bsi.push_back(unbsi->bsi[0].And(this->bsi[i]));
    }
    res->size = this->size;

    k = 1;
    for (int it=1; it<unbsi->size; it++) {
        /* Move the slices of res k positions */
//        HybridBitmap<uword> A, B;
//        A = res->bsi[k];
//        B = this->bsi[0];
        S=res->bsi[k];
        S.XorInPlace(this->bsi[0]);
        C = res->bsi[k].And(this->bsi[0]);
        FS = unbsi->bsi[it].And(S);
        //res->bsi[k] = unbsi.bsi[it].selectMultiplication(res->bsi[k],FS);
//        res->bsi[k].selectMultiplicationInPlace(unbsi->bsi[it],FS);
                //res->bsi[k] = unbsi.bsi[it].Not().And(res->bsi[k]).Or(unbsi.bsi[it].And(FS));
        res->bsi[k] = unbsi->bsi[it].Not().And(res->bsi[k]).Or(unbsi->bsi[it].And(FS));

        for (int i = 1; i < this->size; i++) {// Add the slices of this to the current res
//            B = this->bsi[i];
            if ((i + k) < res->size){
//                A = res->bsi[i + k];
                S=res->bsi[i + k];
                S.XorInPlace(this->bsi[i]);
                S.XorInPlace(C);
//                C = res->bsi[i + k].maj(this->bsi[i], C);
//                C.majInPlace(res->bsi[i + k],this->bsi[i]);
//                C = A.And(B).Or(B.And(C)).Or(A.And(C));
                C = res->bsi[i + k].And(this->bsi[i]).Or(this->bsi[i].And(C)).Or(res->bsi[i + k].And(C));

            } else {
                S=this->bsi[i];
                S.XorInPlace(C);
                C.AndInPlace(this->bsi[i]);
//                C = this->bsi[i].And(C);
                res->size++;
                FS = unbsi->bsi[it].And(S);
                res->bsi.push_back(FS);
            }
            FS = unbsi->bsi[it].And(S);
//            res->bsi[i + k] = unbsi.bsi[it].selectMultiplication(res->bsi[i + k],FS);
//            res->bsi[i + k] = unbsi->bsi[it].selectMultiplication(res->bsi[i + k],FS);
//            res->bsi[i+k].selectMultiplicationInPlace(unbsi->bsi[it],FS);
            //res->bsi[i + k] = res->bsi[i + k].andNot(unbsi.bsi[it]).Or(unbsi.bsi[it].And(FS));
            res->bsi[i + k] = res->bsi[i + k].andNot(unbsi->bsi[it]).Or(unbsi->bsi[it].And(FS));    //selectMultiplication not working for verbatim=false

        }
        for (int i = this->size + k; i < res->size; i++) {// Add the remaining slices of res with the Carry C
            S = res->bsi[i];
            S.XorInPlace(C);
            C.AndInPlace(res->bsi[i]);
//            C = res->bsi[i].And(C);
            FS = unbsi->bsi[it].And(S);
//            res->bsi[k] = unbsi.bsi[it].selectMultiplication(res->bsi[k],FS);
//            res->bsi[k] = unbsi->bsi[it].selectMultiplication(res->bsi[k],FS);
//            res->bsi[k].selectMultiplicationInPlace(unbsi->bsi[it],FS);
            res->bsi[k] = unbsi->bsi[it].andNot(res->bsi[k]).Or(unbsi->bsi[it].And(FS)); //selectMultiplication also works
        }
        if (C.numberOfOnes() > 0) {
            res->bsi.push_back(unbsi->bsi[it].And(C)); // Carry bit
            res->size++;
        }
        k++;
    }
 
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    return res;
};
/*
This method was written to try out the dot product operation
Dot product would basically be multiplication of two vectors
Followed by the scalar sum of the resulting vector
*/
template <class uword>
long BsiUnsigned<uword>::dotProduct(BsiAttribute<uword>* unbsi) const{
    //Dot product for verbatim bitmaps
    std::cout << "Let's try to do dot product\n";
    // Initialize the necessary vectors
    // res = new BsiUnsigned<uint64_t>();
    HybridBitmap<uint64_t> hybridBitmap;
    hybridBitmap.reset();
    hybridBitmap.verbatim = true;
    /*
    for (int j = 0; j < this->size + bsi3->size; j++)
    {
        res->addSlice(hybridBitmap);
    }
    */

    int size_a = this->size;
    int size_b = unbsi->size;
    std::vector<uint64_t> a(size_a);
    std::vector<uint64_t> b(size_b);
    std::vector<uint64_t> answer(size_a + size_b);
    long dotProductSum = 0;
    int ansSize = a.size();
    // For each word in the BSI buffer
    for (int bu = 0; bu < this->bsi[0].bufferSize(); bu++)
    {
        // For each slice, get the decimal representations added into an array
        for (int j = 0; j < this->size; j++)
        {
            a[j] = this->bsi[j].getWord(bu); // fetching one word
        }
        for (int j = 0; j < unbsi->size; j++)
        {
            b[j] = unbsi->bsi[j].getWord(bu);
        }
        // Multiply the two vectors and take their running sum
        // For {1,2,3,4,5} => {21, 6, 24}}
        for (int i = 0; i < a.size(); i++)
        {
            answer[i] = a[i] & b[0];
        }
        for (int i = a.size(); i < b.size() + a.size(); i++)
        {
            answer[i] = 0;
        }
        //{21,4,16,0,0,0}
        uint64_t S, C, FS;
        int k = 1;
        // For each value in the second vector
        for (int it = 1; it < b.size(); it++)
        {
            S = answer[k] ^ a[0];
            C = answer[k] & a[0];
            FS = S & b[it];
            answer[k] = (~b[it] & answer[k]) | (b[it] & FS);
            //{21,0,16,0,0,0}
            // Second iteration of loop via b
            //{21,0,2,4,0,0}
            for (int i = 1; i < a.size(); i++)
            {
                // What happens here
                if ((i + k) < ansSize)
                {
                    S = answer[i + k] ^ a[i] ^ C;
                    C = (answer[i + k] & a[i]) | (a[i] & C) | (answer[i + k] & C);
                }
                else
                {
                    S = a[i] ^ C;
                    C = a[i] & C;
                    FS = S & b[it];
                    ansSize++;
                    answer[ansSize - 1] = FS;
                }
                FS = b[it] & S;
                answer[i + k] = (~b[it] & answer[i + k]) | (b[it] & FS);
                //{21,0,18,0,0,0}
                //{21,0,18,4,0,0}
                // Second iteration of loop via b
                //{21,0,2,20,0,0}
                //{21,0,2,20,24,0}
            }
            // When does this even become a usecase
            // When there is extra to be calculated?
            for (int i = a.size() + k; i < ansSize; i++)
            {
                S = answer[i] ^ C;
                C = answer[i] & C;
                FS = b[it] & S;
                answer[k] = (~b[it] & answer[k]) | (b[it] & FS);
                ;
            }
            if (C > 0)
            {
                ansSize++;
                answer[ansSize - 1] = b[it] & C;
            }
            k++;
        } // End of loop via size of b
        // So we have the answer for one buffer in ans
        // Add it as a slice to our result
        /*
        for (int j = 0; j < ans.size(); j++)
        {
            res->bsi[j].addVerbatim(answer[j]);
        }
        */
        // Get the number of ones in each element of the answer vector and sum it
        for (auto n = 0; n < answer.size(); n++)
        {
            long temp = countOnes(answer[n]) * (1 << n);
            dotProductSum += temp;
        }
    }
    // Finally we have the result in res
    //{21,0,2,20,24,0}
    return dotProductSum;
};
/*
* int countOnes(int n)
{
    int count = 0;
    while (n != 0)
    {
        n = n & (n - 1); // remove the rightmost 1-bit from n
        count++;
    }
    return count;
}
*/




template <class uword>
BsiUnsigned<uword>* BsiUnsigned<uword>::multiplyBSIWithPrecision(const BsiUnsigned &unbsi, int precision) const{
    
    int precisionInBits = 3*precision +1;
    BsiUnsigned<uword>* res = nullptr;
    HybridBitmap<uword> C, S, FS, DS;
    int k = 0;
    res = new BsiUnsigned<uword>();
    res->offset = k;
    for (int i = 0; i < this->size; i++) {
        res->bsi.push_back(unbsi.bsi[0].andVerbatim(this->bsi[i]));
    }
    res->size = this->size;
    k = 1;
    for (int it=1; it<unbsi.size; it++) {
        /* Move the slices of res k positions */
        //HybridBitmap<uword> A, B;
        //A = res->bsi[k];
        //B = this->bsi[0];
        S=res->bsi[k];
        S.XorInPlace(this->bsi[0]);
        C = res->bsi[k].andVerbatim(this->bsi[0]);
        FS = unbsi.bsi[it].andVerbatim(S);
        //res->bsi[k] = unbsi.bsi[it].selectMultiplication(res->bsi[k],FS);
        res->bsi[k].selectMultiplicationInPlace(unbsi.bsi[it],FS);
        //                res->bsi[k] = unbsi.bsi[it].Not().And(res->bsi[k]).Or(unbsi.bsi[it].And(FS));
        
        for (int i = 1; i < this->size; i++) {// Add the slices of this to the current res
            //B = this->bsi[i];
            if ((i + k) < res->size){
                //A = res->bsi[i + k];
                S=res->bsi[i + k];
                S.XorInPlace(this->bsi[i]);
                S.XorInPlace(C);
                //C = res->bsi[i + k].maj(this->bsi[i], C);
                C.majInPlace(res->bsi[i + k],this->bsi[i]);
                //C = A.And(B).Or(B.And(C)).Or(A.And(C));
                
            } else {
                S=this->bsi[i];
                S.XorInPlace(C);
                C.AndInPlace(this->bsi[i]);
                //                C = this->bsi[i].And(C);
                res->size++;
                FS = unbsi.bsi[it].andVerbatim(S);
                res->bsi.push_back(FS);
            }
            FS = unbsi.bsi[it].andVerbatim(S);
            //res->bsi[i + k] = unbsi.bsi[it].selectMultiplication(res->bsi[i + k],FS);
            res->bsi[i+k].selectMultiplicationInPlace(unbsi.bsi[it],FS);
            //res->bsi[i + k] = res->bsi[i + k].andNot(unbsi.bsi[it]).Or(unbsi.bsi[it].And(FS));
        }
        for (int i = this->size + k; i < res->size; i++) {// Add the remaining slices of res with the Carry C
            S = res->bsi[i];
            S.XorInPlace(C);
            C.AndInPlace(res->bsi[i]);
            //            C = res->bsi[i].And(C);
            FS = unbsi.bsi[it].andVerbatim(S);
            //res->bsi[k] = unbsi.bsi[it].selectMultiplication(res->bsi[k],FS);
            res->bsi[k].selectMultiplicationInPlace(unbsi.bsi[it],FS);
        }
        if (C.numberOfOnes() > 0) {
            res->bsi.push_back(unbsi.bsi[it].andVerbatim(C)); // Carry bit
            res->size++;
        }
        k++;
    }
    
    int truncateBits = res->bsi.size() - precisionInBits;
    for (int i=0; i< truncateBits; i++){
        res->bsi.erase(res->bsi.begin());
    }
    res->size = res->bsi.size();
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    return res;
};


//template <class uword=uint64_t>
//void multiply(uword a[], uword b[], uword ans[], int size_a, int size_b){
//    uword S=0,C=0,FS;
//    int k=0, ansSize=0;
////    uword answer[size_b + size_a];
//    for(int i=0; i<size_a; i++){
//        ans[i] = a[i] & b[0];
//    }
//    for(int i = size_a; i< size_a+size_b; i++){
//        ans[i] = 0;
//    }
//    k=1;
//    ansSize = size_a;
//    for(int it=1; it<size_b; it++){
////        uword A,B;
////        A = answer[k];
////        B = a[0];
//        S = ans[k]^a[0];
//        C = ans[k]&a[0];
//        FS = S & b[it];
//        //~buffer[i] & res.buffer[i]) | (buffer[i] & FS.buffer[i])
//        ans[k] = (~b[it] & ans[k]) | (b[it] & FS);
//
//        for(int i=1; i<size_a; i++){
////            B = a[i];
//            if((i+k) < ansSize){
////                A = answer[i+k];
//                S = ans[i+k] ^ a[i] ^ C;
//                C = (ans[i+k]&a[i]) | (a[i]&C) | (ans[i+k]&C);
//            }else{
//                S = a[i] ^ C;
//                C = a[i] & C;
//                FS = S & b[it];
//                ansSize++;
//                ans[ansSize - 1] = FS;
//            }
//            FS = b[it] & S;
//            ans[i + k] =(~b[it] & ans[i + k]) | (b[it] & FS);
//            //res->bsi[i + k] = res->bsi[i + k].andNot(unbsi.bsi[it]).Or(unbsi.bsi[it].And(FS));
//        }
//        for(int i=size_a + k; i< ansSize; i++){
//            S = ans[i] ^ C;
//            C = ans[i] & C;
//            FS = b[it] & S;
//            ans[k] = (~b[it] & ans[k]) | (b[it] & FS);;
//        }
////        answer[it+k+1] = b[it] & C;
//        if(C>0){
//            ansSize++;
//            ans[ansSize-1] = b[it] & C;
//        }
//        k++;
//    }
////    for(int i= 0; i< size_b + size_a; i++){
////        ans[i] = answer[i];
////    }
//
//};
template <class uword>
void BsiUnsigned<uword>::multiply(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const{
    uword S=0,C=0,FS;
    int startPosition = b.size() + a.size() - ans.size();
    int k=0, ansSize=0;
//    uword answer[size_b + size_a];
    
    for(int i=0; i<a.size(); i++){
        ans[i] = a[i] & b[0];
    }
    for(int i = a.size(); i< b.size() + a.size(); i++){
        ans[i] = 0;
    }
    k=1;
    ansSize = a.size();
    for(int it=1; it<b.size(); it++){
        //        uword A,B;
        //        A = answer[k];
        //        B = a[0];
        S = ans[k]^a[0];
        C = ans[k]&a[0];
        FS = S & b[it];
        //~buffer[i] & res.buffer[i]) | (buffer[i] & FS.buffer[i])
        ans[k] = (~b[it] & ans[k]) | (b[it] & FS);
        
        for(int i=1; i<a.size(); i++){
            //            B = a[i];
            if((i+k) < ansSize){
                //                A = answer[i+k];
                S = ans[i+k] ^ a[i] ^ C;
                C = (ans[i+k]&a[i]) | (a[i]&C) | (ans[i+k]&C);
            }else{
                S = a[i] ^ C;
                C = a[i] & C;
                FS = S & b[it];
                ansSize++;
                ans[ansSize - 1] = FS;
            }
            FS = b[it] & S;
            ans[i + k ] =(~b[it] & ans[i + k ]) | (b[it] & FS);
            //res->bsi[i + k] = res->bsi[i + k].andNot(unbsi.bsi[it]).Or(unbsi.bsi[it].And(FS));
        }
        for(int i=a.size() + k; i< ansSize; i++){
            S = ans[i] ^ C;
            C = ans[i] & C;
            FS = b[it] & S;
            ans[k] = (~b[it] & ans[k]) | (b[it] & FS);;
        }
        //        answer[it+k+1] = b[it] & C;
        if(C>0){
            ansSize++;
            ans[ansSize-1] = b[it] & C;
        }
        k++;
    }
//        for(int i= 0; i< size_ans; i++){
//            ans[i] = answer[i];
//        }
    
};


template <class uword=uint64_t>
void multiplyWithPrecision(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans){
    uword S=0,C=0,FS;
    int k=0, ansSize=0;
    //    uword answer[size_b + size_a];
    for(int i=0; i<a.size(); i++){
        ans[i] = a[i] & b[0];
    }
    for(int i = a.size(); i< ans.size(); i++){
        ans[i] = 0;
    }
    k=1;
    ansSize = a.size();
    for(int it=1; it<b.size(); it++){
        //        uword A,B;
        //        A = answer[k];
        //        B = a[0];
        S = ans[k]^a[0];
        C = ans[k]&a[0];
        FS = S & b[it];
        //~buffer[i] & res.buffer[i]) | (buffer[i] & FS.buffer[i])
        ans[k] = (~b[it] & ans[k]) | (b[it] & FS);
        
        for(int i=1; i<a.size(); i++){
            //            B = a[i];
            if((i+k) < ansSize){
                //                A = answer[i+k];
                S = ans[i+k] ^ a[i] ^ C;
                C = (ans[i+k]&a[i]) | (a[i]&C) | (ans[i+k]&C);
            }else{
                S = a[i] ^ C;
                C = a[i] & C;
                FS = S & b[it];
                ansSize++;
                ans[ansSize - 1] = FS;
            }
            FS = b[it] & S;
            ans[i + k] =(~b[it] & ans[i + k]) | (b[it] & FS);
            //res->bsi[i + k] = res->bsi[i + k].andNot(unbsi.bsi[it]).Or(unbsi.bsi[it].And(FS));
        }
        for(int i=a.size() + k; i< ansSize; i++){
            S = ans[i] ^ C;
            C = ans[i] & C;
            FS = b[it] & S;
            ans[k] = (~b[it] & ans[k]) | (b[it] & FS);;
        }
        //        answer[it+k+1] = b[it] & C;
        if(C>0){
            ansSize++;
            ans[ansSize-1] = b[it] & C;
        }
        k++;
    }
    //    for(int i= 0; i< size_b + size_a; i++){
    //        ans[i] = answer[i];
    //    }
    
};

/*
 */

template <class uword>
BsiAttribute<uword>*  BsiUnsigned<uword>::multiplyWithBsiHorizontal(const BsiAttribute<uword> *unbsi, int precision) const{
    int precisionInBits = 3*precision +1;
    BsiUnsigned<uword>* res = nullptr;
    res = new BsiUnsigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.reset();
    hybridBitmap.verbatim = true;
    for(int j=0; j< this->size + unbsi->size; j++){
        res->addSlice(hybridBitmap);
    }
    int size_a = this->size;
    int size_b = unbsi->size;
    std::vector<uword> a(size_a);
    std::vector<uword> b(size_b);
    std::vector<uword> answer(size_a + size_b);
    
    for(int i=0; i< this->bsi[0].bufferSize(); i++){
        for(int j=0; j< this->size; j++){
            a[j] = this->bsi[j].getWord(i); //fetching one word
        }
        for(int j=0; j< unbsi->size; j++){
             b[j] = unbsi->bsi[j].getWord(i);
        }
        this->multiply(a,b,answer);         //perform multiplication on one word
//        this->multiplyBSI(a);         //perform multiplication on one word
//        this->multiplyWithBSI(b);         //perform multiplication on one word

        for(int j=0; j< answer.size() ; j++){
            res->bsi[j].addVerbatim(answer[j]);
        }
    }
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    return res;
};


template <class uword>
BsiAttribute<uword>*  BsiUnsigned<uword>::multiplyWithBsiHorizontal(const BsiAttribute<uword> *unbsi) const{
    BsiUnsigned<uword>* res = nullptr;
    res = new BsiUnsigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.reset();
    hybridBitmap.verbatim = true;
    for(int j=0; j< this->size + unbsi->size; j++){
        res->addSlice(hybridBitmap);
    }
    int size_a = this->size;
    int size_b = unbsi->size;
    std::vector<uword> a(size_a);
    std::vector<uword> b(size_b);
    std::vector<uword> answer(size_a + size_b);
    
    for(int i=0; i< this->bsi[0].bufferSize(); i++){
        for(int j=0; j< this->size; j++){
            a[j] = this->bsi[j].getWord(i); //fetching one word
        }
        for(int j=0; j< unbsi->size; j++){
             b[j] = unbsi->bsi[j].getWord(i);
        }
        this->multiply(a,b,answer);         //perform multiplication on one word
//        this->multiplyBSI(a);         //perform multiplication on one word
//        this->multiplyWithBSI(b);         //perform multiplication on one word

        for(int j=0; j< answer.size() ; j++){
            res->bsi[j].addVerbatim(answer[j]);
        }
    }
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    return res;
};


/*
 * multiplyWithBsiHorizontal_array perform multiplication betwwen bsi using multiply_array
 * support both verbatim and compressed Bsi(using existenceBitmap)
 */

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplication_Horizontal(const BsiAttribute<uword> *a) const{
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
BsiAttribute<uword>* BsiUnsigned<uword>::multiplication_Horizontal_compressed(const BsiAttribute<uword> *a) const{
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiUnsigned<uword>* res = new BsiUnsigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        return res;
    }
    
    BsiUnsigned<uword>* res = new BsiUnsigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.setSizeInBits(this->existenceBitmap.sizeInBits());
    for(int j=0; j< this->size + a->size; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->size;
    int size_y = a->size;
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
    return res;
}


/*
 * multiplication_Horizontal_Verbatim perform multiplication betwwen bsi using multiply_array
 * only support verbatim Bsi(using existenceBitmap)
 */


template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplication_Horizontal_Verbatim(const BsiAttribute<uword> *a) const{
    
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiUnsigned<uword>* res = new BsiUnsigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        return res;
    }
    BsiUnsigned<uword>* res = new BsiUnsigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.reset();
    hybridBitmap.verbatim = true;
    for(int j=0; j< this->size + a->size; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->size;
    int size_y = a->size;
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
    return res;
}


/*
 * multiplication_Horizontal_Hybrid perform multiplication betwwen bsi using multiply_array
 * only support hybrid Bsis(one is verbatim and one is compressed)
 */

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplication_Horizontal_Hybrid(const BsiAttribute<uword> *a) const{

    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiUnsigned<uword>* res = new BsiUnsigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        return res;
    }

    BsiUnsigned<uword>* res = new BsiUnsigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.setSizeInBits(this->existenceBitmap.sizeInBits());
    for(int j=0; j< this->size + a->size; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->size;
    int size_y = a->size;
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
    return res;
};



template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplication_Horizontal_Hybrid_other(const BsiAttribute<uword> *a) const{
    
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiUnsigned<uword>* res = new BsiUnsigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        res->sign = this->sign.Xor(a->sign);
        res->is_signed = true;
        res->twosComplement = false;
        return res;
    }
    
    BsiUnsigned<uword>* res = new BsiUnsigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    for(int j=0; j< this->size + a->size; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->size;
    int size_y = a->size;
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
    return res;
};

/*
 * multiply_array perform multiplication at word level
 * word from every bitmap of Bsi is multiplied with other bsi's word
 * it is modified version of Booth's Algorithm
 */

template <class uword>
void BsiUnsigned<uword>:: multiply_array(uword a[],int size_a, uword b[], int size_b, uword ans[], int size_ans)const{
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


template <class uword=uint64_t>
int makeEqualLength(std::vector<uword> &x, std::vector<uword> &y) {
    int len1 = x.size();
    int len2 = y.size();
    if (len1 < len2)
    {
        for (int i = 0 ; i < len2 - len1 ; i++)
            x.push_back((uword)0);
        return len2;
    }
    else if (len1 > len2)
    {
        for (int i = 0 ; i < len1 - len2 ; i++)
            y.push_back((uword)0);
    }
    return len1; // If len1 >= len2
}

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplication(BsiAttribute<uword> *a)const{
   BsiAttribute<uword>* res = multiplyWithBsiHorizontal(a,3);

    int size = res->bsi.size();
    for(int i=0; i<size; i++){
        res->bsi[i].density = res->bsi[i].numberOfOnes()/(double)res->getNumberOfRows();
    }
    return res;
}


template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplication_array(BsiAttribute<uword> *a)const{
    BsiAttribute<uword>* res = multiplyWithBsiHorizontal(a,3);
    int size = res->bsi.size();
    for(int i=0; i<size; i++){
        res->bsi[i].density = res->bsi[i].numberOfOnes()/(double)res->getNumberOfRows();
    }
    return res;
}

template <class uword>
void BsiUnsigned<uword>::multiplicationInPlace(BsiAttribute<uword> *a){
    BsiAttribute<uword>* res = multiplyWithBsiHorizontal(a,3);
    int size = res->bsi.size();
    for(int i=0; i<size; i++){
        res->bsi[i].density = res->bsi[i].numberOfOnes()/(double)res->getNumberOfRows();
    }
   
}
//template <class uword=uint64_t>
//uword multiplyiSingleBit(std::vector<uword> &x, std::vector<uword> &y){
//        uword ans =
//}

template <class uword=uint64_t>
std::vector<uword>& multiplyByKarstuba(std::vector<uword> &x, std::vector<uword> &y, int size_x, int size_y){
    // Find the maximum of lengths of x and Y and make length
    // of smaller string same as that of larger string
    std::vector<uword> answer;
    int n = makeEqualLength(x, y);
    // Base cases
    if (n == 0) return answer.resize(1);
    if (n == 1) return multiplyiSingleBit(x, y);
    
    
}



//
//BsiAttribute<uword>* BsiUnsigned<uword>::karatsuba(BsiUnsigned &a, int startSlice, int endSlice){
//
//
//}
//
///**
// *
// * @tparam uword
// * @param a is the BSI with smaller number of slices
// * @return
// */
//template <class uword>
//BsiAttribute<uword>* BsiUnsigned<uword>::karatsubaMultiply(BsiUnsigned &a){
//    BsiUnsigned<uword>* res = nullptr;
//    HybridBitmap<uword> C; //carry slice
//    long sizeofThis = this->bsi[0].sizeInBits();
//    C.fastaddStreamOfEmptyWords(false,sizeofThis);
//    C.setSizeInBits(sizeofThis);
//    C.density=0;
//
//    //padding a with slices of zeros to make both sides with same number of slices
//    for(int i=a.size; i< this->size; i++){
//        a.addSlice(C);
//    }
//
//    if(this->size==1){
//        return multiplyTwoSlices(this->bsi[0], a.bsi[0]);
//    }
//
//    int firstHalf= this->size/2;
//    BsiAttribute<uword> P1 = this->karatsuba(a,0, firstHalf);
//    BsiAttribute<uword> P2 = this->karatsuba(a,firstHalf+1, this->size);
//    BsiAttribute<uword> P3 = (this.partialSUM(a, 0, firstHalf)).karatsubaMultiply(this.partialSUM(a, firstHalf+1, this->size));
//
//
//
//
//    res->setExistenceBitmap(this->existenceBitmap);
//    res->rows = this->rows;
//    res->index = this->index;
//    return res;
//};


template <class uword>
long BsiUnsigned<uword>::sumOfBsi() const{
    long sum =0;
//    int power = 1;
    for (int i=0; i< this->bsi.size(); i++){
        sum += this->getSlice(i).numberOfOnes()<<(i);
    }
    return sum;
}

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::peasantMultiply(BsiUnsigned &unbsi) const{
    BsiAttribute<uword>* res = nullptr;
    res = new BsiUnsigned<uword>();
    for (int i = 0; i < this->size; i++) {
        res->bsi.push_back(unbsi.bsi[0].And(this->bsi[i]));
    }
    res->size = this->size;
    BsiAttribute<uword> *temp;
    for(int j=1; j<unbsi.size; j++){

        temp = new BsiUnsigned<uword>();
       for (int i = 0; i < this->size; i++) {
           //temp->addSlice(unbsi.bsi[j].And(this->bsi[i]));
            //temp->bsi[i] = unbsi.bsi[j].And(this->bsi[i]);
           temp->bsi.push_back(unbsi.bsi[j].And(this->bsi[i]));
        }
        temp->size = this->size;
        temp->offset=j;
        res=res->SUM(temp);
    }

    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    return res;

};




/*
 * Dot product of two BSIs. Builds the sum aggregator directly, without the resulting multiplication vector.
 */

template <class uword>
long long int BsiUnsigned<uword>::dot(BsiAttribute<uword>* unbsi) const{
    long long int res =0;

    for(int j=0; j<unbsi->size; j++){
        for (int i = 0; i < this->size; i++) {
            if(j==0 && i==0)
                res = res + unbsi->bsi[j].And(this->bsi[i]).numberOfOnes();
            else
                res = res + unbsi->bsi[j].And(this->bsi[i]).numberOfOnes()*(2<<(j+i-1));
        }
    }
    return res;
};

template <class uword>
long long int BsiUnsigned<uword>::dot_withoutCompression(BsiAttribute<uword>* unbsi) const{
    long long int res =0;

    for(int j=0; j<unbsi->size; j++){
        for (int i = 0; i < this->size; i++) {
            if(j==0 && i==0)
                res = res + unbsi->bsi[j].andVerbatim(this->bsi[i]).numberOfOnes();
            else
                res = res + unbsi->bsi[j].andVerbatim(this->bsi[i]).numberOfOnes()*(2<<(j+i-1));
        }
    }
    return res;
};

//template <class uword>
//long BsiUnsigned<uword>::dotProduct(BsiAttribute<uword>* unbsi) const{
//    //long res =0;
//    long res = (this->bsi[0].And(unbsi->bsi[0])).numberOfOnes();
//    for (int i=1; i<unbsi->size; i++){
//        res = res + (this->bsi[0].And(unbsi->bsi[i])).numberOfOnes()*(2<<(i-1));
//    }
//    for (int i=1; i<this->size; i++){
//        res = res + (this->bsi[i].And(unbsi->bsi[0])).numberOfOnes()*(2<<(i-1));
//    }
//
//    for (int i=1; i<this->size; i++){
//        for (int j=1; j<unbsi->size; j++){
//            res = res + (this->bsi[i].And(unbsi->bsi[j])).numberOfOnes()*(2<<(j+i-1));
//        }
//    }
//    return res;
//};




template <class uword>
void BsiUnsigned<uword>::reset(){
    this->bsi.clear();
    this->size = 0;
    this->rows =0;
    this->index =0;
    int offset =0;
    int decimals = 0;
    
    this->existenceBitmap.reset();
    this->signe = false;
    this->firstSlice = false; //contains first slice
    this->lastSlice = false; //contains last slice

};

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::negate(){
    BsiAttribute<uword>* res = new BsiUnsigned<uword>();
    res->bsi = this->bsi;
    res->sign = new HybridBitmap<uword>(this->getNumberOfRows(),true);
    res->is_signed = true;
    res->twosComplement = false;
    res->setNumberOfRows(this->getNumberOfRows());
    return res;
};


template <class uword>
void BsiUnsigned<uword>::BitWords(std::vector<uword> &bitWords, long value, int offset){
    const uword one = 1;
    int i = 0;
    while (value > 0){
        bitWords[i] = (value & one) << offset;
        value = value/2;
        i++;
    }
}



template <class uword>
bool BsiUnsigned<uword>::append(long value){
    int offset = this->getNumberOfRows()%(sizeof(uword)*8);
    std::vector<uword> bitWords(this->bsi.size());
    BitWords(bitWords, value, offset);
    for (int i=0;i<this->bsi.size(); i++){
        if(this->bsi[i].verbatim == false){
            return false;
        }
    }
    int size = this->bsi[0].buffer.size()-1;
    if(offset == 0){
        for(int i=0; i<this->bsi.size(); i++){
            this->bsi[i].buffer.push_back(bitWords[i]);
            this->bsi[i].setSizeInBits(this->bsi[i].sizeInBits()+1);
        }
    }else{
        for(int i=0; i<this->bsi.size(); i++){
            this->bsi[i].buffer[size] = this->bsi[i].buffer.back() | bitWords[i];
            this->bsi[i].setSizeInBits(this->bsi[i].sizeInBits()+1);
        }
    }
    this->rows++;
    return true;
}

/*
 * Compares values stored in slice "index".
 * Returns -1 if this is less than a, 1 if this is greater than a, 0 otherwise
*/
template <class uword>
int BsiUnsigned<uword>::compareTo(BsiAttribute<uword> *a, int index) {
    for (int i=this->size; i>=0; i--) {
        if (this->bsi[i].get(index) != a->bsi[i].get(index)) {
            if (this->bsi[i].get(index) == 0) return -1;
            else return 1;
        }
    }
    return 0;
}

/*
 * divide funtion implementations
 */
template <class uword>
std::pair<BsiAttribute<uword>*, BsiAttribute<uword>*> BsiUnsigned<uword>::divide(
        const BsiAttribute<uword>& dividend,
        const BsiAttribute<uword>& divisor
) const {
    // Check for division by zero
    bool isDivisorZero = true;
    for (int i = 0; i < divisor.size; i++) {
        if (divisor.bsi[i].numberOfOnes() > 0) {
            isDivisorZero = false;
            break;
        }
    }

    if (isDivisorZero) {
        throw std::runtime_error("Division by zero");
    }

    // Initialize quotient and remainder
    BsiUnsigned<uword>* quotient = new BsiUnsigned<uword>();
    BsiUnsigned<uword>* remainder = new BsiUnsigned<uword>();

    quotient->existenceBitmap = dividend.existenceBitmap;
    quotient->rows = dividend.rows;
    quotient->index = dividend.index;
    quotient->firstSlice = true;
    quotient->lastSlice = true;

    remainder->existenceBitmap = dividend.existenceBitmap;
    remainder->rows = dividend.rows;
    remainder->index = dividend.index;
    remainder->firstSlice = true;
    remainder->lastSlice = true;

    int max_dividend_bits = 0;
    for (int i = dividend.size - 1; i >= 0; i--) {
        if (dividend.bsi[i].numberOfOnes() > 0) {
            max_dividend_bits = i + 1;
            break;
        }
    }

    int max_divisor_bits = 0;
    for (int i = divisor.size - 1; i >= 0; i--) {
        if (divisor.bsi[i].numberOfOnes() > 0) {
            max_divisor_bits = i + 1;
            break;
        }
    }

    int max_quotient_bits = max_dividend_bits;
    int max_remainder_bits = max_divisor_bits;

    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.verbatim = true;
    zeroBitmap.setSizeInBits(dividend.bsi[0].sizeInBits(), false);

    for (int i = 0; i < max_quotient_bits; i++) {
        HybridBitmap<uword> bitmap = zeroBitmap;
        quotient->addSlice(bitmap);
    }

    for (int i = 0; i < max_remainder_bits; i++) {
        HybridBitmap<uword> bitmap = zeroBitmap;
        remainder->addSlice(bitmap);
    }

    for (size_t rowIndex = 0; rowIndex < dividend.rows; rowIndex++) {
        if (!dividend.existenceBitmap.get(rowIndex)) {
            continue;
        }

        long dividendValue = 0;
        for (int i = 0; i < dividend.size; i++) {
            if (i < dividend.bsi.size() && dividend.bsi[i].get(rowIndex)) {
                dividendValue |= (1L << i);
            }
        }

        long divisorValue = 0;
        for (int i = 0; i < divisor.size; i++) {
            if (i < divisor.bsi.size() && divisor.bsi[i].get(rowIndex)) {
                divisorValue |= (1L << i);
            }
        }

        if (divisorValue == 0) {
            continue;
        }
        /*
         * Performing native operators division
         */
        long quotientValue = dividendValue / divisorValue;
        long remainderValue = dividendValue % divisorValue;

        int wordPos = rowIndex / (sizeof(uword) * 8);
        int bitOffset = rowIndex % (sizeof(uword) * 8);

        for (int i = 0; i < quotient->size; i++) {
            if (wordPos >= quotient->bsi[i].buffer.size()) {
                quotient->bsi[i].buffer.resize(wordPos + 1, 0);
            }
        }

        for (int i = 0; i < remainder->size; i++) {
            if (wordPos >= remainder->bsi[i].buffer.size()) {
                remainder->bsi[i].buffer.resize(wordPos + 1, 0);
            }
        }

        for (int i = 0; i < quotient->size; i++) {
            quotient->bsi[i].buffer[wordPos] &= ~(static_cast<uword>(1) << bitOffset);
        }

        for (int i = 0; i < remainder->size; i++) {
            remainder->bsi[i].buffer[wordPos] &= ~(static_cast<uword>(1) << bitOffset);
        }

        for (int i = 0; i < quotient->size; i++) {
            if ((quotientValue & (1L << i)) != 0) {
                quotient->bsi[i].buffer[wordPos] |= (static_cast<uword>(1) << bitOffset);
            }
        }

        for (int i = 0; i < remainder->size; i++) {
            if ((remainderValue & (1L << i)) != 0) {
                remainder->bsi[i].buffer[wordPos] |= (static_cast<uword>(1) << bitOffset);
            }
        }
    }

    for (int i = 0; i < quotient->size; i++) {
        quotient->bsi[i].setSizeInBits(dividend.bsi[0].sizeInBits());
    }

    for (int i = 0; i < remainder->size; i++) {
        remainder->bsi[i].setSizeInBits(dividend.bsi[0].sizeInBits());
    }

    for (int i = 0; i < quotient->size; i++) {
        quotient->bsi[i].density = quotient->bsi[i].numberOfOnes() / static_cast<double>(quotient->rows);
    }

    for (int i = 0; i < remainder->size; i++) {
        remainder->bsi[i].density = remainder->bsi[i].numberOfOnes() / static_cast<double>(remainder->rows);
    }

    return std::make_pair(quotient, remainder);
}

#endif /* BsiUnsigned_hpp */
