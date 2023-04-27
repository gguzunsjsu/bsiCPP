#ifndef testBSI_H
#define testBSI_H
template <class uword = uint64_t>
class testBSI {
public:
    int range;
    double compressionThreshold;
    vector<long> array;
    BsiSigned<uword> signed_bsi;
    BsiAttribute<uword>* bsi_attribute;
    int numberOfElementsInTheArray;
    //Constructors
    testBSI<uword>();
    testBSI<uword>(int range);
    void buildBSIAttribute();
};

#endif
