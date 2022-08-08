
#ifndef UnitTestsOfHybridBitmap_hpp
#define UnitTestsOfHybridBitmap_hpp

#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include "hybridbitmap.h"
#include "boolarray.h"

#define SSTR(x) (to_string(x))

using namespace std;

static string testfailed = "---\ntest failed.\n\n\n\n\n\n";

// for Microsoft compilers
#if _MSC_VER >= 1400
#define unlink _unlink
#endif

template <class uword>
class UnitTestOfHybridbitmap{

public:
    
 bool testadd() {
    cout << "testing testadd()"<< endl;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(4, 1, 10, 11, 65);
     b.add(2);
     if(b.compareBitmap(b3)){
         return true;
     }

    return false;
    
}


 bool testaddVerbatim() {
    cout << "testing testaddVerbatim()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testAnd() {
    cout << "testing testAnd()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testandC() {
    cout << "testing testandC()" << endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testandHybrid() {
    cout << "testing testandHybrid()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testandHybridCompress() {
    cout << "testing testandHybridCompress()" << endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testandNot() {
    cout << "testing testandNot()" << endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    return true;
}

 bool testandNotC() {
    cout << "testing testandNotC()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testandNotHybrid() {
    cout << "testing testandNotHybrid()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testandNotHybridCompress() {
    cout << "testing testandNotHybridCompress()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testandNotV() {
    cout << "testing testandNotV()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


bool testandNotVerbatim() {
    cout << "testing testandNotV()" << endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testandNotVerbatimCompress() {
    cout << "testing testandNotVerbatimCompress()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testandVerbatim() {
    cout << "testing testandVerbatim()" << endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testandVerbatimCompress() {
    cout << "testing testandVerbatimCompress()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testmaj() {
    cout << "testing testmaj()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}

 bool testOr() {
    cout << "testing testOr()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testorAndNotV() {
    cout << "testing testorAndNotV()" << endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testorAndV() {
    cout << "testing testorAndV()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testorHybrid() {
    cout << "testing testorHybrid()" << endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testorHybridCompress() {
    cout << "testing testorHybridCompress()" << endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}

 bool testorVerbatim() {
    cout << "testing testorVerbatim()" << endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testorVerbatimCompress() {
    cout << "testing testorVerbatimCompress()" << endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}

 bool testXor() {
    cout << "testing testXor()" << endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testxorC() {
    cout << "testing testxorC()" << endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testxorHybrid() {
    cout << "testing testxorHybrid()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testxorHybridCompress() {
    cout << "testing testxorHybridCompress()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testxorNot() {
    cout << "testing testxorNot()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testxorNotHybrid() {
    cout << "testing testxorNotHybrid()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}


 bool testxorNotVerbatim() {
    cout << "testing testxorNotVerbatim()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}

 bool testxorVerbatim() {
    cout << "testing testxorVerbatim()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}

 bool testxorVerbatimCompress() {
    cout << "testing testxorVerbatimCompress()"<< endl;
    HybridBitmap<uword> b1;
    HybridBitmap<uword> b = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    HybridBitmap<uword> b3 = HybridBitmap<uword>::bitmapOf(3, 1, 10, 11);
    
    return true;
}

};
#endif
