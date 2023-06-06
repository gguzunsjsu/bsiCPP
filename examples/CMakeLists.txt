project(test)
include_directories(include)
#[[file(GLOB SOURCES
    "testBSIAttributeBuilding.h"
    "testBSIAttributeBuilding.cpp"
    "test.cpp"
    "testBSI.hpp"
)]]
add_executable(tests ${SOURCES} testBsiSigned.cpp testBsiSigned.h)
#add_executable(example example.cpp)

#target_link_libraries(example PUBLIC bsi)
target_link_libraries(tests PUBLIC bsi)
