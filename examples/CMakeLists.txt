project(test)
include_directories(include)
file(GLOB SOURCES
    "testBSIAttributeBuilding.h"
    "testBSIAttributeBuilding.cpp"
    "test.cpp"
    "testBSI.hpp"
)
project(testHorizontalDotProduct)
include_directories(include)
file(GLOB SOURCES
        "testBSIAttributeBuilding.h"
        "testBSIAttributeBuilding.cpp"
        "testHorizontalDotProduct.cpp"
        "testBSI.hpp"
)
add_executable(tests ${SOURCES} zipf.h)
add_executable(example example.cpp)
add_executable(testHorizontalDotProduct ${SOURCES} zipf.h)
target_link_libraries(example PUBLIC bsi)
target_link_libraries(tests PUBLIC bsi)
target_link_libraries(testHorizontalDotProduct PUBLIC bsi)