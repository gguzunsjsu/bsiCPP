project(test)
include_directories(include)
file(GLOB SOURCES
    "testBSIAttributeBuilding.h"
    "testBSIAttributeBuilding.cpp"
    "test.cpp"
    "testBSI.hpp"
)
add_executable(tests ${SOURCES} zipf.h)
add_executable(example example.cpp
        example_corrected.cpp
        example_add.cpp
        example_multiply.cpp
        division_test.cpp)

target_link_libraries(example PUBLIC bsi)
target_link_libraries(tests PUBLIC bsi)
