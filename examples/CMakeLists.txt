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
        division_test.cpp
        testing_skeletonCode/binaryLongDivision.cpp
        divison_test_new.cpp
        RandomBSIGeneration.cpp
        zipf_testRuns/zipf_10k.cpp
        zipf_testRuns/zipf_1M.cpp
        zipf_testRuns/zipf_100_to_1M/zipf_100_1M.cpp)

target_link_libraries(example PUBLIC bsi)
target_link_libraries(tests PUBLIC bsi)
