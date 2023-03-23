project(test)
include_directories(include)
file(GLOB SOURCES
    "testBSIAttributeBuilding.h"
    "testBSIAttributeBuilding.cpp"
    "test.cpp"
)
add_executable(tests ${SOURCES})
add_executable(example example.cpp)
target_link_libraries(example PUBLIC bsi)
target_link_libraries(tests PUBLIC bsi)
