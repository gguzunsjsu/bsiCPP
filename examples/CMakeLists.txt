project(test)
include_directories(include)
set(CMAKE_PREFIX_PATH '/Users/akankshajoshi/Downloads/libtorch')
find_package(Torch REQUIRED)
file(GLOB SOURCES
    "testBSIAttributeBuilding.h"
    "testBSIAttributeBuilding.cpp"
    "test.cpp"
    "testBSI.hpp"
)
add_executable(tests ${SOURCES} zipf.h)
add_executable(example example.cpp)

target_link_libraries(example PUBLIC bsi)
target_link_libraries(tests PUBLIC bsi)
