set(CMAKE_PREFIX_PATH '/Users/akankshajoshi/Downloads/libtorch')
# Find and configure libtorch
find_package(Torch REQUIRED)
add_library(bsi INTERFACE)
target_include_directories(
        bsi
        INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/bsi>
        $<INSTALL_INTERFACE:bsi>
)

# Link your target with libtorch
target_link_libraries(bsi INTERFACE ${TORCH_LIBRARIES})