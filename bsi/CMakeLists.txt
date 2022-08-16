add_library(bsi INTERFACE)
target_include_directories(
        bsi
        INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/bsi>
        $<INSTALL_INTERFACE:bsi>
)