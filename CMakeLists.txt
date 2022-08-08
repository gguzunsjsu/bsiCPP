# Copyright (c) Gheorghi Guzun
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 2.9)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MACOSX_RPATH OFF)
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

project(BSIHybrid)

add_library(bsi INTERFACE)
target_include_directories(
        bsi
        INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/bsi>
        $<INSTALL_INTERFACE:include/bsi>
)

add_library(hybridBitmap INTERFACE)
target_include_directories(
        hybridBitmap
        INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/hybridBitmap>
        $<INSTALL_INTERFACE:include/hybridBitmap>
)


enable_testing()

add_subdirectory(examples)

