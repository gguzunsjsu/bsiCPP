# Copyright (c) Gheorghi Guzun
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.1...3.18)
# Fallback for using newer policies on CMake <3.12.
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()


project(bsi
  VERSION ${VERSION}
  DESCRIPTION "Bit-sliced columnar vectors with compression"
  HOMEPAGE_URL "https://github.com/gguzunsjsu/bsiCPP"
  LANGUAGES CXX)
include(GNUInstallDirs)

# Include guards
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(
            FATAL_ERROR
            "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
    )
endif()

set(CMAKE_CXX_STANDARD 20)

add_library(bsi INTERFACE)
target_include_directories(
        bsi
        INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/bsi>
        $<INSTALL_INTERFACE:bsi>
)

add_library(hybridBitmap INTERFACE)
target_include_directories(
        hybridBitmap
        INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/hybridBitmap>
        $<INSTALL_INTERFACE:hybridBitmap>
)