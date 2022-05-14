# Copyright (c) Gheorghi Guzun
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.12)
include(bsiCPP)
bsiCPP_getversion(VERSION)

project(bsiCPP
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

include_directories(.)
include_directories(bsiCPP)
include_directories(bsiCPP/hybridBitmap)