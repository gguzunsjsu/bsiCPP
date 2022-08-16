# Copyright (c) Gheorghi Guzun
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 2.9)
#set(CMAKE_CXX_STANDARD 11)
#et(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_MACOSX_RPATH OFF)
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

project(BSIHybrid)


enable_testing()

add_subdirectory(bsi)
add_subdirectory(examples)


