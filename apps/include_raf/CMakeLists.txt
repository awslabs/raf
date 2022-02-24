# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.8)
project(include_raf C CXX)

add_executable(include_raf main.cc)
target_include_directories(include_raf PRIVATE ${RAF_INCLUDE_DIRS})
target_link_libraries(include_raf PRIVATE raf ${RAF_LINK_LIBS})
target_compile_features(include_raf PRIVATE cxx_std_14)

set_target_properties(include_raf PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)
