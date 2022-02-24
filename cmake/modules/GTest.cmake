# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

if (${RAF_USE_GTEST} STREQUAL "ON")
  include(CTest)
  include(${PROJECT_SOURCE_DIR}/cmake/utils/RAFTest.cmake)
  add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/googletest/ EXCLUDE_FROM_ALL)
elseif (${RAF_USE_GTEST} STREQUAL "OFF")
  message(STATUS "Build without googletest")
else()
  message(FATAL_ERROR "Cannot recognize RAF_USE_GTEST = ${RAF_USE_GTEST}")
endif()
