# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Provides:
#  - RAF_CUDA_INCLUDE
#
#  See https://cmake.org/cmake/help/latest/module/FindCUDA.html
if (${RAF_USE_CUDA} STREQUAL "OFF")
  message(STATUS "Build without CUDA")
  set(RAF_CUDA_INCLUDE "")
else()
  find_package(CUDA REQUIRED)
  message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
  set(RAF_CUDA_INCLUDE ${CUDA_INCLUDE_DIRS})
  message(STATUS "Found RAF_CUDA_INCLUDE = ${RAF_CUDA_INCLUDE}")
endif()
