# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Find the nccl libraries
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT_DIR: Base directory where all NCCL components are found
#  NCCL_INCLUDE_DIR: Directory where NCCL header is found
#  NCCL_LIB_DIR: Directory where NCCL library is found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install NCCL in the same location as the CUDA toolkit.
# See https://github.com/caffe2/caffe2/issues/1601

find_path(NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS
  ${NCCL_INCLUDE_DIR}
  ${NCCL_ROOT_DIR}
  ${NCCL_ROOT_DIR}/include
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  $ENV{NCCL_DIR}/include
  )

find_library(NCCL_LIBRARIES
  NAMES nccl
  HINTS
  ${NCCL_LIB_DIR}
  ${NCCL_ROOT_DIR}
  ${NCCL_ROOT_DIR}/lib
  ${NCCL_ROOT_DIR}/lib/x86_64-linux-gnu
  ${NCCL_ROOT_DIR}/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  $ENV{NCCL_DIR}/lib
  )

# if not found in any of the above paths, finally, check in the /usr/local/cuda for UNIX systems
if (UNIX)
  set (search_paths "/usr/local/cuda")

  find_path(NCCL_INCLUDE_DIRS
    NAMES nccl.h
    PATHS ${search_paths}
    PATH_SUFFIXES include
  )

  find_library(NCCL_LIBRARIES
    NAMES nccl
    PATHS ${search_paths}
    PATH_SUFFIXES lib
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIRS NCCL_LIBRARIES)

if(NCCL_FOUND)
  message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
  mark_as_advanced(NCCL_ROOT_DIR NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
endif()

