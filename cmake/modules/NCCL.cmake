# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

##############################################################################
# Provide:
#  - RAF_NCCL_LIBRARY
#  - RAF_NCCL_INCLUDE

if (${RAF_USE_NCCL} STREQUAL "OFF")
  message(STATUS "Build without NCCL support")
  set(RAF_NCCL_INCLUDE "")
  set(RAF_NCCL_LIBRARY "")
else()
  if (${RAF_USE_CUDA} STREQUAL "OFF")
    message(FATAL_ERROR "Cannot enable NCCL without using CUDA.")
  endif()
  if (${RAF_USE_MPI} STREQUAL "OFF")
    message(FATAL_ERROR "Cannot enable NCCL without using MPI now.")
  endif()
  find_package(NCCL REQUIRED)
  set(RAF_NCCL_INCLUDE ${NCCL_INCLUDE_DIRS})
  message(STATUS "Found RAF_NCCL_INCLUDE = ${RAF_NCCL_INCLUDE}")
  set(RAF_NCCL_LIBRARY ${NCCL_LIBRARIES})
  message(STATUS "Found RAF_NCCL_LIBRARY = ${RAF_NCCL_LIBRARY}")
endif()