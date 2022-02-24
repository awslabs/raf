# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

##############################################################################
# Provide:
#  - RAF_CUBLAS_LIBRARY

if (${RAF_USE_CUBLAS} STREQUAL "OFF")
  message(STATUS "Build without cuBLAS support")
  set(RAF_CUBLAS_LIBRARY "")
else()
  if (${RAF_USE_CUDA} STREQUAL "OFF")
    message(FATAL_ERROR "Cannot enable cuBLAS without using CUDA.")
  endif()
  set(RAF_CUBLAS_LIBRARY ${CUDA_CUBLAS_LIBRARIES})
  message(STATUS "Found RAF_CUBLAS_LIBRARY = ${RAF_CUBLAS_LIBRARY}")
endif()
