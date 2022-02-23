# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

##############################################################################
# Provide:
#  - MNM_CUBLAS_LIBRARY

if (${MNM_USE_CUBLAS} STREQUAL "OFF")
  message(STATUS "Build without cuBLAS support")
  set(MNM_CUBLAS_LIBRARY "")
else()
  if (${MNM_USE_CUDA} STREQUAL "OFF")
    message(FATAL_ERROR "Cannot enable cuBLAS without using CUDA.")
  endif()
  set(MNM_CUBLAS_LIBRARY ${CUDA_CUBLAS_LIBRARIES})
  message(STATUS "Found MNM_CUBLAS_LIBRARY = ${MNM_CUBLAS_LIBRARY}")
endif()
