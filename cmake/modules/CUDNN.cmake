# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

##############################################################################
# Provide:
#  - MNM_CUDNN_FOUND
#  - MNM_CUDNN_VERSION
#  - MNM_CUDNN_INCLUDE
#  - MNM_CUDNN_LIBRARY

include(FindPackageHandleStandardArgs)
if (${MNM_USE_CUDNN} STREQUAL "OFF")
  message(STATUS "Build without cuDNN support")
  set(MNM_CUDNN_FOUND FALSE)
  set(MNM_CUDNN_VERSION "")
  set(MNM_CUDNN_INCLUDE "")
  set(MNM_CUDNN_LIBRARY "")
else()
  if (${MNM_USE_CUDA} STREQUAL "OFF")
    message(FATAL_ERROR "Cannot enable cuDNN without using CUDA.")
  endif()

  if (${MNM_USE_CUDNN} STREQUAL "ON")
    set(__hint_dir "")
  else()
    set(__hint_dir ${MNM_USE_CUDNN})
  endif()

  find_path(MNM_CUDNN_INCLUDE cudnn.h
    HINTS ${__hint_dir} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES cuda/include include)

  find_library(MNM_CUDNN_LIBRARY cudnn
    HINTS ${__hint_dir} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

  find_package_handle_standard_args(MNM_CUDNN DEFAULT_MSG MNM_CUDNN_INCLUDE MNM_CUDNN_LIBRARY)
  if (NOT ${MNM_CUDNN_FOUND})
    message(FATAL_ERROR "Please specify the path to cuDNN by setting MNM_USE_CUDNN")
  endif()

  if (EXISTS ${MNM_CUDNN_INCLUDE}/cudnn_version.h)
    file(READ ${MNM_CUDNN_INCLUDE}/cudnn_version.h __content)
  else()
    file(READ ${MNM_CUDNN_INCLUDE}/cudnn.h __content)
  endif()
  string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)" __major "${__content}")
  string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1" __major "${__major}")
  string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)" __minor "${__content}")
  string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1" __minor "${__minor}")
  string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)" __patch "${__content}")
  string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1" __patch "${__patch}")
  if(NOT __major)
    message(FATAL_ERROR "Cannot find cuDNN version")
  endif()
  set(MNM_CUDNN_VERSION "${__major}.${__minor}.${__patch}")
  message(STATUS "Found MNM_CUDNN_VERSION = v${MNM_CUDNN_VERSION}")
  message(STATUS "Found MNM_CUDNN_INCLUDE = ${MNM_CUDNN_INCLUDE}")
  message(STATUS "Found MNM_CUDNN_LIBRARY = ${MNM_CUDNN_LIBRARY}")
endif()
