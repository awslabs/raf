# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

##############################################################################
# Provide:
#  - RAF_CUDNN_FOUND
#  - RAF_CUDNN_VERSION
#  - RAF_CUDNN_INCLUDE
#  - RAF_CUDNN_LIBRARY

include(FindPackageHandleStandardArgs)
if (${RAF_USE_CUDNN} STREQUAL "OFF")
  message(STATUS "Build without cuDNN support")
  set(RAF_CUDNN_FOUND FALSE)
  set(RAF_CUDNN_VERSION "")
  set(RAF_CUDNN_INCLUDE "")
  set(RAF_CUDNN_LIBRARY "")
else()
  if (${RAF_USE_CUDA} STREQUAL "OFF")
    message(FATAL_ERROR "Cannot enable cuDNN without using CUDA.")
  endif()

  if (${RAF_USE_CUDNN} STREQUAL "ON")
    set(__hint_dir "")
  else()
    set(__hint_dir ${RAF_USE_CUDNN})
  endif()

  find_path(RAF_CUDNN_INCLUDE cudnn.h
    HINTS ${__hint_dir} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES cuda/include include)

  find_library(RAF_CUDNN_LIBRARY cudnn
    HINTS ${__hint_dir} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

  find_package_handle_standard_args(RAF_CUDNN DEFAULT_MSG RAF_CUDNN_INCLUDE RAF_CUDNN_LIBRARY)
  if (NOT ${RAF_CUDNN_FOUND})
    message(FATAL_ERROR "Please specify the path to cuDNN by setting RAF_USE_CUDNN")
  endif()

  if (EXISTS ${RAF_CUDNN_INCLUDE}/cudnn_version.h)
    file(READ ${RAF_CUDNN_INCLUDE}/cudnn_version.h __content)
  else()
    file(READ ${RAF_CUDNN_INCLUDE}/cudnn.h __content)
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
  set(RAF_CUDNN_VERSION "${__major}.${__minor}.${__patch}")
  message(STATUS "Found RAF_CUDNN_VERSION = v${RAF_CUDNN_VERSION}")
  message(STATUS "Found RAF_CUDNN_INCLUDE = ${RAF_CUDNN_INCLUDE}")
  message(STATUS "Found RAF_CUDNN_LIBRARY = ${RAF_CUDNN_LIBRARY}")
endif()
