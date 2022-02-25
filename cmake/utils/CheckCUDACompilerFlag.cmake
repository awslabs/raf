# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

macro(check_cuda_compiler_flag _flag _result)
  message(STATUS "Performing Test ${_result}")
  set(__cmd "${CUDA_NVCC_EXECUTABLE} ${_flag}")
  separate_arguments(__cmd)
  execute_process(COMMAND ${__cmd} ERROR_VARIABLE NVCC_OUT)
  if ("${NVCC_OUT}" MATCHES "Unknown option")
    message(STATUS "Performing Test ${_result} - Failed")
    set(${_result} 0)
  else()
    message(STATUS "Performing Test ${_result} - Success")
    set(${_result} 1)
  endif()
endmacro()
