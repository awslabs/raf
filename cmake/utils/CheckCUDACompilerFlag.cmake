# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
