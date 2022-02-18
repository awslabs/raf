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
