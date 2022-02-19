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

# Provides:
#  - MNM_CUDA_INCLUDE
#
#  See https://cmake.org/cmake/help/latest/module/FindCUDA.html
if (${MNM_USE_CUDA} STREQUAL "OFF")
  message(STATUS "Build without CUDA")
  set(MNM_CUDA_INCLUDE "")
else()
  find_package(CUDA REQUIRED)
  message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
  set(MNM_CUDA_INCLUDE ${CUDA_INCLUDE_DIRS})
  message(STATUS "Found MNM_CUDA_INCLUDE = ${MNM_CUDA_INCLUDE}")
endif()
