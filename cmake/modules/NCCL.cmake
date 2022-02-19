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
#  - MNM_NCCL_LIBRARY
#  - MNM_NCCL_INCLUDE

if (${MNM_USE_NCCL} STREQUAL "OFF")
  message(STATUS "Build without NCCL support")
  set(MNM_NCCL_INCLUDE "")
  set(MNM_NCCL_LIBRARY "")
else()
  if (${MNM_USE_CUDA} STREQUAL "OFF")
    message(FATAL_ERROR "Cannot enable NCCL without using CUDA.")
  endif()
  if (${MNM_USE_MPI} STREQUAL "OFF")
    message(FATAL_ERROR "Cannot enable NCCL without using MPI now.")
  endif()
  find_package(NCCL REQUIRED)
  set(MNM_NCCL_INCLUDE ${NCCL_INCLUDE_DIRS})
  message(STATUS "Found MNM_NCCL_INCLUDE = ${MNM_NCCL_INCLUDE}")
  set(MNM_NCCL_LIBRARY ${NCCL_LIBRARIES})
  message(STATUS "Found MNM_NCCL_LIBRARY = ${MNM_NCCL_LIBRARY}")
endif()