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

if (${MNM_USE_GTEST} STREQUAL "ON")
  include(CTest)
  include(${PROJECT_SOURCE_DIR}/cmake/utils/MNMTest.cmake)
  add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/googletest/ EXCLUDE_FROM_ALL)
elseif (${MNM_USE_GTEST} STREQUAL "OFF")
  message(STATUS "Build without googletest")
else()
  message(FATAL_ERROR "Cannot recognize MNM_USE_GTEST = ${MNM_USE_GTEST}")
endif()
