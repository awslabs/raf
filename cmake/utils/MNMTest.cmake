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

macro(mnm_test TESTNAME)
  add_executable(${TESTNAME} EXCLUDE_FROM_ALL ${ARGN})

  if (${MNM_USE_CUDA} STREQUAL "OFF")
    set(TEST_CUDA_INCLUDE, "")
  else()
    set(TEST_CUDA_INCLUDE ${MNM_CUDA_INCLUDE})
  endif()


  target_include_directories(${TESTNAME}
    PRIVATE
      ${MNM_INCLUDE_DIRS}
      ${gtest_SOURCE_DIR}/include
      ${TEST_CUDA_INCLUDE}
  )

  target_link_libraries(${TESTNAME}
    PRIVATE
      gtest
      gmock
      gtest_main
      mnm
      ${MNM_LINK_LIBS}
      ${MNM_BACKEND_LINK_LIBS}
  )
  target_compile_options(${TESTNAME} PRIVATE ${MNM_CXX_FLAGS})
  target_compile_features(${TESTNAME} PRIVATE cxx_std_14)
  set_target_properties(${TESTNAME} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON
    FOLDER mnm-cpptest
  )
  add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
endmacro()
