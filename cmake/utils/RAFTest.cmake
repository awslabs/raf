# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

macro(raf_test TESTNAME)
  add_executable(${TESTNAME} EXCLUDE_FROM_ALL ${ARGN})

  if (${RAF_USE_CUDA} STREQUAL "OFF")
    set(TEST_CUDA_INCLUDE, "")
  else()
    set(TEST_CUDA_INCLUDE ${RAF_CUDA_INCLUDE})
  endif()


  target_include_directories(${TESTNAME}
    PRIVATE
      ${RAF_INCLUDE_DIRS}
      ${gtest_SOURCE_DIR}/include
      ${TEST_CUDA_INCLUDE}
  )

  target_link_libraries(${TESTNAME}
    PRIVATE
      gtest
      gmock
      gtest_main
      raf
      ${RAF_LINK_LIBS}
      ${RAF_BACKEND_LINK_LIBS}
  )
  target_compile_options(${TESTNAME} PRIVATE ${RAF_CXX_FLAGS})
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
    FOLDER raf-cpptest
  )
  add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
endmacro()
