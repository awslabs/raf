macro(mnm_test TESTNAME)
  add_executable(${TESTNAME} EXCLUDE_FROM_ALL ${ARGN})
  target_include_directories(${TESTNAME}
    PRIVATE
      ${MNM_INCLUDE_DIRS}
      ${gtest_SOURCE_DIR}/include
  )
  if (${MNM_USE_CUDA} STREQUAL "OFF")
    set(CUDART_LIB "")
  else()
    set(CUDART_LIB "cudart")
  endif()
  target_link_libraries(${TESTNAME}
    PRIVATE
      gtest
      gmock
      gtest_main
      mnm
      ${MNM_LINK_LIBS}
      ${CUDART_LIB}
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
    FOLDER mnm-cpptest
  )
  add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
endmacro()
