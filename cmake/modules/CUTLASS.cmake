# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

##############################################################################
# Provide:
#  - RAF_CUTLASS_LIBRARY
message("########## Configuring CUTLASS ##########")

# raf customized kernels
function(raf_customized_cutlass_kernels)
  find_package(Python3 3.5 COMPONENTS Interpreter REQUIRED)
  include(${PROJECT_SOURCE_DIR}/3rdparty/cutlass/CUDA.cmake)
  set(CUTLASS_GENERATOR_CUDA_COMPILER_VERSION ${CMAKE_CUDA_COMPILER_VERSION})
  set(CUTLASS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/cutlass/tools/library)
  execute_process(
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/scripts/src_codegen/cutlass
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=$PYTHONPATH:${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/library/scripts
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/src_codegen/cutlass/generator_ext.py
      --operations "conv2d,gemm"
      --curr-build-dir ${CUTLASS_BINARY_DIR}
      --generator-target library
      --architectures "${CUTLASS_NVCC_ARCHS}"
      --kernels ${CUTLASS_LIBRARY_KERNELS}
      --ignore-kernels ${CUTLASS_LIBRARY_IGNORE_KERNELS}
      --cuda-version ${CUTLASS_GENERATOR_CUDA_COMPILER_VERSION}
    RESULT_VARIABLE cutlass_lib_INSTANCE_GENERATION_RESULT
    OUTPUT_FILE ${CUTLASS_BINARY_DIR}/cutlass_library_instance_generation.log
    ERROR_FILE ${CUTLASS_BINARY_DIR}/cutlass_library_instance_generation.log
  )

  if(NOT cutlass_lib_INSTANCE_GENERATION_RESULT EQUAL 0)
    message(FATAL_ERROR "Error generating library instances: ${cutlass_lib_INSTANCE_GENERATION_RESULT}. See ${CUTLASS_BINARY_DIR}/cutlass_library_instance_generation.log")
  endif()

  # include auto-instantiated kernels in he CUTLASS Deliverables Library
  set(CUTLASS_LIBRARY_MANIFEST_CMAKE_FILE ${CUTLASS_BINARY_DIR}/generated/manifest.cmake)
  if(EXISTS "${CUTLASS_LIBRARY_MANIFEST_CMAKE_FILE}")
    include(${CUTLASS_LIBRARY_MANIFEST_CMAKE_FILE})
  else()
    message(STATUS "auto-generated library manifest cmake file (${CUTLASS_LIBRARY_MANIFEST_CMAKE_FILE}) not found.")
  endif()

  target_include_directories(
    cutlass_library_objs
    PRIVATE ${PROJECT_SOURCE_DIR}/include/
  )

  # Force O3 and no debug info to avoid undefined symbol error
  # in Debug build.
  target_compile_options(
    cutlass_library_objs
    PRIVATE
    -O3 -DNODEBUG
  )

  target_include_directories(
    cutlass_lib
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/library/src
  )

  # Force O3 and no debug info to avoid undefined symbol error
  # in Debug build.
  target_compile_options(
    cutlass_lib
    PRIVATE
    -O3 -DNODEBUG
  )

  cutlass_target_sources(
    cutlass_lib
    PRIVATE
    ${PROJECT_SOURCE_DIR}/src/op/dialect/cutlass/operation_table_ext.cu
    ${PROJECT_SOURCE_DIR}/src/op/dialect/cutlass/singleton_ext.cu
  )
endfunction()

if (${RAF_USE_CUTLASS} STREQUAL "OFF")
  message(STATUS "Build without CUTLASS support")
  set(RAF_CUTLASS_LIBRARY "")
else()
  set(CUTLASS_NVCC_ARCHS ${RAF_CUDA_ARCH} CACHE STRING "The SM architectures requested.")
  if (${RAF_USE_CUDA} STREQUAL "OFF")
    message(FATAL_ERROR "Cannot enable CUTLASS without using CUDA.")
  endif()

  set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "Enable CUTLASS Examples")
  set(CUTLASS_ENABLE_LIBRARY ON CACHE BOOL "Enable CUTLASS Library")
  set(CUTLASS_ENABLE_PROFILER OFF CACHE BOOL "Enable CUTLASS Profiler")
  set(CUTLASS_LIBRARY_KERNELS "sgemm,s884gemm,h884gemm,s*fprop,h*fprop" CACHE STRING "Comma delimited list of kernel name filters. If unspecified, only the largest tile size is enabled. If 'all' is specified, all kernels are enabled.")
  # The ignored ops are complex ops, sparse ops, and integer ops.
  set(CUTLASS_LIBRARY_IGNORE_KERNELS "complex,sp,i8816,i8832" CACHE STRING "Comma delimited list of kernel names to exclude from build.")
  set(CUTLASS_LIBRARY_KERNELS "invalid_kernel_name")
  add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/cutlass/)
  message(STATUS "Set CUTLASS_NVCC_ARCHS=${CUTLASS_NVCC_ARCHS}")
  unset(CUTLASS_LIBRARY_KERNELS)
  raf_customized_cutlass_kernels()

  set_target_properties(cutlass_lib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set(RAF_CUTLASS_LIBRARY "cutlass_lib")
endif()

message("########## Done Configuring CUTLASS ##########")
