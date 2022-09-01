# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)
message(STATUS "Building RAF with CMake version: ${CMAKE_VERSION}")
project(raf C CXX)

################# User-defined configurations from config.cmake #################
include(${PROJECT_SOURCE_DIR}/cmake/utils/RAFCmakeUtils.cmake)
raf_option(RAF_USE_LLVM "RAF depends on LLVM to do code generation. Option: [ON/OFF/Path-to-llvm-config-executable]" ON)
raf_option(RAF_USE_CUDA "Build RAF with CUDA. Option: [ON/OFF]" OFF)
raf_option(RAF_USE_CUDNN "Build RAF with cuDNN. Option: [ON/OFF/Path-to-cuDNN]" OFF)
raf_option(RAF_USE_CUTLASS "Build RAF with CUTLASS. Option: [ON/OFF/NVCC-Arch]" OFF)
raf_option(RAF_CUDA_ARCH "Specify the CUDA architecture" 70)
raf_option(RAF_USE_MPI "Build RAF with MPI. Option: [ON/OFF]" OFF)
raf_option(RAF_USE_NCCL "Build RAF with NCCL. Option: [ON/OFF]" OFF)
raf_option(RAF_USE_CUBLAS "Build RAF with cuBLAS. Option: [ON/OFF]" OFF)
raf_option(RAF_USE_GTEST "Build cpptests for RAF. Option: [ON/OFF]" OFF)
raf_option(RAF_USE_SANITIZER "Build RAF with sanitizer. Option: [OFF/ASAN/MSAN/TSAN/UBSAN]" OFF)
raf_find_config()

################# Modules and Third-Party Targets #################
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules;${CMAKE_MODULE_PATH}")
include(${PROJECT_SOURCE_DIR}/cmake/modules/Threads.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/Git.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/CUDA.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/CUBLAS.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/CUDNN.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/CUTLASS.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/Sanitizer.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/TVM.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/GTest.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/MPI.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/NCCL.cmake)

################# Primary Target: RAF #################
set(RAF_INCLUDE_DIRS
  ${CMAKE_CURRENT_LIST_DIR}/include/
  ${CMAKE_CURRENT_LIST_DIR}/3rdparty/tvm/include/
  ${CMAKE_CURRENT_LIST_DIR}/3rdparty/tvm/src
  ${CMAKE_CURRENT_LIST_DIR}/3rdparty/tvm/3rdparty/compiler-rt/
  ${CMAKE_CURRENT_LIST_DIR}/3rdparty/tvm/3rdparty/dlpack/include/
  ${CMAKE_CURRENT_LIST_DIR}/3rdparty/tvm/3rdparty/dmlc-core/include/
  ${CMAKE_CURRENT_LIST_DIR}/3rdparty/cutlass/include/
  ${CMAKE_CURRENT_LIST_DIR}/3rdparty/cutlass/tools/library/include/
  ${CMAKE_CURRENT_LIST_DIR}/3rdparty/cutlass/tools/util/include/
)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message(STATUS "Build in Debug mode")
else()
  message(STATUS "Build in Release mode")
endif()
set(RAF_CXX_FLAGS -fPIC)
set(RAF_CUDA_FLAGS -Xcompiler=-Wall -Xcompiler=-fPIC)

set(RAF_BACKEND_INCLUDE_DIRS
  ${RAF_CUDA_INCLUDE}
  ${RAF_CUDNN_INCLUDE}
  ${RAF_NCCL_INCLUDE}
  ${RAF_MPI_INCLUDE}
)

set(RAF_LINK_LIBS
  tvm
  Threads::Threads
  ${RAF_CUTLASS_LIBRARY}
  ${CMAKE_DL_LIBS}
)

set(RAF_BACKEND_LINK_LIBS
  ${RAF_CUDNN_LIBRARY}
  ${RAF_CUBLAS_LIBRARY}
  ${RAF_NCCL_LIBRARY}
  ${RAF_MPI_LIBRARY}
)

set_property(
  SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/impl/build_info.cc
  APPEND
  PROPERTY COMPILE_DEFINITIONS
  RAF_GIT_VERSION="${RAF_GIT_VERSION}"
  RAF_CUDA_VERSION="${CUDA_VERSION_STRING}"
  RAF_USE_LLVM="${RAF_USE_LLVM}"
  RAF_USE_CUBLAS="${RAF_USE_CUBLAS}"
  RAF_USE_CUDNN="${RAF_USE_CUDNN}"
  RAF_CUDNN_VERSION="${RAF_CUDNN_VERSION}"
  RAF_CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  RAF_USE_CUTLASS="${RAF_USE_CUTLASS}"
)

file(GLOB_RECURSE RAF_CXX_SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/src/*/*.cc
)
file(GLOB_RECURSE RAF_EXCLUDE_CXX_SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/src/device_api/cuda/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/distributed/cuda/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/cublas/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/cuda/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/cudnn/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/cutlass/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/nccl/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/profiler/cuda/*.cc
)
list(REMOVE_ITEM RAF_CXX_SOURCE_FILES ${RAF_EXCLUDE_CXX_SOURCE_FILES})

if (${RAF_USE_CUDA} STREQUAL "OFF")
  set(RAF_CUDA_SOURCE_FILES "")
  set(RAF_CUDA_KERNEL_FILES "")
  set(RAF_CUDA_FLAGS "")
else ()
  message(STATUS "Build with CUDA enabled")
  message(STATUS "Build for CUDA architecture: ${RAF_CUDA_ARCH}")
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES ${RAF_CUDA_ARCH})
  set(RAF_CXX_FLAGS ${RAF_CXX_FLAGS} -DRAF_USE_CUDA)

  foreach(ARCH ${RAF_CUDA_ARCH})
    set(CODES)
    list(APPEND CODES sm_${ARCH})
    list(APPEND CODES compute_${ARCH})
    list(JOIN CODES "," CODES_STR)
    list(APPEND RAF_CUDA_ARCH_LIST "-gencode=arch=compute_${ARCH},code=\[${CODES_STR}\]")
  endforeach()
  set(RAF_CUDA_FLAGS ${RAF_CUDA_FLAGS} -DRAF_USE_CUDA ${RAF_CUDA_ARCH_LIST})

  file(GLOB_RECURSE RAF_CUDA_SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/device_api/cuda/*.cc
    ${CMAKE_CURRENT_LIST_DIR}/src/profiler/cuda/*.cc
    ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/cuda/*.cc
  )

  file(GLOB_RECURSE RAF_CUDA_KERNEL_FILES
	${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/cuda/kernels/*.cu
  )
endif()

if (${RAF_USE_CUDNN} STREQUAL "OFF")
  set(RAF_CUDNN_SOURCE_FILES "")
else()
  set(RAF_CXX_FLAGS ${RAF_CXX_FLAGS} -DRAF_CXX_USE_CUDNN)
  file(GLOB_RECURSE RAF_CUDNN_SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/cudnn/*.cc
  )
endif()

if (${RAF_USE_CUBLAS} STREQUAL "OFF")
  set(RAF_CUBLAS_SOURCE_FILES "")
else()
  file(GLOB_RECURSE RAF_CUBLAS_SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/cublas/*.cc
  )
endif()

if (${RAF_USE_CUTLASS} STREQUAL "OFF")
  set(RAF_CUTLASS_SOURCE_FILES "")
else()
  file(GLOB_RECURSE RAF_CUTLASS_SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/cutlass/*.cc
  )
endif()

if (${RAF_USE_NCCL} STREQUAL "OFF")
  set(RAF_NCCL_SOURCE_FILES "")
else ()
  set(RAF_CXX_FLAGS ${RAF_CXX_FLAGS} -DRAF_USE_NCCL)
  file(GLOB_RECURSE RAF_NCCL_SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/distributed/cuda/nccl*.cc
    ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/nccl/*.cc
  )
endif()

if (${RAF_USE_MPI} STREQUAL "OFF")
  set(RAF_MPI_SOURCE_FILES "")
else ()
  set(RAF_CXX_FLAGS ${RAF_CXX_FLAGS} -DRAF_USE_MPI)
  file(GLOB_RECURSE RAF_MPI_SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/distributed/cuda/mpi*.cc
  )
endif()

set(RAF_SOURCE_FILES
  ${RAF_CXX_SOURCE_FILES}
  ${RAF_CUDA_SOURCE_FILES}
  ${RAF_CUDNN_SOURCE_FILES}
  ${RAF_CUBLAS_SOURCE_FILES}
  ${RAF_CUTLASS_SOURCE_FILES}
  ${RAF_MPI_SOURCE_FILES}
  ${RAF_NCCL_SOURCE_FILES}
)

add_library(raf_objs OBJECT ${RAF_SOURCE_FILES})
target_include_directories(raf_objs PRIVATE ${RAF_INCLUDE_DIRS})
target_include_directories(raf_objs SYSTEM PRIVATE ${RAF_BACKEND_INCLUDE_DIRS})
target_include_directories(raf_objs SYSTEM PRIVATE "3rdparty/tvm/3rdparty/compiler-rt")
target_compile_options(raf_objs PRIVATE ${RAF_CXX_FLAGS})
target_compile_features(raf_objs PRIVATE cxx_std_17)
target_compile_definitions(raf_objs PRIVATE DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

if (${RAF_USE_CUDA} STREQUAL "OFF")
  add_library(raf $<TARGET_OBJECTS:raf_objs>)
else()
  add_library(raf_cuda_objs OBJECT ${RAF_CUDA_KERNEL_FILES})
  target_include_directories(raf_cuda_objs PRIVATE ${RAF_INCLUDE_DIRS})
  target_include_directories(raf_cuda_objs SYSTEM PRIVATE ${RAF_BACKEND_INCLUDE_DIRS})
  target_include_directories(raf_cuda_objs SYSTEM PRIVATE "3rdparty/tvm/3rdparty/compiler-rt")
  target_compile_options(raf_cuda_objs PRIVATE ${RAF_CUDA_FLAGS})
  target_compile_features(raf_cuda_objs PRIVATE cxx_std_17)
  target_compile_definitions(raf_cuda_objs PRIVATE DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

  add_library(raf $<TARGET_OBJECTS:raf_objs> $<TARGET_OBJECTS:raf_cuda_objs>)
endif()

target_compile_options(raf PRIVATE ${RAF_CXX_FLAGS})
target_link_libraries(raf PRIVATE ${RAF_LINK_LIBS} ${RAF_BACKEND_LINK_LIBS})
target_compile_features(raf PRIVATE cxx_std_17)

set_target_properties(raf PROPERTIES
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
  # CUDA_PTX_COMPILATION
  # CUDA_RESOLVE_DEVICE_SYMBOLS
)
raf_target_add_sanitizer(raf)

install(
      DIRECTORY "include/."
      DESTINATION "include"
      FILES_MATCHING
      PATTERN "*.h")
install(TARGETS raf
      LIBRARY DESTINATION lib)

################# Tests #################
if (${RAF_USE_GTEST} STREQUAL "ON")
  add_subdirectory(${PROJECT_SOURCE_DIR}/tests/cpp)
endif()
