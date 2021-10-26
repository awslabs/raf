cmake_minimum_required(VERSION 3.12)
message(STATUS "Building MNM with CMake version: ${CMAKE_VERSION}")
project(mnm C CXX)

################# User-defined configurations from config.cmake #################
include(${PROJECT_SOURCE_DIR}/cmake/utils/MNMCmakeUtils.cmake)
mnm_option(MNM_USE_LLVM "MNM depends on LLVM to do code generation. Option: [ON/OFF/Path-to-llvm-config-executable]" ON)
mnm_option(MNM_USE_CUDA "Build MNM with CUDA. Option: [ON/OFF]" OFF)
mnm_option(MNM_USE_CUDNN "Build MNM with cuDNN. Option: [ON/OFF/Path-to-cuDNN]" OFF)
mnm_option(MNM_USE_CUTLASS "Build MNM with CUTLASS. Option: [ON/OFF/NVCC-Arch]" OFF)
mnm_option(MNM_CUDA_ARCH "Specify the CUDA architecture" 70)
mnm_option(MNM_USE_MPI "Build MNM with MPI. Option: [ON/OFF]" OFF)
mnm_option(MNM_USE_NCCL "Build MNM with NCCL. Option: [ON/OFF]" OFF)
mnm_option(MNM_USE_CUBLAS "Build MNM with cuBLAS. Option: [ON/OFF]" OFF)
mnm_option(MNM_USE_GTEST "Build cpptests for MNM. Option: [ON/OFF]" OFF)
mnm_option(MNM_USE_SANITIZER "Build MNM with sanitizer. Option: [OFF/ASAN/MSAN/TSAN/UBSAN]" OFF)
mnm_find_config()

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

################# Primary Target: MNM #################
set(MNM_INCLUDE_DIRS
  ${CMAKE_CURRENT_LIST_DIR}/include/
  ${CMAKE_CURRENT_LIST_DIR}/3rdparty/tvm/include/
  ${CMAKE_CURRENT_LIST_DIR}/3rdparty/tvm/src
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
set(MNM_CXX_FLAGS -fPIC)
set(MNM_CUDA_FLAGS -Xcompiler=-Wall -Xcompiler=-fPIC)

set(MNM_BACKEND_INCLUDE_DIRS
  ${MNM_CUDA_INCLUDE}
  ${MNM_CUDNN_INCLUDE}
  ${MNM_NCCL_INCLUDE}
  ${MNM_MPI_INCLUDE}
)

set(MNM_LINK_LIBS
  tvm
  Threads::Threads
  ${MNM_CUTLASS_LIBRARY}
  ${CMAKE_DL_LIBS}
)

set(MNM_BACKEND_LINK_LIBS
  ${MNM_CUDNN_LIBRARY}
  ${MNM_CUBLAS_LIBRARY}
  ${MNM_NCCL_LIBRARY}
  ${MNM_MPI_LIBRARY}
)

set_property(
  SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/impl/build_info.cc
  APPEND
  PROPERTY COMPILE_DEFINITIONS
  MNM_GIT_VERSION="${MNM_GIT_VERSION}"
  MNM_CUDA_VERSION="${CUDA_VERSION_STRING}"
  MNM_USE_LLVM="${MNM_USE_LLVM}"
  MNM_USE_CUBLAS="${MNM_USE_CUBLAS}"
  MNM_USE_CUDNN="${MNM_USE_CUDNN}"
  MNM_CUDNN_VERSION="${MNM_CUDNN_VERSION}"
  MNM_CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  MNM_USE_MPI="${MNM_USE_MPI}"
  MNM_USE_CUTLASS="${MNM_USE_CUTLASS}"
)

file(GLOB_RECURSE MNM_CXX_SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/src/analysis/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/common/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/device_api/cpu/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/memory_pool/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/schema/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/declare/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/regs/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/grad/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/tvm/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/from_relay/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/ty/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/pass/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/impl/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/profiler/base/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/distributed/common/*.cc
)

if (${MNM_USE_CUDA} STREQUAL "OFF")
  set(MNM_CUDA_SOURCE_FILES "")
  set(MNM_CUDA_KERNEL_FILES "")
  set(MNM_CUDA_FLAGS "")
else ()
  message(STATUS "Build with CUDA enabled")
  message(STATUS "Build for CUDA architecture: ${MNM_CUDA_ARCH}")
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES ${MNM_CUDA_ARCH})
  set(MNM_CXX_FLAGS ${MNM_CXX_FLAGS} -DMNM_USE_CUDA)
  set(MNM_CUDA_FLAGS ${MNM_CUDA_FLAGS} -DMNM_USE_CUDA
      -gencode=arch=compute_${MNM_CUDA_ARCH},code=sm_${MNM_CUDA_ARCH})

  file(GLOB_RECURSE MNM_CUDA_SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/device_api/cuda/*.cc
    ${CMAKE_CURRENT_LIST_DIR}/src/profiler/cuda/*.cc
    ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/cuda/*.cc
  )

  file(GLOB_RECURSE MNM_CUDA_KERNEL_FILES
	${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/cuda/kernels/*.cu
  )
endif()

if (${MNM_USE_CUDNN} STREQUAL "OFF")
  set(MNM_CUDNN_SOURCE_FILES "")
else()
  file(GLOB_RECURSE MNM_CUDNN_SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/cudnn/*.cc
  )
endif()

if (${MNM_USE_CUBLAS} STREQUAL "OFF")
  set(MNM_CUBLAS_SOURCE_FILES "")
else()
  file(GLOB_RECURSE MNM_CUBLAS_SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/cublas/*.cc
  )
endif()

if (${MNM_USE_CUTLASS} STREQUAL "OFF")
  set(MNM_CUTLASS_SOURCE_FILES "")
else()
  file(GLOB_RECURSE MNM_CUTLASS_SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/cutlass/*.cc
  )
endif()

if (${MNM_USE_NCCL} STREQUAL "OFF")
  set(MNM_DISTRIBUTED_SOURCE_FILES "")
else ()
  set(MNM_CXX_FLAGS ${MNM_CXX_FLAGS} -DMNM_USE_NCCL)
  file(GLOB_RECURSE MNM_DISTRIBUTED_SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/distributed/cuda/*.cc
    ${CMAKE_CURRENT_LIST_DIR}/src/op/dialect/nccl/*.cc
  )
endif()

set(MNM_SOURCE_FILES
  ${MNM_CXX_SOURCE_FILES}
  ${MNM_CUDA_SOURCE_FILES}
  ${MNM_CUDNN_SOURCE_FILES}
  ${MNM_CUBLAS_SOURCE_FILES}
  ${MNM_CUTLASS_SOURCE_FILES}
  ${MNM_DISTRIBUTED_SOURCE_FILES}
)

add_library(mnm_objs OBJECT ${MNM_SOURCE_FILES})
target_include_directories(mnm_objs PRIVATE ${MNM_INCLUDE_DIRS})
target_include_directories(mnm_objs SYSTEM PRIVATE ${MNM_BACKEND_INCLUDE_DIRS})
target_include_directories(mnm_objs SYSTEM PRIVATE "3rdparty/tvm/3rdparty/compiler-rt")
target_compile_options(mnm_objs PRIVATE ${MNM_CXX_FLAGS})
target_compile_features(mnm_objs PRIVATE cxx_std_14)
target_compile_definitions(mnm_objs PRIVATE DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

if (${MNM_USE_CUDA} STREQUAL "OFF")
  add_library(mnm $<TARGET_OBJECTS:mnm_objs>)
else()
  add_library(mnm_cuda_objs OBJECT ${MNM_CUDA_KERNEL_FILES})
  target_include_directories(mnm_cuda_objs PRIVATE ${MNM_INCLUDE_DIRS})
  target_include_directories(mnm_cuda_objs SYSTEM PRIVATE ${MNM_BACKEND_INCLUDE_DIRS})
  target_include_directories(mnm_cuda_objs SYSTEM PRIVATE "3rdparty/tvm/3rdparty/compiler-rt")
  target_compile_options(mnm_cuda_objs PRIVATE ${MNM_CUDA_FLAGS})
  target_compile_features(mnm_cuda_objs PRIVATE cxx_std_14)
  target_compile_definitions(mnm_cuda_objs PRIVATE DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

  add_library(mnm $<TARGET_OBJECTS:mnm_objs> $<TARGET_OBJECTS:mnm_cuda_objs>)
endif()

target_compile_options(mnm PRIVATE ${MNM_CXX_FLAGS})
target_link_libraries(mnm PRIVATE ${MNM_LINK_LIBS} ${MNM_BACKEND_LINK_LIBS})
target_compile_features(mnm PRIVATE cxx_std_14)

set_target_properties(mnm PROPERTIES
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
mnm_target_add_sanitizer(mnm)

################# Apps #################
add_subdirectory(${PROJECT_SOURCE_DIR}/apps/include_mnm/)

################# Tests #################
if (${MNM_USE_GTEST} STREQUAL "ON")
  add_subdirectory(${PROJECT_SOURCE_DIR}/tests/cpp)
endif()
