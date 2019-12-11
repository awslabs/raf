cmake_minimum_required(VERSION 3.8)
message(STATUS "Building MNM with CMake version: ${CMAKE_VERSION}")
project(mnm C CXX)

################# User-defined configurations from config.cmake #################
include(${PROJECT_SOURCE_DIR}/cmake/utils/MNMCmakeUtils.cmake)
mnm_option(MNM_USE_LLVM "MNM depends on LLVM to do code generation. Option: [ON/OFF/Path-to-llvm-config-executable]" ON)
mnm_option(MNM_USE_CUDA "Build MNM with CUDA. Option: [ON/OFF]" OFF)
mnm_option(MNM_USE_CUDNN "Build MNM with cuDNN. Option: [ON/OFF/Path-to-cuDNN]" OFF)
mnm_option(MNM_USE_GTEST "Build cpptests for MNM. Option: [ON/OFF]" OFF)
mnm_option(MNM_USE_SANITIZER "Build MNM with sanitizer. Option: [OFF/ASAN/MSAN/TSAN/UBSAN]" OFF)
mnm_find_config()

if (NOT ${MNM_USE_CUDA} STREQUAL "OFF")
  enable_language(CUDA)
endif()

################# Modules and Third-Party Targets #################
include(${PROJECT_SOURCE_DIR}/cmake/modules/Threads.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/Git.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/CUDA.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/CUBLAS.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/CUDNN.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/Sanitizer.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/TVM.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/GTest.cmake)

################# Primary Target: MNM #################
set(MNM_INCLUDE_DIRS
  ${CMAKE_CURRENT_LIST_DIR}/include/
  ${CMAKE_CURRENT_LIST_DIR}/3rdparty/tvm/include/
  ${CMAKE_CURRENT_LIST_DIR}/3rdparty/tvm/3rdparty/dlpack/include/
  ${CMAKE_CURRENT_LIST_DIR}/3rdparty/tvm/3rdparty/dmlc-core/include/
)

set(MNM_BACKEND_INCLUDE_DIRS
  ${MNM_CUDA_INCLUDE}
  ${MNM_CUDNN_INCLUDE}
)

set(MNM_LINK_LIBS
  tvm
  Threads::Threads
  ${CMAKE_DL_LIBS}
)

set(MNM_BACKEND_LINK_LIBS
  ${MNM_CUDNN_LIBRARY}
  ${MNM_CUBLAS_LIBRARY}
)

set_property(
  SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/impl/build_info.cc
  APPEND
  PROPERTY COMPILE_DEFINITIONS
  MNM_GIT_VERSION="${MNM_GIT_VERSION}"
  MNM_USE_CUDA="${MNM_USE_CUDA}"
  MNM_USE_LLVM="${MNM_USE_LLVM}"
  MNM_USE_CUDNN="${MNM_USE_CUDNN}"
)

file(GLOB_RECURSE MNM_CXX_SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/src/common/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/device_api/cpu/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/memory_pool/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/schema/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/generic/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/regs/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/op/backend/tvmjit/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/pass/*.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/impl/*.cc
)

if (${MNM_USE_CUDA} STREQUAL "OFF")
  set(MNM_CUDA_SOURCE_FILES "")
else ()
  file(GLOB_RECURSE MNM_CUDA_SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/device_api/cuda/*.cc
    ${CMAKE_CURRENT_LIST_DIR}/src/device_api/cuda/*.cu
  )
endif()

if (${MNM_USE_CUDNN} STREQUAL "OFF")
  set(MNM_CUDNN_SOURCE_FILES "")
else()
  file(GLOB_RECURSE MNM_CUDNN_SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/op/backend/cudnn/*.cc
  )
endif()

if (${MNM_USE_CUBLAS} STREQUAL "OFF")
  set(MNM_CUBLAS_SOURCE_FILES "")
else()
  file(GLOB_RECURSE MNM_CUBLAS_SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/op/backend/cublas/*.cc
  )
endif()

set(MNM_SOURCE_FILES
  ${MNM_CXX_SOURCE_FILES}
  ${MNM_CUDA_SOURCE_FILES}
  ${MNM_CUDNN_SOURCE_FILES}
  ${MNM_CUBLAS_SOURCE_FILES}
)

add_library(mnm ${MNM_SOURCE_FILES})
target_include_directories(mnm PRIVATE ${MNM_INCLUDE_DIRS})
target_include_directories(mnm SYSTEM PRIVATE ${MNM_BACKEND_INCLUDE_DIRS})
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
