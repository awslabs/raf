cmake_minimum_required(VERSION 3.8)

################# Utilities #################
include(cmake/Util.cmake)
include(cmake/ClangFormat.cmake)
clangformat()

################# User-defined configurations #################
if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  message(STATUS "Found ${CMAKE_CURRENT_BINARY_DIR}/config.cmake")
else()
  message(WARNING "config.cmake is not found")
endif()

mnm_option(MNM_USE_CUDA "The path to CUDA prefix" OFF)
mnm_option(MNM_USE_CUDNN "The path to CUDNN prefix" OFF)
mnm_option(MNM_USE_GTEST "The path to GTEST prefix" OFF)

if (NOT MNM_USE_CUDA)
  project(mnm C CXX)
else()
  project(mnm C CXX CUDA)
  set(USE_CUDA ${MNM_USE_CUDA})
endif()

################# Basic requirement #################
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
add_definitions(-DDMLC_USE_FOPEN64=0)

################# Thrid-party libraries #################
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/tvm)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/tvm/3rdparty/dmlc-core/)
set(MNM_3RD_PARTY_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/include
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/3rdparty/dlpack/include/
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/3rdparty/dmlc-core/include/
)

################# Target: MNM #################
file(GLOB_RECURSE MNM_SRC_FILES
  ${CMAKE_SOURCE_DIR}/src/common/*.cc
  ${CMAKE_SOURCE_DIR}/src/device_api/cpu/*.cc
  ${CMAKE_SOURCE_DIR}/src/memory_pool/*.cc
  ${CMAKE_SOURCE_DIR}/src/op/schema/*.cc
  ${CMAKE_SOURCE_DIR}/src/op/generic/*.cc
  ${CMAKE_SOURCE_DIR}/src/op/regs/*.cc
  ${CMAKE_SOURCE_DIR}/src/impl/*.cc
)

set(MNM_HEADER_FILES ${CMAKE_SOURCE_DIR}/include)
set(MNM_LINK_LIBRARIES "")

list(APPEND MNM_LINK_LIBRARIES "dmlc")
list(APPEND MNM_LINK_LIBRARIES "tvm")

################# Backends #################
include(${CMAKE_SOURCE_DIR}/cmake/backend/CUDA.cmake)

# Finalize the compilation target here
add_library(mnm SHARED ${MNM_SRC_FILES})
set_target_properties(mnm PROPERTIES CXX_STANDARD 14)
target_compile_features(mnm PRIVATE cxx_std_14)
target_include_directories(mnm PRIVATE ${MNM_HEADER_FILES})
target_include_directories(mnm PRIVATE ${MNM_3RD_PARTY_INCLUDE_DIRS})
target_link_libraries(mnm ${MNM_LINK_LIBRARIES})

################# Apps #################
add_subdirectory(apps)
add_subdirectory(tests/cpp)
add_subdirectory(tests/cuda)
add_subdirectory(tests/cudnn)
