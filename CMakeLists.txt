cmake_minimum_required(VERSION 3.8)
project(mnm C CXX)

################# Basic requirement #################
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
add_definitions(-DDMLC_USE_FOPEN64=0)

################# Utilities #################
include(cmake/Util.cmake)

################# User-defined configurations #################
# TODO(@junrushao1994): improve it
if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  message(STATUS "Found ${CMAKE_CURRENT_BINARY_DIR}/config.cmake")
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  message(STATUS "Found ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake")
  else()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
      include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
      message(STATUS "Found ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake")
    else()
      message(WARNING "config.cmake is not found")
    endif()
  endif()
endif()

################# Thrid-party libraries #################
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/tvm)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/tvm/3rdparty/dlpack/)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/tvm/3rdparty/dmlc-core/)
set(MNM_3RD_PARTY_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/include
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/3rdparty/dlpack/include/
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/3rdparty/dmlc-core/include/
)

################# Target: MNM #################
file(GLOB_RECURSE MNM_SRC_FILES ${CMAKE_SOURCE_DIR}/src/mnm/*.cc)
set(MNM_HEADER_FILES ${CMAKE_SOURCE_DIR}/include)

add_library(mnm SHARED ${MNM_SRC_FILES})
target_compile_features(mnm PRIVATE cxx_std_14)
set_target_properties(mnm PROPERTIES CXX_STANDARD 14)
target_include_directories(mnm PUBLIC ${MNM_HEADER_FILES})
target_include_directories(mnm PUBLIC ${MNM_3RD_PARTY_INCLUDE_DIRS})
target_link_libraries(mnm dmlc)
target_link_libraries(mnm tvm)

################# Apps #################
add_subdirectory(apps)
