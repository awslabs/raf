cmake_minimum_required(VERSION 3.8)
project(mnm C CXX)

################# Utilities #################
include(cmake/Util.cmake)

################# User-defined configurations #################
if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
  else()
    message(WARNING "config.cmake is not found")
  endif()
endif()

################# Thrid-party libraries #################

SET_OPTION(USE_CXX14_IF_AVAILABLE ON)   # for dmlc-core

add_subdirectory("3rdparty/dmlc-core")
# add_subdirectory("3rdparty/tvm") # let's enable it once we use tvm
set(MNM_3RD_PARTY_INCLUDE_DIRS
  "3rdparty/dmlc-core/include"
  "3rdparty/tvm"
)

################# Target: MNM #################

file(GLOB_RECURSE MNM_SRC_FILES "${CMAKE_SOURCE_DIR}/src/mnm/*.cc")
set(MNM_HEADER_FILES ${CMAKE_SOURCE_DIR}/include)

add_library(mnm SHARED ${MNM_SRC_FILES})
target_compile_features(mnm PRIVATE cxx_std_14)
set_target_properties(mnm PROPERTIES CXX_STANDARD 14)
target_include_directories(mnm PUBLIC ${MNM_HEADER_FILES})
target_include_directories(mnm PUBLIC ${MNM_3RD_PARTY_INCLUDE_DIRS})

################# Apps #################
add_subdirectory(apps)
