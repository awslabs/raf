cmake_minimum_required(VERSION 3.0)
project(mnm C CXX)

file(GLOB_RECURSE MNM_SRC_FILES "${CMAKE_SOURCE_DIR}/src/mnm/*.cc")
set(MNM_HEADER_FILES ${CMAKE_SOURCE_DIR}/include)

add_library(mnm SHARED ${MNM_SRC_FILES})
target_compile_features(mnm PRIVATE cxx_std_14)
set_target_properties(mnm PROPERTIES CXX_STANDARD 14)
target_include_directories(mnm PRIVATE ${MNM_HEADER_FILES})

add_subdirectory(apps)
