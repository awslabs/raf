# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Set options for 3rdparty/dmlc-core
message("########## Configuring dmlc-core ##########")
add_definitions(-DDMLC_USE_FOPEN64=0)
set(USE_CXX14_IF_AVAILABLE ON CACHE BOOL "Build with C++14 if the compiler supports it" FORCE)
set(USE_OPENMP OFF CACHE BOOL "Build with OpenMP" FORCE)
# Introduce targets from dmlc-core
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/tvm/3rdparty/dmlc-core/)
# Unset cache variables to make sure it does not affect other projects
unset(USE_HDFS CACHE)
unset(USE_AZURE CACHE)
unset(USE_S3 CACHE)
unset(USE_OPENMP CACHE)
unset(USE_CXX14_IF_AVAILABLE CACHE)
unset(GOOGLE_TEST CACHE)
unset(INSTALL_DOCUMENTATION CACHE)
remove_definitions(-DDMLC_USE_FOPEN64=0)
message("########## Done configuring dmlc-core ##########")
