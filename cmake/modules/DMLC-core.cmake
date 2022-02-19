# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
