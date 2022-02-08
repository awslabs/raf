#!/bin/bash
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

set -e
set -u
set -o pipefail

# install libraries for building c++ core on ubuntu
apt-get update && apt-get install -y --no-install-recommends \
        git make cmake wget unzip libtinfo-dev libz-dev \
        libcurl4-openssl-dev libopenblas-dev g++ sudo \
        doxygen graphviz libprotobuf-dev protobuf-compiler curl \
        clang-format-10 ssh openmpi-bin openmpi-doc libopenmpi-dev

# upgrade git to 2.18+
apt install -y -q software-properties-common
add-apt-repository -y ppa:git-core/ppa
apt update
apt install git -y -q

# install ccache 4.0 to cache CUDA kernels
pushd .
cd /tmp
wget https://github.com/ccache/ccache/releases/download/v4.0/ccache-4.0.tar.gz
tar -xzf ccache-4.0.tar.gz
cd ccache-4.0
mkdir build; cd build
cmake -DZSTD_FROM_INTERNET=ON -DCMAKE_BUILD_TYPE=Release ..
make -j
make install
popd

