#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
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

# update symbolic links
for t in `ls -d -- /usr/bin/{gcc,gcc-[0-9+],g++,g++-[0-9+],clang,clang-[0-9+],clang++-[0-9+]}`; do
    ln -fvs /usr/local/bin/ccache /usr/local/bin/$(basename $t);
done

