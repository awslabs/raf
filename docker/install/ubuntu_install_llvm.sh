#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
set -u
set -o pipefail

echo deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-4.0 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-4.0 main\
     >> /etc/apt/sources.list.d/llvm.list

echo deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-7 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-7 main\
     >> /etc/apt/sources.list.d/llvm.list

echo deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-8 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-8 main\
     >> /etc/apt/sources.list.d/llvm.list

echo deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-9 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-9 main\
     >> /etc/apt/sources.list.d/llvm.list

echo deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial main\
     >> /etc/apt/sources.list.d/llvm.list

wget -q -O - http://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
apt-get update && apt-get install -y llvm-4.0 llvm-9 llvm-8 llvm-7 clang-9 clang-8 clang-7
