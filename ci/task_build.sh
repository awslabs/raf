#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail
source ./ci/env.sh

BUILD_DIR=$1
MAKE_FLAGS=$2

git checkout --recurse-submodules .
./scripts/src_codegen/run_all.sh

# build
mkdir -p $BUILD_DIR
pushd .
cd $BUILD_DIR && cmake .. && make $MAKE_FLAGS && make raf-cpptest $MAKE_FLAGS
popd

# test build wheels
export TVM_LIBRARY_PATH=${PWD}/build/lib
pushd .
cd 3rdparty/tvm/python
python3 setup.py bdist_wheel -d ../build/pip/public/tvm_latest
python3 -m pip install ../build/pip/public/tvm_latest/*.whl --upgrade --force-reinstall --no-deps
popd

pushd .
cd python
TVM_FFI=auto python3 setup.py bdist_wheel -d ../build/pip/public/raf
python3 -m pip install ../build/pip/public/raf/*.whl --upgrade --force-reinstall --no-deps
popd
