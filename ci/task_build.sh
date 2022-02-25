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
cd $BUILD_DIR && cmake .. && make $MAKE_FLAGS && make raf-cpptest $MAKE_FLAGS && cd ..
