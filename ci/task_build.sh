#!/bin/bash
set -e
set -u
set -o pipefail
source ./ci/env.sh

BUILD_DIR=$1
MAKE_FLAGS=$2

# build
mkdir -p $BUILD_DIR
cd $BUILD_DIR && cmake .. && make $MAKE_FLAGS && make mnm-cpptest $MAKE_FLAGS && cd ..
