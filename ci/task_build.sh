#!/bin/bash
set -e
set -u
set -o pipefail

BUILD_DIR=$1
MAKE_FLAGS=$2

mkdir -p $BUILD_DIR
cd $BUILD_DIR && cmake .. && make $MAKE_FLAGS && cd ..
