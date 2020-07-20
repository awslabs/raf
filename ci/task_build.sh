#!/bin/bash
set -e
set -u
set -o pipefail
source ./ci/env.sh

BUILD_DIR=$1
MAKE_FLAGS=$2

pip3 install cmake
pip3 install dataclasses
pip3 install pycparser

# check src_gen
git checkout --recurse-submodules .
./scripts/src_codegen/run_all.sh
if [[ ! -z `git status --porcelain --ignore-submodules *.cc *.py *.h` ]]; then
   echo "src gen created git diff, please update the source gen file"
   git status --porcelain
   exit 1
fi

# build
mkdir -p $BUILD_DIR
cd $BUILD_DIR && $HOME/.local/bin/cmake .. && make $MAKE_FLAGS && make mnm-cpptest $MAKE_FLAGS && cd ..
