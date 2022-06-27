# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/bin/bash

# Create the build directory
git clone https://github.com/awslabs/raf --recursive
cd raf
export RAF_HOME=$(pwd)
mkdir $RAF_HOME/build

# Run the codegen for auto-generated source code
bash ./scripts/src_codegen/run_all.sh

# TODO: Make this part of creating CMake configurable. It should take arguments from the command line to determine the configuration.  
# Configuration file for CMake
pushd .
cd $RAF_HOME/build
cp ../cmake/config.cmake .

# Edit the configuration file
echo set\(RAF_USE_LLVM \"llvm-config-8\"\) >> config.cmake
echo set\(HIDE_PRIVATE_SYMBOLS ON\) >> config.cmake
echo set\(RAF_USE_GTEST ON\) >> config.cmake
echo set\(RAF_USE_SANITIZER OFF\) >> config.cmake
echo set\(RAF_USE_MPI OFF\) >> config.cmake
echo set\(RAF_USE_NCCL OFF\) >> config.cmake

# Set environment variables for the version and platform to build RAF
# Check if environment variables are set, else set default
# RAF_BUILD_VERSION. Option: [stable/nightly/dev]
if [ -z "$RAF_BUILD_VERSION" ]
then
    export RAF_BUILD_VERSION=dev
fi 
# RAF_BUILD_PLATFORM. Option: [cpu/cu113]
if [ -z "$RAF_BUILD_PLATFORM" ]
then
    export RAF_BUILD_PLATFORM=cu113
fi
echo "Setting RAF_BUILD_VERSION="$RAF_BUILD_VERSION
echo "Setting RAF_BUILD_PLATFORM="$RAF_BUILD_PLATFORM
if [ $RAF_BUILD_PLATFORM = "cu113" ]
then
    echo set\(RAF_USE_CUDA ON\) >> config.cmake
    echo set\(RAF_CUDA_ARCH 75\) >> config.cmake
    echo set\(RAF_USE_CUBLAS ON\) >> config.cmake
    echo set\(RAF_USE_CUDNN ON\) >> config.cmake
    echo set\(RAF_USE_CUTLASS ON\) >> config.cmake
fi

# Configure the project
cmake ..
# Finally let's trigger build
make -j$(nproc)

# Install python dependencies
pip install six numpy pytest cython decorator scipy tornado typed_ast mypy orderedset pydot \
             antlr4-python3-runtime attrs requests Pillow packaging psutil dataclasses pycparser

# Run RAF
export PYTHONPATH=$RAF_HOME/python/:$RAF_HOME/3rdparty/tvm/python
export TVM_LIBRARY_PATH=$RAF_HOME/build/lib
python3 -c "import raf"

# Generate source distribution
popd
pushd .
cd python
python3 ./setup.py bdist_wheel -d ../build/pipe/public/raf
