# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#! /usr/bin/env bash
# The script to build Python wheel for manylinux.
# Notes:
# 1) This script is supposed to be used in the manylinux container.
# 2) There are some environment variables that should be configured in advance.
#
# RAF_BUILD_VERSION. Option: [stable/nightly/dev], default: dev.
# RAF_BUILD_PLATFORM. Option: [cpu/cu113], default: cu113.
# RAF_CUDA_ARCH. Default: 70;75.
set -xe

source /multibuild/manylinux_utils.sh

function build_wheel() {
    python_version=$1

    CPYTHON_PATH="$(cpython_path ${python_version})"
    PYTHON_BIN="${CPYTHON_PATH}/bin/python"
    PIP_BIN="${CPYTHON_PATH}/bin/pip"

    # Build TVM wheel
    pushd .
    cd "${RAF_HOME}/3rdparty/tvm/python"
    TVM_LIBRARY_PATH=${RAF_HOME}/build/lib PATH="${CPYTHON_PATH}/bin:$PATH" \
        ${PYTHON_BIN} setup.py bdist_wheel
    popd

    # Build RAF wheel
    pushd .
    cd "${RAF_HOME}/python"
    PATH="${CPYTHON_PATH}/bin:$PATH" ${PYTHON_BIN} setup.py bdist_wheel
    popd
}

function audit_wheel() {
    python_version=$1

    # Remove the . in version string, e.g. "3.8" turns into "38"
    python_version_str="$(echo "${python_version}" | sed -r 's/\.//g')"

    # Audit the TVM wheel
    pushd .
    cd "${RAF_HOME}/3rdparty/tvm/python"
    mkdir -p repaired_wheels
    auditwheel repair ${AUDITWHEEL_OPTS} dist/*cp${python_version_str}*.whl
    popd 

    # Audit the RAF wheel
    pushd .
    cd "${RAF_HOME}/python"
    mkdir -p repaired_wheels
    auditwheel repair ${AUDITWHEEL_OPTS} dist/*cp${python_version_str}*.whl
    popd
}

PYTHON_VERSIONS=("3.7" "3.8")

AUDITWHEEL_OPTS="--plat ${AUDITWHEEL_PLAT} -w repaired_wheels/"
SKIP_LIBS="libtvm"
if [[ ${CUDA} != "none" ]]; then
    SKIP_LIBS="${SKIP_LIBS},libcuda"
fi
AUDITWHEEL_OPTS="--skip-libs ${SKIP_LIBS} ${AUDITWHEEL_OPTS}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAF_HOME=$SCRIPT_DIR/../../

pushd .
cd $RAF_HOME

mkdir -p ./build

# Run the codegen for auto-generated source code
bash ./scripts/src_codegen/run_all.sh

# Configuration file for CMake
cd ./build
touch config.cmake

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
if [ -z "$RAF_CUDA_ARCH" ]
then
    export RAF_CUDA_ARCH="70;75"
fi
echo "Setting RAF_BUILD_VERSION=$RAF_BUILD_VERSION"
echo "Setting RAF_BUILD_PLATFORM=$RAF_BUILD_PLATFORM"
echo "Setting RAF_CUDA_ARCH=$RAF_CUDA_ARCH"

# Rewrite python/setup.py for RAF package name.
RAF_PACKAGE_NAME="raf-dev-$RAF_BUILD_PLATFORM"
if [ "$RAF_BUILD_VERSION" = "nightly" ]
then
    RAF_PACKAGE_NAME="raf-nightly-$RAF_BUILD_PLATFORM"
elif [ "$RAF_BUILD_VERSION" = "stable" ]
then
    RAF_PACKAGE_NAME="raf-$RAF_BUILD_PLATFORM"
fi
sed -Ei "s/name=(.+)/name=\"${RAF_PACKAGE_NAME}\",/g" ../python/setup.py

# Rewrite 3rdparty/tvm/python/setup.py for TVM package name.
TVM_PACKAGE_NAME="tvm-$RAF_BUILD_PLATFORM"
sed -Ei "s/name=(.+)/name=\"${TVM_PACKAGE_NAME}\",/g" ../3rdparty/tvm/python/setup.py

# Generate the configuration file.
# Note that this configuration is for wheel build, so we disable ccache and backtrace.
echo set\(BUILD_SHARED_LIBS ON\) >> config.cmake
echo set\(CMAKE_EXPORT_COMPILE_COMMANDS ON\) >> config.cmake
echo set\(CMAKE_BUILD_TYPE Release\) >> config.cmake
echo set\(RAF_USE_CUDA OFF\) >> config.cmake
echo set\(RAF_USE_SANITIZER OFF\) >> config.cmake
echo set\(RAF_USE_LLVM \"llvm-config\"\) >> config.cmake
echo set\(HIDE_PRIVATE_SYMBOLS ON\) >> config.cmake
echo set\(USE_LIBBACKTRACE OFF\) >> config.cmake

if [[ $RAF_BUILD_PLATFORM == cu11* ]];
then
    echo set\(RAF_USE_CUDA ON\) >> config.cmake
    echo set\(RAF_CUDA_ARCH $RAF_CUDA_ARCH\) >> config.cmake
    echo set\(RAF_USE_CUBLAS ON\) >> config.cmake
    echo set\(RAF_USE_CUDNN ON\) >> config.cmake
    echo set\(RAF_USE_CUTLASS ON\) >> config.cmake
    # TODO: Support MPI and NCCL in build wheel docker images.
    echo set\(RAF_USE_MPI OFF\) >> config.cmake
    echo set\(RAF_USE_NCCL OFF\) >> config.cmake
else
    echo set\(RAF_USE_CUDA OFF\) >> config.cmake
    echo set\(RAF_USE_CUBLAS OFF\) >> config.cmake
    echo set\(RAF_USE_CUDNN OFF\) >> config.cmake
    echo set\(RAF_USE_CUTLASS OFF\) >> config.cmake
    echo set\(RAF_USE_MPI OFF\) >> config.cmake
    echo set\(RAF_USE_NCCL OFF\) >> config.cmake
fi

# Configure the project
cmake ..
# Finally let's trigger build. Use nproc-1 to avoid out-of-memory during build.
make -j$(($(nproc) - 1))

# Generate source distribution
for python_version in ${PYTHON_VERSIONS[*]}
do
    cpython_dir="$(cpython_path ${python_version} 2> /dev/null)"
    if [ -d "${cpython_dir}" ]; then
      echo "Generating package for Python ${python_version}."
      build_wheel ${python_version}

      echo "Running auditwheel on package for Python ${python_version}."
      audit_wheel ${python_version}
    else
      echo "Python ${python_version} not found. Skipping.";
    fi
done

popd
