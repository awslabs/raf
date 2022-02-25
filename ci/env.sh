#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

export RAF_HOME=$(pwd)
export PYTHONPATH=$RAF_HOME/python/
export PYTHONPATH=$RAF_HOME/3rdparty/tvm/python:$PYTHONPATH
export TVM_LIBRARY_PATH=$RAF_HOME/build/lib/
export TVM_FFI=cython

if [[ -z ${CUDA_HOME+x} ]]; then
  export CUDA_HOME=/usr/local/cuda/targets/x86_64-linux  
fi
echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC Version: `nvcc --version`"
export CUDNN_HOME=$CUDA_HOME

