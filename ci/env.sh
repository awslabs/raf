#!/bin/bash
set -e
set -u
set -o pipefail

export MNM_HOME=$(pwd)
export PYTHONPATH=$MNM_HOME/python/
export PYTHONPATH=$MNM_HOME/3rdparty/tvm/python:$PYTHONPATH
export TVM_LIBRARY_PATH=$MNM_HOME/build/lib/
export TVM_FFI=cython

if [[ -z ${CUDA_HOME+x} ]]; then
  export CUDA_HOME=/usr/local/cuda/targets/x86_64-linux  
fi
echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC Version: `nvcc --version`"
export CUDNN_HOME=$CUDA_HOME

