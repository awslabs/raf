#!/bin/bash
set -e
set -u
set -o pipefail

export MNM_HOME=$(pwd)
export PYTHONPATH=$MNM_HOME/python/
export PYTHONPATH=$MNM_HOME/3rdparty/tvm/python:$PYTHONPATH
export TVM_LIBRARY_PATH=$MNM_HOME/build/lib/
export TVM_FFI=cython
export CUDA_HOME=/usr/local/cuda-10.2/targets/x86_64-linux
export CUDNN_HOME=/usr
