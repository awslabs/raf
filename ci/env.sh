#!/bin/bash
set -e
set -u
set -o pipefail

export MNM_HOME=$(pwd)
export PYTHONPATH=$MNM_HOME/python/
export PYTHONPATH=$MNM_HOME/3rdparty/tvm/topi/python:$PYTHONPATH
export PYTHONPATH=$MNM_HOME/3rdparty/tvm/python:$PYTHONPATH
export TVM_LIBRARY_PATH=$MNM_HOME/build/lib/
export TVM_FFI=cython
