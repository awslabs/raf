#!/bin/bash

set -e
set -u
set -o pipefail

export PYTHONPATH=python/
export PYTHONPATH=3rdparty/tvm/topi/python:$PYTHONPATH
export PYTHONPATH=3rdparty/tvm/python:$PYTHONPATH
export TVM_LIBRARY_PATH=build/3rdparty/tvm/

python3 -m pytest -v tests/python/

