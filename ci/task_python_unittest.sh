#!/bin/bash
set -e
set -u
set -o pipefail
source ./ci/env.sh

pip3 install pytest-xdist
cd 3rdparty/tvm/ && make cython3 && cd ../../
rm -rf .pkl_memoize_py3 .pytest_cache .tvm_test_data
python3 -m pytest tests/python/ -n 8
