#!/bin/bash
set -e
set -u
set -o pipefail
source ./ci/env.sh

cd 3rdparty/tvm/ && make cython3 && cd ../../
python3 -m pytest tests/python/