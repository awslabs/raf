#!/bin/bash
set -e
set -u
set -o pipefail
source ./ci/env.sh

cd 3rdparty/tvm/ && make cython3 && cd ../../

# pytest
python3 -m pytest --assert=plain tests/python/
