#!/bin/bash
set -e
set -u
set -o pipefail
source ./ci/env.sh

cd 3rdparty/tvm/ && make cython3 && cd ../../

# distributed training test
mpirun -np 2 python3 tests/python/distributed/test_collective_communication.py

# pytest
python3 -m pytest --assert=plain tests/python/
