#!/bin/bash
set -e
set -u
set -o pipefail
source ./ci/env.sh

cd 3rdparty/tvm/ && make cython3 && cd ../../

# distributed training test
mpirun -np 2 python3 tests/python/distributed/test_collective_communication.py

# distributed type function test
mpirun -np 2 python3 tests/python/op/ty/test_type_comm.py
