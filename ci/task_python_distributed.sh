#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail
source ./ci/env.sh

cd 3rdparty/tvm/ && make cython3 && cd ../../

# distributed training test
mpirun -np 2 --allow-run-as-root python3 tests/python/distributed/test_data_parallel.py

# collective communication operator test
mpirun -np 2 --allow-run-as-root python3 tests/python/distributed/test_collective_communication.py

# distributed type function test
mpirun -np 2 --allow-run-as-root python3 tests/python/op/ty/test_type_comm.py

# test Zero with NCCL enabled
python3 tests/python/pass/test_pass_partition_gradient.py
