#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
