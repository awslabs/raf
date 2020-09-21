#!/bin/bash
set -e
set -u
set -o pipefail
source ./ci/env.sh

cd 3rdparty/tvm/ && make cython3 && cd ../../

# distributed training test
mpirun -np 2 python3 tests/python/distributed/test_collective_communication.py

# pytest
sudo pip3 uninstall -y torch
sudo pip3 install https://download.pytorch.org/whl/cu102/torch-1.6.0-cp36-cp36m-linux_x86_64.whl
python3 -m pytest --assert=plain tests/python/
