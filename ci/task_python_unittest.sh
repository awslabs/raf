#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail
source ./ci/env.sh

cd 3rdparty/tvm/ && make cython3 && cd ../../

# pytest
python3 -m pytest --assert=plain tests/python/
