#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

if [[ `python3 -c "import torch; print(torch.cuda.is_available())"` == "False" ]]; then
    echo "PyTorch is not built with CUDA. Skipping apex installation"
    exit 0
fi
git clone https://github.com/szhengac/apex --branch lans
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../..
