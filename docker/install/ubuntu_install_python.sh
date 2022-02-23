#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


set -e
set -u
set -o pipefail

apt-get update
apt-get install -y python-dev python3-dev
apt-get install -y python-pip python3-pip
pip3 install pip --upgrade
pip3 install cmake
pip3 install scikit-build==0.11.1
pip3 install pylint==2.4.3 cpplint black==20.8b1
pip3 install six numpy pytest cython decorator scipy tornado typed_ast pytest mypy orderedset \
             antlr4-python3-runtime attrs requests Pillow packaging psutil dataclasses pycparser \
             pydot
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install mxnet==1.6.0
pip3 install gluoncv==0.10.1

if [[ "$1" == "gpu" ]]; then
    mkdir -p build && cd build
    git clone https://github.com/szhengac/apex --branch lans
    cd apex
    pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    cd ../..
fi
