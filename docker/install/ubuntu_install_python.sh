#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

apt-get update
apt-get install -y build-essential
apt-get install -y python3 python3-dev python3-pip
apt-get install -y python3.7 python3.7-dev python3.7-venv
rm /usr/bin/python3
ln -s /usr/bin/python3.7 /usr/bin/python3

python3 -m pip install -U --force-reinstall pip
python3 -m pip install cmake
python3 -m pip install scikit-build==0.11.1
python3 -m pip install pylint==2.4.3 cpplint black==22.3.0
python3 -m pip install six numpy pytest cython decorator scipy tornado typed_ast \
                       pytest mypy orderedset antlr4-python3-runtime attrs requests \
                       Pillow packaging psutil dataclasses pycparser pydot filelock
python3 -m pip install astunparse numpy ninja pyyaml mkl mkl-include setuptools cffi \
                       typing_extensions future glob2 pygithub boto3
python3 -m pip install mxnet==1.6.0
python3 -m pip install gluoncv==0.10.1
python3 -m pip install datasets==1.15.1
python3 -m pip install transformers==4.17
