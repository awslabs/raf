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

apt-get update
apt-get install -y python-dev python3-dev
apt-get install -y python-pip python3-pip
pip3 install pip --upgrade
pip3 install cmake
pip3 install scikit-build==0.11.1
pip3 install pylint==2.4.3 cpplint
pip3 install six numpy pytest cython decorator scipy tornado typed_ast pytest mypy orderedset \
             antlr4-python3-runtime attrs requests Pillow packaging psutil dataclasses pycparser \
             pydot
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip3 install mxnet==1.6.0
pip3 install gluoncv==0.10.1
