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

BUILD_DIR=$1
MAKE_FLAGS=$2

git checkout --recurse-submodules .
./scripts/src_codegen/run_all.sh

# build
mkdir -p $BUILD_DIR
cd $BUILD_DIR && cmake .. && make $MAKE_FLAGS && make mnm-cpptest $MAKE_FLAGS && cd ..
