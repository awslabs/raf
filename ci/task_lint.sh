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

echo "Check ASF header..."
python3 scripts/lint/check_asf.py HEAD~1
python3 scripts/lint/check_asf.py origin/main

echo "Check C++ formats using clang-format-10..."
# check lastest change, for squash merge into main
./scripts/lint/git-clang-format.sh HEAD~1
# chekc against origin/main for PRs.
./scripts/lint/git-clang-format.sh origin/main

echo "Check Python formats using black..."
./scripts/lint/git-black.sh HEAD~1
./scripts/lint/git-black.sh origin/main

echo "Running pylint on python/mnm and scripts/op_def"
python3 -m pylint python/mnm scripts/op_def --rcfile=./scripts/lint/pylintrc

echo "Running pylint on tests/python"
python3 -m pylint tests/python --rcfile=./scripts/lint/pytestlintrc

