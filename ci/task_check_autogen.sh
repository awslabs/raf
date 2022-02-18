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

echo "Checking auto generated files by scripts/src_codegen/run_all.sh..."
git checkout --recurse-submodules .
./scripts/src_codegen/run_all.sh
if [[ ! -z `git status --porcelain --ignore-submodules *.cc *.py *.h` ]]; then
   echo "ERROR: Some auto generated source code files are not committed. Please make sure to check in the changes on these files."
   git status --porcelain
   exit 1
fi

echo "Checking auto generated files by docs/wiki/gen_readme.py..."
python3 docs/wiki/gen_readme.py docs/wiki
if [[ ! -z `git status --porcelain --ignore-submodules *.md` ]]; then
   echo "ERROR: Some auto generated source code files are not committed. Please make sure to check in the changes on these files."
   git status --porcelain
   exit 1
fi

