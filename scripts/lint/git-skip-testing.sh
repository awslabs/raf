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

FOUND_CHANGED_FILES=0
CHANGE_IN_SKIP_DIR=0
CHANGE_IN_OTHER_DIR=0
SKIP_DIRS="docs/ docker/"

changed_files=`git diff --no-commit-id --name-only -r origin/main`

for file in $changed_files; do
    FOUND_CHANGED_FILES=1
    CHANGE_IN_SKIP_DIR=0
    for dir in $SKIP_DIRS; do
        if grep -q "$dir" <<< "$file"; then
            CHANGE_IN_SKIP_DIR=1
            break
        fi
    done
    if [ ${CHANGE_IN_SKIP_DIR} -eq 0 ]; then
        CHANGE_IN_OTHER_DIR=1
        break
    fi
done

if [ ${FOUND_CHANGED_FILES} -eq 0 -o ${CHANGE_IN_OTHER_DIR} -eq 0 ]; then
    # Skip testing if no files changed or if change is in a non-skip dir
    echo "Rest tests can be skipped"
    exit 1
else
    exit 0
fi
