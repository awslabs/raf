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
            >&2 echo "[Skip] $file"
            CHANGE_IN_SKIP_DIR=1
            break
        fi
    done
    if [ ${CHANGE_IN_SKIP_DIR} -eq 0 ]; then
        >&2 echo "[Non-Skip] $file...break"
        CHANGE_IN_OTHER_DIR=1
        break
    fi
done

>&2 echo "Change? ${FOUND_CHANGED_FILES}, Non-skip? ${CHANGE_IN_OTHER_DIR}"
if [ ${FOUND_CHANGED_FILES} -eq 0 -o ${CHANGE_IN_OTHER_DIR} -eq 1 ]; then
    # Cannot skip testing if
    # 1) No file change against main branch. This happens when merging to main.
    # 2) One or more changes are in non-skip dirs.
    echo "0"
else
    # Skip testing if all changes are in skip dirs
    >&2 echo "Rest tests can be skipped"
    echo "1"
fi
