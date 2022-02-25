#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail
source ./ci/env.sh

echo "Checking auto generated files by docs/wiki/gen_readme.py..."
python3 docs/wiki/gen_readme.py docs/wiki
if [[ ! -z `git status --porcelain --ignore-submodules *.md` ]]; then
   echo "ERROR: Auto generated docs/wiki/README.md is not committed"
   git status --porcelain
   exit 1
fi
