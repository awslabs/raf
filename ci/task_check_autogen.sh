#!/bin/bash
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

