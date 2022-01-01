#!/bin/bash
set -e
set -u
set -o pipefail
source ./ci/env.sh

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
