#!/bin/bash
set -e
set -u
set -o pipefail
source ./ci/env.sh

echo "clang-format check..."
# check lastest change, for squash merge into master
./scripts/lint/git-clang-format.sh HEAD~1
# chekc against origin/master for PRs.
./scripts/lint/git-clang-format.sh origin/master

echo "Running pylint on python/mnm and scripts/op_def"
python3 -m pylint python/mnm scripts/op_def --rcfile=./scripts/lint/pylintrc

echo "Running pylint on tests/python"
python3 -m pylint tests/python --rcfile=./scripts/lint/pytestlintrc
