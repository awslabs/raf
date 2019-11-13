#!/bin/bash
set -e
set -u
set -o pipefail
source ./ci/env.sh

echo "Running pylint on tests/python"
python3 -m pylint ${MNM_HOME}/tests/python --rcfile=${MNM_HOME}/tests/lint/pylintrc

echo "Running pylint on python/mnm"
python3 -m pylint ${MNM_HOME}/python/mnm --rcfile=${MNM_HOME}/tests/lint/pylintrc
