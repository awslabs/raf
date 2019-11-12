#!/bin/bash
set -u
set -e
set -o pipefail
source ./ci/env.sh
cd build && make test && cd ..
