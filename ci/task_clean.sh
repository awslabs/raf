#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail
source ./ci/env.sh

echo "Cleanup data..."
BUILD_DIR=$1
cd $BUILD_DIR && find . -type f,l ! -name 'config.cmake' -delete && find . -type d -delete && cd ..
