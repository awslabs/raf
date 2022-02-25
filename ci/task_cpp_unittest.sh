#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -u
set -e
set -o pipefail
source ./ci/env.sh
cd build && make test && cd ..
