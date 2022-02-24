#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ROOTPATH=$SCRIPTPATH/../../

python3 -m pylint $ROOTPATH/python/raf $ROOTPATH/scripts/op_def --rcfile=$ROOTPATH/scripts/lint/pylintrc
python3 -m pylint $ROOTPATH/tests/python --rcfile=$ROOTPATH/scripts/lint/pytestlintrc
$ROOTPATH/scripts/lint/git-clang-format.sh HEAD~1
