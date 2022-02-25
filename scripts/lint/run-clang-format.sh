#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source $SCRIPTPATH/../env.sh

if [ -z $1 ]; then
  repo=`git rev-parse --show-toplevel`
  find $repo/{src,tests,include,apps} \( -iname \*.cc -o -iname \*.h \) | xargs $CLANG_FORMAT_BIN -i
else
  $CLANG_FORMAT_BIN -i "$@"
fi
