#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# This is mainly borrowed from https://github.com/apache/tvm/blob/main/tests/lint/git-clang-format.sh
set -e
set -u
set -o pipefail

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source $SCRIPTPATH/../env.sh

if [[ "$1" == "-i" ]]; then
    INPLACE_FORMAT=1
    shift 1
else
    INPLACE_FORMAT=0
fi

if [[ "$#" -lt 1 ]]; then
    echo "Usage: scripts/lint/git-clang-format.sh [-i] <commit>"
    echo ""
    echo "Run clang-format on files that changed since <commit>"
    echo "Examples:"
    echo "- Compare last one commit: tests/lint/git-clang-format.sh HEAD~1"
    echo "- Compare against upstream/main: scripts/lint/git-clang-format.sh upstream/main"
    echo "You can also add -i option to do inplace format"
    exit 1
fi

cleanup()
{
  rm -rf /tmp/$$.clang-format.txt
}
trap cleanup 0

# Print out specific version
${CLANG_FORMAT_BIN} --version

if [[ ${INPLACE_FORMAT} -eq 1 ]]; then
    echo "Running inplace git-clang-format against" $1
    git-${CLANG_FORMAT_BIN} --extensions h,mm,c,cc --binary=${CLANG_FORMAT_BIN} $1
    exit 0
fi

echo "Running git-clang-format against" $1
git-${CLANG_FORMAT_BIN} --diff --extensions h,mm,c,cc --binary=${CLANG_FORMAT_BIN} $1 1> /tmp/$$.clang-format.txt
echo "---------clang-format log----------"
cat /tmp/$$.clang-format.txt
echo ""
if grep --quiet -E "diff" < /tmp/$$.clang-format.txt; then
    echo "clang-format lint error found. Consider running scripts/lint/run-clang-format.sh to fix them."
    exit 1
fi
