#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


CLANG_FORMAT=clang-format-10

if [ -x "$(command -v clang-format-10)" ]; then
    CLANG_FORMAT=clang-format-10
elif [ -x "$(command -v clang-format)" ]; then
    echo "clang-format might be different from clang-format-10, expect potential difference."
    CLANG_FORMAT=clang-format
else
    echo "Cannot find clang-format-10"
    exit 1
fi

export CLANG_FORMAT_BIN=$CLANG_FORMAT
