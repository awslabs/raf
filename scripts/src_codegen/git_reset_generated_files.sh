#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# See git_add_generated_files.sh for details and usage.

SCRIPT_DIR=$(dirname "$0")
(< $SCRIPT_DIR/generated_file_list.txt xargs -i find {} -type f) | xargs git reset HEAD
