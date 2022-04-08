#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# The entry point of AWS Batch job. This script is in charge of configuring
# the repo, executing the given command, and uploading the results.
set -e

date

# Parse arguments
SOURCE_REF=$1
REPO=$2
COMMAND=$3
SAVE_OUTPUT=$4
REMOTE_FOLDER=$5 # e.g., s3://ci-raf/pr-7

echo "Job Info"
echo "-------------------------------------"
echo "jobId: $AWS_BATCH_JOB_ID"
echo "jobQueue: $AWS_BATCH_JQ_NAME"
echo "computeEnvironment: $AWS_BATCH_CE_NAME"
echo "SOURCE_REF: $SOURCE_REF"
echo "REPO: $REPO"
echo "S3_PATH: $S3_PATH"
echo "CCACHE_DIR: $CCACHE_DIR"
echo "SAVE_OUTPUT: $SAVE_OUTPUT"
echo "REMOTE_FOLDER: $REMOTE_FOLDER"
echo "COMMAND: $COMMAND"
echo "-------------------------------------"

if [ -z $GITHUB_TOKEN ]; then
    echo "GITHUB_TOKEN is not set"
    exit 1
fi;

# Checkout the repo.
git clone https://$GITHUB_TOKEN:x-oauth-basic@github.com/$REPO --recursive
cd raf

# Config the repo
git fetch origin $SOURCE_REF:working
git checkout working
git submodule update --init --recursive --force

# Execute the command
/bin/bash -o pipefail -c "$COMMAND"
COMMAND_EXIT_CODE=$?

# Upload results
if [ ! -z $SAVE_OUTPUT ]; then
    FULL_S3_PATH=$REMOTE_FOLDER/$AWS_BATCH_JOB_ID
    echo "Uploading results to $FULL_S3_PATH"
    if [[ -f $SAVE_OUTPUT ]]; then
      aws s3 cp $SAVE_OUTPUT $FULL_S3_PATH/ --quiet;
    elif [[ -d $SAVED_OUTPUT ]]; then
      aws s3 cp --recursive $SAVE_OUTPUT $FULL_S3_PATH/ --quiet;
    fi
fi

exit $COMMAND_EXIT_CODE
