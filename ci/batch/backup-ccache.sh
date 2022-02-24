#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# The script to backup or recover ccache cache with Amazon S3.
# Note that recovering ccache cache will override the existing cache.
set -e

MODE=$1     # upload or download
PLATFORM=$2 # CPU, GPU, or multi-GPU
TAG=$3      # e.g., refs/heads/main, pr-7

S3_BUCKET="ci-raf"
S3_FOLDER=`echo cache-${TAG} | sed 's/\//_/g'`
S3_PATH="s3://$S3_BUCKET/$S3_FOLDER"

CACHE_DIR=`ccache -k cache_dir`
CACHE_ZIP="ccache-$PLATFORM.tar.gz"

if [[ $MODE == "download" ]]; then
    echo "Download cache from $S3_PATH"
    aws s3 cp ${S3_PATH}/${CACHE_ZIP} .
    rm -rf $CACHE_DIR
    tar -xzf $CACHE_ZIP -C /

    echo "Done. Current ccache stats:"
    echo "---------------------------"
    ccache -s
    echo "---------------------------"
elif [[ $MODE == "upload" ]]; then
    echo "Upload cache to $S3_PATH/$CACHE_ZIP"
    tar -czf ./$CACHE_ZIP $CACHE_DIR
    aws s3 cp ./$CACHE_ZIP $S3_PATH/$CACHE_ZIP
else
    echo "Unrecognized mode: $MODE"
    exit 1
fi
