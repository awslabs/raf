#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# The script to backup or recover ccache cache with Amazon S3.
# Note that recovering ccache cache will override the existing cache.
set -e

MODE=$1     # upload or download
PLATFORM=$2 # CPU, GPU, or multi-GPU
TAG=$3      # e.g., refs/heads/main, pr-7

S3_BUCKET="ci-meta"
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
