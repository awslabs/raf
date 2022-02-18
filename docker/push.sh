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

set -e

#
# Push the docker image to docker hub.
#
# Usage: push.sh <CONTAINER_TYPE> <VERSION> <PASSWORD>
#
# CONTAINER_TYPE: It can be ci_cpu or ci_gpu.
#                 The image named "mnm.<CONTAINER_TYPE>:latest" will be pushed.
#
# VERSION: The pushed version. The format should be like v0.12.
#
# PASSWORD: The Meta docker hub account password.
#
DOCKER_HUB_ACCOUNT=metaprojdev

# Get the container type.
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1

# Get the version.
VERSION=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1

# Get the docker hub account password.
PASSWORD="$1"
shift 1

LOCAL_IMAGE_NAME=mnm.${CONTAINER_TYPE}:latest
REMOTE_IMAGE_NAME=${DOCKER_HUB_ACCOUNT}/meta:${CONTAINER_TYPE}-${VERSION}

echo "Login docker hub"
docker login -u ${DOCKER_HUB_ACCOUNT} -p ${PASSWORD}

echo "Uploading ${LOCAL_IMAGE_NAME} as ${REMOTE_IMAGE_NAME}"
docker tag ${LOCAL_IMAGE_NAME} ${REMOTE_IMAGE_NAME}
docker push ${REMOTE_IMAGE_NAME}

