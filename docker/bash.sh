#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Start a bash, mount /workspace to be current directory.
#
# Usage: docker/bash.sh <CONTAINER_NAME>
#     Starts an interactive session
#
# Usage2: docker/bash.sh <CONTAINER_NAME> [COMMAND]
#     Execute command in the docker image, non-interactive
#
if [ "$#" -lt 1 ]; then
    echo "Usage: docker/bash.sh <CONTAINER_NAME> [COMMAND]"
    exit -1
fi

DOCKER_BINARY="docker"
DOCKER_IMAGE_NAME=("$1")

if [ "$#" -eq 1 ]; then
    COMMAND="bash"
    if [[ $(uname) == "Darwin" ]]; then
        # Docker's host networking driver isn't supported on macOS.
        # Use default bridge network and expose port for jupyter notebook.
        CI_DOCKER_EXTRA_PARAMS=("-it -p 8888:8888")
    else
        CI_DOCKER_EXTRA_PARAMS=("-it --net=host")
    fi
else
    shift 1
    COMMAND=("$@")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(pwd)"

if [[ "${DOCKER_IMAGE_NAME}" == *"gpu"* ]] && [ -x "$(command -v nvidia-smi)" ]; then
    GPUS="--gpus all"
else
    GPUS=""
fi

if [[ ! -z $CUDA_VISIBLE_DEVICES ]]; then
    CUDA_ENV="-e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
    CUDA_ENV=""
fi

# Print arguments.
echo "WORKSPACE: ${WORKSPACE}"
echo "DOCKER CONTAINER NAME: ${DOCKER_IMAGE_NAME}"
echo ""

echo "Running '${COMMAND[@]}' inside ${DOCKER_IMAGE_NAME}..."

# By default we cleanup - remove the container once it finish running (--rm)
# and share the PID namespace (--pid=host) so the process inside does not have
# pid 1 and SIGKILL is propagated to the process inside (jenkins can kill it).
${DOCKER_BINARY} run --rm --pid=host\
    -v ${WORKSPACE}:/workspace \
    -v ${SCRIPT_DIR}:/docker \
    -w /workspace \
    -e "CI_BUILD_HOME=/workspace" \
    -e "CI_BUILD_USER=$(id -u -n)" \
    -e "CI_BUILD_UID=$(id -u)" \
    -e "CI_BUILD_GROUP=$(id -g -n)" \
    -e "CI_BUILD_GID=$(id -g)" \
    ${GPUS}\
    ${CUDA_ENV}\
    ${CI_DOCKER_EXTRA_PARAMS[@]} \
    ${DOCKER_IMAGE_NAME}\
    bash --login /docker/with_the_same_user \
    ${COMMAND[@]}
