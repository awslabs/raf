#!/usr/bin
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -ex

if [ "$#" -lt 1 ]; then
    echo "Usage: ubuntu_install_torch.sh <cpu|cu113>"
    exit 1
fi
PLATFORM=$1

PT_VERSION=1.12.0
TV_VERSION=0.13.0

# Install PyTorch and torchvision
python3 -m pip install --force-reinstall torch==$PT_VERSION+$PLATFORM torchvision==$TV_VERSION+$PLATFORM \
    -f https://download.pytorch.org/whl/$PLATFORM/torch_stable.html

