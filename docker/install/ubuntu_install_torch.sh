#!/usr/bin
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -ex

if [ "$#" -lt 2 ]; then
    echo "Usage: ubuntu_install_torch.sh <cpu|cu113> <nightly|pinned|version>"
    exit 1
fi
PLATFORM=$1
VERSION=$2

# Install PyTorch
if [ "$VERSION" == "nightly" ]; then
    # Nightly build
    python3 -m pip install --force-reinstall --pre torch torchvision -f https://download.pytorch.org/whl/nightly/$PLATFORM/torch_nightly.html
elif [ "$VERSION" == "pinned" ]; then
    python3 -m pip install --force-reinstall --pre torch==$PINNED_NIGHTLY_VERSION+$PLATFORM torchvision -f https://download.pytorch.org/whl/nightly/$PLATFORM/torch_nightly.html
else
    # Stable build
    python3 -m pip install --force-reinstall torch==$VERSION+$PLATFORM torchvision -f https://download.pytorch.org/whl/$PLATFORM/torch_stable.html
fi

