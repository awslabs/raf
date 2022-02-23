# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-class-docstring,missing-function-docstring
"""Blocks for constructing structural Model."""
from .model import Model


class Sequential(Model):

    # pylint: disable=attribute-defined-outside-init
    def build(self, *args):
        self.num_layers = len(args)
        for idx, layer in enumerate(args):
            setattr(self, "seq_" + str(idx), layer)

    def forward(self, x):
        for idx in range(self.num_layers):
            layer = getattr(self, "seq_" + str(idx))
            x = layer(x)
        return x
