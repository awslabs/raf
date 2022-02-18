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
