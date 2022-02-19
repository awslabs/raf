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

# pylint: disable=attribute-defined-outside-init
import sys
import numpy as np
import pytest
from tvm import relay
import mnm
from mnm.testing import run_vm_model
from mnm.model.trace import trace_mutate_attr


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("shape", [(4, 4), (100,)])
def test_threefry_generate(device, shape):
    class ThreefryGenerate(mnm.Model):
        def build(self, key, shape):
            self.key = key
            self.shape = shape

        @mnm.model.trace
        def forward(self):
            res = mnm.threefry_generate(self.key, self.shape)
            trace_mutate_attr(self, "key", res[0])
            return res[1]

    seed = np.random.randint(0, high=sys.maxsize)
    key = mnm.array(relay.random.threefry_key(seed).data.numpy(), dtype="uint64", device=device)
    model = ThreefryGenerate(key, shape)
    m_y0 = model()
    for _ in range(10):
        m_y = model()
        if m_y == m_y0:
            raise ValueError("Random state not updated.")
    v_y0 = run_vm_model(model, device, [])
    for _ in range(10):
        v_y = run_vm_model(model, device, [])
        if v_y == v_y0:
            raise ValueError("Random state not updated.")


if __name__ == "__main__":
    pytest.main([__file__])
