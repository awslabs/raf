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

import numpy as np
import pytest
import mnm
from tvm import relay
from mnm.testing import get_testable_devices, check, run_vm_model, get_vm_executor, resnet
from mnm._core.ndarray import Symbol
from mnm.model.trace import _get_func_inputs


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("fuse", [True, False])
def test_simple(device, fuse):
    # pylint: disable=no-self-use
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.argwhere(x)
            y = mnm.split(y, 2)
            y = mnm.add(y[0], y[1])
            y = mnm.abs(y)
            return y

    model = Model()
    n_x = np.ones((2, 2)).astype("float32")
    m_x = mnm.array(n_x, device=device)
    m_res = model(m_x)
    v_res = run_vm_model(model, device, [m_x], disable_fusion=not fuse)
    expected = mnm.array([[1, 0], [1, 2]], dtype="int32")
    check(m_res, expected)
    check(v_res, expected)


@pytest.mark.parametrize("device", get_testable_devices())
def test_dynamic_reshape(device):
    # pylint: disable=no-self-use
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.argwhere(x)
            y = mnm.split(y, 2)
            y = mnm.add(y[0], y[1])
            y = mnm.abs(y)
            y = mnm.expand_dims(y, 0)
            return y

    model = Model()
    n_x = np.ones((2, 2)).astype("float32")
    m_x = mnm.array(n_x, device=device)
    m_res = model(m_x)
    v_res = run_vm_model(model, device, [m_x])
    expected = mnm.array([[[1, 0], [1, 2]]], dtype="int32")
    check(m_res, expected)
    check(v_res, expected)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("fuse", [True, False])
def test_resnet(device, fuse):
    # pylint: disable=invalid-name, protected-access
    m_model, _ = resnet.get_model([1, 1, 1, 1], False)
    m_model.infer_mode()
    m_model.to(device=device)

    x_ty = relay.TensorType((relay.Any(), 3, 224, 224))
    x = Symbol.make_var("x", x_ty)
    record = m_model._internal(x)
    mod = record.mod
    vm = get_vm_executor(mod, device, 2, not fuse)

    (m_x, _), _ = resnet.get_input(batch_size=1, device=device)
    m_x.requires_grad = False
    inputs = _get_func_inputs(record, (m_x,), {}, get_handle=False)

    v_res = vm(*inputs)
    m_res = m_model(m_x)
    check(m_res, v_res)


if __name__ == "__main__":
    pytest.main([__file__])
