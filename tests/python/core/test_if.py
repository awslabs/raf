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
from mnm.testing import get_testable_devices, randn, check
from mnm._core.ndarray import get_ndarray_handle, ndarray
from mnm._core.module import IRModule
from mnm._lib import relay
from mnm._ffi.model import RunModel


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("shape", [[3, 3], [4, 4]])
def test_basic_if(device, shape):
    # pylint: disable=too-many-locals
    x = relay.var("x")
    cond_p = relay.var("cond_p")
    cond_q = relay.var("cond_q")
    a = relay.var("a")
    b = relay.var("b")
    c = relay.var("c")
    cond = relay.var("cond")
    ret = relay.var("ret")
    lets = [
        (a, mnm.ir.op.cos(x)),
        (cond, mnm.ir.op.greater(cond_p, cond_q)),
        (b, mnm.ir.op.subtract(a, x)),
        (c, mnm.ir.op.add(a, x)),
        (ret, relay.If(cond, b, c)),
        ret,
    ]

    def assemble(lets):
        if len(lets) == 1:
            return lets[0]
        var, value = lets[0]
        return relay.Let(var, value, assemble(lets[1:]))

    body = assemble(lets)
    func = relay.Function([x, cond_p, cond_q], body)
    m_x, n_x = randn(shape, device=device)
    m_p, n_p = randn((), device=device)
    m_q, n_q = randn((), device=device)
    inputs = [m_x, m_p, m_q]
    inputs = [get_ndarray_handle(arg) for arg in inputs]
    mod = IRModule.from_expr(func)
    m_y = RunModel(mod, inputs)
    m_y = ndarray(m_y)
    if n_p[()] <= n_q[()]:
        n_y = np.cos(n_x) + n_x
    else:
        n_y = np.cos(n_x) - n_x
    check(m_y, mnm.array(n_y, device=device))


if __name__ == "__main__":
    pytest.main([__file__])
