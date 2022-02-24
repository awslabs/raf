# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import raf
from raf.testing import get_testable_devices, randn, check
from raf._core.ndarray import get_ndarray_handle, ndarray
from raf._core.module import IRModule
from raf._lib import relay
from raf._ffi.model import RunModel


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
        (a, raf.ir.op.cos(x)),
        (cond, raf.ir.op.greater(cond_p, cond_q)),
        (b, raf.ir.op.subtract(a, x)),
        (c, raf.ir.op.add(a, x)),
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
    check(m_y, raf.array(n_y, device=device))


if __name__ == "__main__":
    pytest.main([__file__])
