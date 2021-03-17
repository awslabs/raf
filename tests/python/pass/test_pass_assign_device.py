
# pylint: disable=unused-import, attribute-defined-outside-init
# pylint: disable=missing-module-docstring, missing-function-docstring
import pytest
import mnm
import tvm
from tvm import relay as _relay
from mnm.frontend import FrameworkModel
from mnm._ffi.pass_ import FromRelay
from mnm._core.module import IRModule

@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [[3, 2]])
@pytest.mark.parametrize("val", [1])
def test_full(shape, val):
    # Meta cannot specify constant so we convert it from Relay.
    r_c = _relay.const(val)
    r_func = _relay.Function(params=[], body=_relay.full(r_c, shape=shape, dtype="int64"))
    m_func = FromRelay(r_func)
    m_mod = IRModule.from_expr(m_func)

    m_model = FrameworkModel(m_mod, m_mod, {}, {})
    m_model.to(device="cuda")
    out = m_model()
    assert out.device.startswith("cuda")
