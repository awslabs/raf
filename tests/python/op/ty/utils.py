import numpy as np
import mnm
from tvm import relay
from mnm._ffi.pass_ import InferType


def check_type(expr, typ):
    checked_type = expr.checked_type
    if checked_type != typ:
        raise RuntimeError(f"Type mismatch {checked_type} vs {typ}")


def run_infer_type(func):
    # pylint: disable=protected-access
    mod = mnm._ffi.ir._make.Module({relay.GlobalVar("main"): func})
    mod = InferType(mod)
    return mod['main']


def randn(shape, *, ctx="cpu", dtype="float32"):
    x = np.random.randn(*shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    return m_x, n_x
