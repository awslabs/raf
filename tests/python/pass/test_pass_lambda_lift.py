import pytest
import numpy as np
import mnm
from mnm._lib import relay

def randn(shape, *, ctx="cpu", dtype="float32"):
    x = np.random.randn(*shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    m_x.requires_grad = True
    return m_x, n_x


def get_ctx_list():
    ret = ["cpu"]
    if mnm.build.with_cuda():
        ret.append("cuda")
    return ret

@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_basic(ctx, shape):
    # pylint: disable=protected-access
    # Create a symbolic model and run it
    class Add(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):  # pylint: disable=no-self-use
            return mnm.add(x, y)

    # Get a Relay func
    model = Add()
    m_x, _ = randn(shape, ctx=ctx)
    m_y, _ = randn(shape, ctx=ctx)
    _ = model(m_x, m_y)
    func = model.get_relay_func()

    # Run AutoDiff to get nested functions
    # The backward function will be lifted
    func = mnm._ffi.pass_.AutoDiff(func)

    # Create a Meta module and set the func as main
    mod = mnm._ffi.ir._make.Module({relay.GlobalVar("main"): func})

    # Call Lambda lift pass on the Meta module
    lifted_mod = mnm._ffi.pass_.LambdaLift(mod)

    assert len(lifted_mod.functions) == 2


if __name__ == "__main__":
    pytest.main([__file__])
