import pytest
import mnm
from mnm._lib import relay
from mnm.testing import randn, get_device_list

@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_basic(device, shape):
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
    m_x, _ = randn(shape, device=device, requires_grad=True)
    m_y, _ = randn(shape, device=device, requires_grad=True)
    _ = model(m_x, m_y)
    func = model._internal().func

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
