
# pylint: disable=unused-import, attribute-defined-outside-init, protected-access
# pylint: disable=missing-module-docstring, missing-function-docstring, no-self-use
import pytest
import mnm
from mnm.frontend import FrameworkModel
from mnm.testing import check, randint


def verify_device(m_model, args=None):
    args = [] if args is None else args
    mod = m_model._internal(*args).mod
    ref_out = m_model(*args)

    # Assign device now only applies to FrameworkModel.
    m_model = FrameworkModel(mod, mod, {}, {})
    m_model.to(device="cuda")
    args = [arg.to(device="cuda") for arg in args]
    out = m_model(*args)
    assert out.device.startswith("cuda")
    check(ref_out, out)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("op_n_args", [
    (mnm._op.sym.zeros, {"shape": (3, 2), "dtype": "float32", "device": "cpu"}),
    (mnm._op.sym.ones, {"shape": (3, 2), "dtype": "float32", "device": "cpu"}),
    (mnm._op.sym.full, {"fill_value": 0, "shape": (3, 2), "dtype": "float32", "device": "cpu"}),
])
def test_init_ops(op_n_args):
    class Model(mnm.Model):
        def build(self, op_n_args):
            self.op, self.args = op_n_args

        @mnm.model.trace
        def forward(self):
            return self.op(**self.args)

    verify_device(Model(op_n_args))


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_one_hot():
    class Model(mnm.Model):
        def build(self, depth, dtype, device):
            self.depth = depth
            self.dtype = dtype
            self.device = device

        @mnm.model.trace
        def forward(self, indices, on_value, off_value):
            return mnm.one_hot(indices, on_value, off_value, depth=self.depth,
                               dtype=self.dtype, device=self.device)

    m_indices, _ = randint(shape=(1, 2, 3), high=10, device="cpu")
    m_on_value = mnm.array(1.0, device="cpu")
    m_off_value = mnm.array(0.0, device="cpu")
    verify_device(Model(3, "float32", "cpu"), [m_indices, m_on_value, m_off_value])


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_arange():
    start, stop, step = (1, 10, 2)
    dtype = "float32"
    device = "cpu"

    class Model(mnm.Model):
        def build(self, dtype, device):
            self.dtype = dtype
            self.device = device

        @mnm.model.trace
        def forward(self, start, stop, step):
            return mnm.arange(start, stop, step, dtype=self.dtype, device=self.device)

    m_start = mnm.array(start, dtype=dtype, device=device)
    m_stop = mnm.array(stop, dtype=dtype, device=device)
    m_step = mnm.array(step, dtype=dtype, device=device)
    verify_device(Model(dtype, device), [m_start, m_stop, m_step])


if __name__ == "__main__":
    pytest.main([__file__])
