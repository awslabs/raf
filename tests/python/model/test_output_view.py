# pylint: disable=no-self-use, invalid-name, protected-access
import pytest
import numpy as np

import mnm


@pytest.mark.parametrize("view_op", [
    (mnm._op.sym.batch_flatten, None),
    (mnm._op.sym.reshape, ((4, 64),)),
    (mnm._op.sym.expand_dims, (0, 1))
])
def test_output_view(view_op):
    class TestOp(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x):
            op, args = view_op
            ret = op(x, *args) if args else op(x)
            return ret

    class TestOutputView(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x):
            y = mnm.add(x, x)
            op, args = view_op
            ret = op(y, *args) if args else op(y)
            return ret

    class TestOutputViewGPU(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x):
            y = mnm.max_pool2d(x, 1, 1)
            op, args = view_op
            ret = op(y, *args) if args else op(y)
            return ret

    x = np.random.randn(4, 4, 4, 4).astype("float32")
    x = mnm.array(x)
    model1 = TestOp()
    model2 = TestOutputView()
    y1 = model1(x)
    y2 = model2(x)
    np.testing.assert_equal(y1.numpy().flatten(), x.numpy().flatten())
    np.testing.assert_equal(y2.numpy().flatten(), x.numpy().flatten() * 2)
    np.testing.assert_equal(y1.numpy() * 2, y2.numpy())

    if mnm.build.with_cuda():
        x = x.to(device='cuda')
        model3 = TestOutputViewGPU()
        y1 = model1(x)
        y3 = model3(x)
        np.testing.assert_equal(y1.numpy().flatten(), x.numpy().flatten())
        np.testing.assert_equal(y3.numpy().flatten(), x.numpy().flatten())
        np.testing.assert_equal(y1.numpy(), y3.numpy())


if __name__ == "__main__":
    pytest.main([__file__])
