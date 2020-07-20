import numpy as np
import pytest
import mnm
import tvm

def get_ctx_list():
    ret = ["cpu"]
    if mnm.build.with_cuda():
        ret.append("cuda")
    return ret

def randn(shape, *, ctx="cpu", dtype="float32"):
    x = np.random.randn(*shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    m_x.requires_grad = True
    return m_x, n_x

def check(m_x, n_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    np.testing.assert_allclose(m_x, n_x, rtol=rtol, atol=atol)

@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_fold_const_model(ctx, shape):
    const, _ = randn(shape, ctx=ctx)
    class ModelWithConst(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            self.c = const

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            y = mnm.add(self.c, self.c)
            return mnm.add(x, y)

    model = ModelWithConst()
    m_x, _ = randn(shape, ctx=ctx)
    m_y = model(m_x)
    m_dy, n_dy = randn(shape, ctx=ctx)
    m_y.backward(m_dy)
    m_dx = m_x.grad
    n_dx = 1 * n_dy
    check(m_dx, n_dx)
    check(m_y, mnm.add(mnm.add(const, const), m_x).asnumpy())


@pytest.mark.parametrize("ctx", get_ctx_list()[1:])
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_fold_const_ir(ctx, shape):
    # pylint: disable=protected-access
    const, _ = randn(shape, ctx=ctx)
    class ModelWithConst(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            self.c = const

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            y = mnm.matmul(self.c, self.c)
            z = mnm.matmul(x, y)
            return mnm.matmul(x, z)

    def expected():
        x = tvm.relay.var('x')
        c = tvm.relay.var('c')
        # we are only interested in the structure
        t_value = mnm._core.value.TensorValue.from_numpy(const.asnumpy())
        const_var = mnm._ffi.ir._make.Constant(t_value)
        matmul_op = mnm._ffi.op.GetOp('mnm.op.matmul')
        closure2 = tvm.relay.Call(matmul_op, [x, const_var])
        var_a2 = tvm.relay.var('a2')
        var_a3 = tvm.relay.var('a3')
        closure3 = tvm.relay.Call(matmul_op, [x, var_a2])
        let3 = tvm.relay.Let(var_a3, closure3, var_a3)
        let2 = tvm.relay.Let(var_a2, closure2, let3)
        return tvm.relay.Function([x, c], let2)

    model_before = ModelWithConst()
    model_before.infer_mode()
    m_x, _ = randn(shape, ctx=ctx)

    func_before = model_before.get_relay_func(m_x)
    print(func_before)

    # bind parameters
    args = [m_x._ndarray__handle, model_before.c._ndarray__handle]
    func_bound = mnm._ffi.pass_.BindParam(func_before, args)
    print(func_bound)

    # fold constant
    func_folded = mnm._ffi.pass_.FoldConstant(func_bound, mnm._ffi.ir.module.Global())
    print(func_folded)

    func_expected = expected()
    print(func_expected)

    assert tvm.ir.structural_equal(func_folded, func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
