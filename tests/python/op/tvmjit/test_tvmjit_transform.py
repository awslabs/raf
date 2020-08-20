from functools import reduce
import operator

import numpy as np
import pytest
import mxnet as mx
import torch
import mnm
import topi.testing


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
    return m_x, n_x


def randint(shape, *, low=0, high=None, ctx="cpu", dtype="int64"):
    x = np.random.randint(low, high, shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    return m_x, n_x


def check(m_x, n_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    np.testing.assert_allclose(m_x, n_x, rtol=rtol, atol=atol)


def randn_torch(shape, *, ctx="cpu", dtype="float32", std=1.0):
    x = np.random.randn(*shape) * std
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    x = x.astype(dtype)
    m_x = mnm.array(x, ctx=ctx)
    t_x = torch.tensor(x, requires_grad=True)  # pylint: disable=not-callable
    return m_x, t_x


def check_torch(m_x, t_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    t_x = t_x.detach().cpu().numpy()
    np.testing.assert_allclose(m_x, t_x, rtol=rtol, atol=atol)


# pylint: disable=too-many-locals
# pylint: disable=attribute-defined-outside-init
# pylint: disable=no-member
@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [
    [(5, 4, 3), (1, 2)],
    [(6, 5), (2, 2)],
    [(1, 1), (2, 2, 2)],
])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_take(shape, axis, ctx):
    class Take(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, x, indices):
            return mnm.take(x, indices=indices, axis=self._axis)
    size = reduce(operator.mul, shape[0], 1) if axis is None else shape[0][axis]
    m_x, n_x = randn(shape[0], ctx=ctx)
    m_x.requires_grad = True
    m_indices, n_indices = randint(shape[1], low=0, high=size, ctx=ctx)
    model = Take(axis)
    m_y = model(m_x, m_indices)
    n_y = np.take(n_x, n_indices, axis=axis, mode="clip")
    # check forward
    check(m_y, n_y)
    # check backward
    m_dy, n_dy = randn(n_y.shape, ctx=ctx)
    mx_x = mx.nd.array(n_x)
    mx_x.attach_grad()
    mx_dy = mx.nd.array(n_dy)
    mx_indices = mx.nd.array(n_indices)
    with mx.autograd.record():
        mx_y = mx.nd.take(mx_x, indices=mx_indices, axis=axis, mode="clip")
        mx_y.backward(mx_dy)
    m_y.backward(m_dy)
    check(m_x.grad, mx_x.grad.asnumpy())


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("max_length", [3, 4, 5, 6])
@pytest.mark.parametrize("batch_size", [2, 3, 4])
@pytest.mark.parametrize("other_feature_dims", [[1, 2], [3, 4], [5, 6]])
@pytest.mark.parametrize("axis", [0, 1])
def test_sequence_mask(max_length, batch_size, other_feature_dims,
                       axis, ctx):
    x_shape = [max_length, batch_size] if axis == 0 else [batch_size, max_length]
    x_shape += other_feature_dims
    m_x, n_x = randn(x_shape, ctx=ctx)
    m_length, n_length = randint([batch_size], low=0, high=max_length, ctx=ctx)
    m_y = mnm.sequence_mask(m_x, m_length, axis=axis, mask_value=-10)
    n_y = topi.testing.sequence_mask(n_x, n_length, axis=axis, mask_value=-10)
    check(m_y, n_y)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [
    [[1, 4, 1], [1, 2, 4, 1]],
    [[4, 1, 1], [3, 4, 2, 2]]
])
def test_broadcast_to(shape, ctx):
    m_x, n_x = randn(shape[0], ctx=ctx)
    m_y = mnm.broadcast_to(m_x, shape[1])
    n_y = np.broadcast_to(n_x, shape[1])
    check(m_y, n_y)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [
    (1, 4, 1),
    (3, 4, 2, 2),
    (4, 1, 1),
    (1, 2, 4, 1)
])
def test_repeat(shape, ctx):
    m_x, n_x = randn(shape, ctx=ctx)
    ndim = len(shape)
    for axis in range(-ndim, ndim):
        m_y = mnm.repeat(m_x, 2, axis)
        n_y = np.repeat(n_x, 2, axis)
        check(m_y, n_y)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [
    [(2, 2), (1, 0)],
    [(2, 2), None],
    [(2, 2, 2), (1, 2, 0)],
    [(2, 2, 2), (2, 1, 0)],
    [(2, 2, 2), None],
    [(4, 4, 4, 4), (3, 2, 1, 0)],
    [(4, 4, 4, 4), (1, 2, 3, 0)]
])  # pylint: disable-msg=too-many-locals
def test_transpose(shape, ctx):

    class Transpose(mnm.Model):
        def build(self, axes=None):
            self._axes = axes  # pylint: disable=attribute-defined-outside-init

        @mnm.model.trace
        def forward(self, x):
            ret = mnm.transpose(x, self._axes)
            return ret

    axes = shape[1]
    model = Transpose(axes)
    m_x, n_x = randn(shape[0], ctx=ctx)
    m_x.requires_grad = True
    m_y = model(m_x)
    n_y = np.transpose(n_x, shape[1])
    # check forward
    check(m_y, n_y)
    # check backward
    y_shape = n_y.shape
    m_dy, n_dy = randn(y_shape, ctx=ctx)
    if axes is not None:
        axes_inverse = list(axes).copy()
        for idx, i in enumerate(list(axes)):
            axes_inverse[i] = idx
        n_x_grad = np.transpose(n_dy, axes=tuple(axes_inverse))
    else:
        n_x_grad = np.transpose(n_dy)
    m_y.backward(m_dy)
    check(m_x.grad, n_x_grad)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [
    [[1, 4, 1], [1, 4, 1]],
    [[1, 4, 1], [1, 2, 4, 1]],
    [[4, 1, 1], [3, 4, 2, 2]]
])
def test_broadcast_to_like(shape, ctx):
    m_x, n_x = randn(shape[0], ctx=ctx)
    m_broadcast_type, _ = randn(shape[1], ctx=ctx)
    m_y = mnm.broadcast_to_like(m_x, m_broadcast_type)
    n_y = np.broadcast_to(n_x, shape[1])
    check(m_y, n_y)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [[10, 20, 30], [6, 8, 9, 3]])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("indices_or_sections", [(2, 4), (1, 4)])
def test_split(shape, axis, indices_or_sections, ctx):
    m_x, n_x = randn(shape, ctx=ctx)
    m_y = mnm.split(m_x, indices_or_sections=indices_or_sections, axis=axis)
    n_y = np.split(n_x, indices_or_sections=indices_or_sections, axis=axis)
    assert len(m_y) == len(n_y)
    for m, n in zip(m_y, n_y):
        check(m, n)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("inputs", [
    {"shape": (3, 3, 3), "seq_length": [1, 2, 3]},
    {"shape": (5, 5, 5), "seq_length": [1, 2, 3, 4, 5]},
    {"shape": (5, 5, 5), "seq_length": [2, 2, 3, 3, 4]}
])
@pytest.mark.parametrize("axes", [[0, 1]])
def test_reverse_sequence(inputs, axes, ctx):
    class ReverseSequence(mnm.Model):
        def build(self, seq_axis, batch_axis):
            self._seq_axis = seq_axis
            self._batch_axis = batch_axis

        @mnm.model.trace
        def forward(self, x, seq_length):
            return mnm.reverse_sequence(x, seq_length, self._seq_axis, self._batch_axis)
    shape = inputs["shape"]
    m_seq_length = mnm.array(inputs["seq_length"], dtype=int, ctx=ctx)
    mx_seq_length = mx.nd.array(inputs["seq_length"], dtype=int)
    seq_axis = axes[0]
    batch_axis = axes[1]
    m_x, n_x = randn(shape, dtype='float32', ctx=ctx)
    m_dy, n_dy = randn(shape, dtype='float32', ctx=ctx)
    mx_x = mx.nd.array(n_x)
    mx_dy = mx.nd.array(n_dy)
    mx_x.attach_grad()
    m_x.requires_grad = True
    model = ReverseSequence(seq_axis, batch_axis)

    m_y = model(m_x, m_seq_length)
    with mx.autograd.record():
        mx_y = mx.nd.SequenceReverse(mx_x, mx_seq_length, use_sequence_length=True)
        # check forward
        check(m_y, mx_y.asnumpy())
        mx_y.backward(mx_dy)
    m_y.backward(m_dy)
    # check backward
    check(m_x.grad, mx_x.grad.asnumpy())


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [[10, 10, 10], [6, 8, 9, 10]])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reverse(shape, axis, ctx):
    class Reverse(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, x):
            return mnm.reverse(x, self._axis)

    m_x, n_x = randn(shape, dtype='float32', ctx=ctx)
    m_dy, n_dy = randn(shape, dtype='float32', ctx=ctx)
    m_x.requires_grad = True
    model = Reverse(axis=axis)
    m_y = model(m_x)
    # check forward
    n_y = np.flip(n_x, axis=axis)
    check(m_y, n_y)
    # check backward
    m_y.backward(m_dy)
    n_grad = np.flip(n_dy, axis=axis)
    check(m_x.grad, n_grad)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("params", [
    {"shapes": [[1, 4, 1], [2, 4, 1]], "axis": 0},
    {"shapes": [[2, 2, 2], [2, 3, 2], [2, 4, 2]], "axis": -2},
    {"shapes": [[2, 1, 1], [2, 2, 1], [2, 3, 1], [2, 4, 1]], "axis": 1},
])
def test_concatenate(params, ctx):
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=not-callable
    # pylint: disable=no-member
    # pylint: disable=too-many-locals
    class Concatenate1(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a):
            return mnm.concatenate([a], axis=self._axis)

    class Concatenate2(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a, b):
            return mnm.concatenate([a, b], axis=self._axis)

    class Concatenate3(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a, b, c):
            return mnm.concatenate([a, b, c], axis=self._axis)

    class Concatenate4(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a, b, c, d):
            return mnm.concatenate([a, b, c, d], axis=self._axis)

    concat = [None, Concatenate1, Concatenate2, Concatenate3, Concatenate4]
    shapes, axis = params["shapes"], params["axis"]
    m_i, t_i = [], []
    for shape in shapes:
        m_x, t_x = randn_torch(shape, ctx=ctx)
        m_x.requires_grad = True
        m_i.append(m_x)
        t_i.append(t_x)
    model = concat[len(m_i)](axis=axis)
    m_y = model(*m_i)
    t_y = torch.cat(t_i, dim=axis)
    # check forward
    check_torch(m_y, t_y)
    # check backward
    m_dy, t_dy = randn_torch(tuple(t_y.size()), ctx=ctx)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    for m_x, t_x in zip(m_i, t_i):
        check_torch(m_x.grad, t_x.grad)



@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("params", [
    {"shapes": [[1, 4, 1], [1, 4, 1]], "axis": 0},
    {"shapes": [[2, 2, 2], [2, 2, 2], [2, 2, 2]], "axis": -1},
    {"shapes": [[2, 1, 1], [2, 1, 1], [2, 1, 1], [2, 1, 1]], "axis": 1},
])
def test_stack(params, ctx):
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=not-callable
    # pylint: disable=no-member
    # pylint: disable=too-many-locals
    class Stack1(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a):
            return mnm.stack([a], axis=self._axis)

    class Stack2(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a, b):
            return mnm.stack([a, b], axis=self._axis)

    class Stack3(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a, b, c):
            return mnm.stack([a, b, c], axis=self._axis)

    class Stack4(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a, b, c, d):
            return mnm.stack([a, b, c, d], axis=self._axis)

    stack = [None, Stack1, Stack2, Stack3, Stack4]
    shapes, axis = params["shapes"], params["axis"]
    m_i, n_i = [], []
    for shape in shapes:
        m_x, n_x = randn(shape, ctx=ctx)
        m_i.append(m_x)
        n_i.append(n_x)
    model = stack[len(m_i)](axis=axis)
    m_y = model(*m_i)
    n_y = np.stack(n_i, axis=axis)
    # check forward
    check(m_y, n_y)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [(1, 3), (1, 2), (4, 3, 2, 1),
                                   (2, 4, 1, 3), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("a_min", [0.1, 0.2, 0.3, 0.4, 0.5])
@pytest.mark.parametrize("a_max", [0.6, 0.7, 0.8, 0.9, 1.0])
def test_clip(shape, a_min, a_max, ctx):
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=not-callable
    # pylint: disable=no-member
    # pylint: disable=too-many-locals
    # pylint: disable=no-self-use
    class Clip(mnm.Model):
        def build(self, shape):
            self._shape = shape

        @mnm.model.trace
        def forward(self, x):
            return mnm.clip(x, a_min, a_max)

    m_x, n_x = randn(shape, dtype='float32', ctx=ctx)
    m_dy, n_dy = randn(shape, dtype='float32', ctx=ctx)
    m_x.requires_grad = True
    model = Clip(shape=shape)
    m_y = model(m_x)
    # check forward
    n_y = np.clip(n_x, a_min, a_max)
    check(m_y, n_y)
    # check backward
    m_y.backward(m_dy)
    n_s = np.where(n_x <= a_min, 0, 1)
    n_grad = n_s * n_dy
    n_s = np.where(n_x >= a_max, 0, 1)
    n_grad = n_s * n_grad
    check(m_x.grad, n_grad)

@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("params", [
    {"orig_shape": (8, 8, 8, 8), "to_shape": (2, 2048)},
    {"orig_shape": (8, 1000), "to_shape": (2, 2, 2, 1000)},
    {"orig_shape": (3, 3, 3, 3), "to_shape": (81, 1)},
])
def test_reshape(params, ctx):
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=not-callable
    # pylint: disable=no-member
    # pylint: disable=too-many-locals
    class Reshape(mnm.Model):
        def build(self, shape):
            self._shape = shape

        @mnm.model.trace
        def forward(self, x):
            return mnm.reshape(x, shape=self._shape)

    orig_shape, to_shape = params["orig_shape"], params["to_shape"]
    m_x, n_x = randn(orig_shape, ctx=ctx)
    m_dy, n_dy = randn(to_shape, ctx=ctx)
    m_x.requires_grad = True
    model = Reshape(shape=to_shape)
    m_y = model(m_x)
    # check forward
    n_y = np.reshape(n_x, to_shape)
    n_dy = np.reshape(n_dy, orig_shape)
    check(m_y, n_y)
    # check backward
    m_y.backward(m_dy)
    check(m_x.grad, n_dy)

@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [
    [10, 3, 2, 5],
    [1, 4, 5, 2],
    [9, 12, 18, 2, 1]])
@pytest.mark.parametrize("axis", [0, 1, 2, 3])
@pytest.mark.parametrize("num_newaxis", [0, 1, 2, 5])
def test_expand_dims(ctx, shape, axis, num_newaxis):
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=not-callable
    # pylint: disable=no-member
    # pylint: disable=too-many-locals
    class ExpandDims(mnm.Model):
        def build(self, axis, num_newaxis):
            self.axis = axis
            self.num_newaxis = num_newaxis

        @mnm.model.trace
        def forward(self, x):
            return mnm.expand_dims(x, axis=self.axis, num_newaxis=self.num_newaxis)

    m_x, n_x = randn(shape, ctx=ctx)
    m_x.requires_grad = True
    model = ExpandDims(axis, num_newaxis)
    m_y = model(m_x)
    # check forward
    n_y = n_x
    if num_newaxis == 0:
        pass
    elif num_newaxis == 1:
        n_y = np.expand_dims(n_y, axis=axis)
    else:
        for _ in range(num_newaxis):
            n_y = np.expand_dims(n_y, axis=axis)
    check(m_y, n_y)
    # check backward
    m_dy, n_dy = randn(m_y.shape, ctx=ctx)
    m_y.backward(m_dy)
    check(m_x.grad, np.reshape(n_dy, n_x.shape))

if __name__ == "__main__":
    pytest.main([__file__])
