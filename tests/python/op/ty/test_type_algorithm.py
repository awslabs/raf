# pylint: disable=protected-access
import pytest
import mnm
from mnm.testing import check_type, run_infer_type, randn, get_device_list
from mnm._ffi.pass_ import AutoDiff, InferType
from tvm.relay import TensorType, FuncType, TupleType


# pylint: disable=attribute-defined-outside-init
@pytest.mark.parametrize("shape", [
    (2, 3, 4),
    (1, 4, 6),
    (3, 5, 6),
])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("is_ascend", [True, False])
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_argsort(shape, axis, is_ascend, dtype):

    class Argsort(mnm.Model):
        def build(self, axis, is_ascend, dtype):
            self._axis = axis
            self._is_ascend = is_ascend
            self._dtype = dtype

        @mnm.model.trace
        def forward(self, data):
            return mnm.argsort(data, axis=self._axis,
                               is_ascend=self._is_ascend, dtype=self._dtype)

    model = Argsort(axis, is_ascend, dtype)
    # forward
    m_x, _ = randn(shape)
    m_func = model._internal(m_x).mod['main']
    m_func = run_infer_type(m_func)
    x_ty = TensorType(shape, dtype=m_x.dtype)
    y_ty = TensorType(shape, dtype=dtype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_func, expected_type)


# pylint: disable=attribute-defined-outside-init, too-many-locals
@pytest.mark.parametrize("shape", [
    (2, 3, 4),
    (1, 4, 6),
    (3, 5, 6),
])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("is_ascend", [True, False])
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_sort(shape, axis, is_ascend, dtype):

    class Sort(mnm.Model):
        def build(self, axis, is_ascend):
            self._axis = axis
            self._is_ascend = is_ascend

        @mnm.model.trace
        def forward(self, data):
            return mnm.sort(data, axis=self._axis,
                            is_ascend=self._is_ascend)

    model = Sort(axis, is_ascend)
    # forward
    m_x, _ = randn(shape, dtype=dtype)
    m_x.requires_grad = True
    record = model._internal(m_x)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    x_ty = TensorType(shape, dtype=m_x.dtype)
    y_ty = TensorType(shape, dtype=m_x.dtype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_mod['main'], expected_type)
    # backward
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    bwd_ty = FuncType([y_ty], x_ty)
    desired_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_mod['main'], desired_type)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("axis", [0, 2, -1])
@pytest.mark.parametrize("ret_type", ["values", "both", "indices"])
@pytest.mark.parametrize("is_ascend", [True, False])
@pytest.mark.parametrize("shape", [(5, 3, 3), (5, 5, 5, 5, 5, 5, 5), (224, 224, 3)])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_topk(shape, k, axis, ret_type, is_ascend, dtype, device):
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=not-callable
    # pylint: disable=no-member
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # pylint: disable=no-self-use
    class Topk(mnm.Model):
        def build(self, k, axis, ret_type, is_ascend, dtype):
            self._axis = axis
            self._k = k
            self._ret_type = ret_type
            self._is_ascend = is_ascend
            self._dtype = dtype

        @mnm.model.trace
        def forward(self, data):
            return mnm.topk(data, k=self._k, axis=self._axis, ret_type=self._ret_type,
                            is_ascend=self._is_ascend, dtype=self._dtype)

    m_x, n_x = randn(shape, device=device, dtype=dtype)
    m_x = mnm.array(n_x, dtype=dtype, device=device)
    m_x.requires_grad = True
    model = Topk(k=k, axis=axis, ret_type=ret_type,
                 is_ascend=is_ascend, dtype=dtype)
    record = model._internal(m_x)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    x_ty = TensorType(shape, dtype=m_x.dtype)
    shape_list = list(shape)
    shape_list[axis] = k
    shape = tuple(shape_list)
    y_a_ty = TensorType(shape, dtype=m_x.dtype)
    y_b_ty = TensorType(shape, dtype=dtype)
    # check forward
    if ret_type == "values":
        expected_type = FuncType([x_ty], y_a_ty)
    elif ret_type == "indices":
        expected_type = FuncType([x_ty], y_b_ty)
    else:
        y_ty = TupleType([y_a_ty, y_b_ty])
        expected_type = FuncType([x_ty], y_ty)
    check_type(m_mod['main'], expected_type)
    # check backward
    if ret_type == "both":
        m_mod = AutoDiff(record.requires_grads)(m_mod)
        m_mod = InferType()(m_mod)
        bwd_ty = FuncType([y_ty], x_ty)
        desired_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
        check_type(m_mod['main'], desired_type)

if __name__ == "__main__":
    pytest.main([__file__])
