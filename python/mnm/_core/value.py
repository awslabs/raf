from mnm._core.core_utils import ctx2str, register_node, str2ctx
from mnm._ffi import value as ffi
from mnm._ffi.ir._make import Constant as make_const_expr
from mnm._ffi.value import _make
from mnm._lib import _NodeBase as NodeBase


@register_node("mnm.value.Value")
class Value(NodeBase):
    pass

    @staticmethod
    def as_const_expr(value):
        if isinstance(value, Value):
            return make_const_expr(value)

        if isinstance(value, int):
            return make_const_expr(IntValue(value))

        if isinstance(value, float):
            return make_const_expr(FloatValue(value))

        if isinstance(value, bool):
            return make_const_expr(BoolValue(value))
        raise NotImplementedError


@register_node("mnm.value.TensorValue")
class TensorValue(Value):

    @property
    def dltensor_handle(self):
        return self._tensor.handle

    @property
    def data(self):
        handle = self.dltensor_handle

        return handle.contents.data

    @property
    def ctx(self):
        return ctx2str(self.dltensor_handle.contents.ctx)

    @property
    def ndim(self):
        handle = self.dltensor_handle

        return handle.contents.ndim

    @property
    def dtype(self):
        handle = self.dltensor_handle

        return str(handle.contents.dtype)

    @property
    def shape(self):
        handle = self.dltensor_handle
        ndim = self.ndim

        return tuple(handle.contents.shape[i] for i in range(ndim))

    @property
    def strides(self):
        handle = self.dltensor_handle
        ndim = self.ndim

        return tuple(handle.contents.strides[i] for i in range(ndim))

    @property
    def byte_offset(self):
        handle = self.dltensor_handle

        return handle.contents.byte_offset

    @staticmethod
    def assemble(shape, dtype, ctx, strides=None, data=None):
        return ffi.AssembleTensorValue(str2ctx(ctx), dtype, shape, strides, data)

    @staticmethod
    def from_tvm(tvm_array):
        return ffi.FromTVM(tvm_array)

    @staticmethod
    def from_numpy(np_array):
        from mnm._lib import tvm

        return TensorValue.from_tvm(tvm.ndarray.array(np_array))


@register_node("mnm.value.IntValue")
class IntValue(Value):

    def __init__(self, data):
        assert isinstance(data, int)
        self.__init_handle_by_constructor__(_make.IntValue, data)


@register_node("mnm.value.FloatValue")
class FloatValue(Value):

    def __init__(self, data):
        assert isinstance(data, float)
        self.__init_handle_by_constructor__(_make.FloatValue, data)


@register_node("mnm.value.BoolValue")
class BoolValue(Value):

    def __init__(self, data):
        assert isinstance(data, bool)
        self.__init_handle_by_constructor__(_make.BoolValue, data)


@register_node("mnm.value.StringValue")
class StringValue(Value):

    def __init__(self, data):
        assert isinstance(data, str)
        self.__init_handle_by_constructor__(_make.StringValue, data)


@register_node("mnm.value.TupleValue")
class TupleValue(Value):

    def __init__(self, values):
        if isinstance(values, list):
            values = tuple(values)
        assert isinstance(values, tuple)

        for value in values:
            assert isinstance(value, Value)
        self.__init_handle_by_constructor__(_make.TupleValue, values)

    def __getitem__(self, index: int):
        return self._de_tuple[index]

    def __len__(self):
        return len(self._de_tuple)

    @property
    def _de_tuple(self):
        return ffi.DeTuple(self)


@register_node("mnm.value.BoundExpr")
class BoundExpr(NodeBase):

    def __init__(self, expr, value, executor=None):
        self.__init_handle_by_constructor__(
            _make.BoundExpr, expr, value, executor)
