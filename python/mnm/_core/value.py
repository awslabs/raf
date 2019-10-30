from .._ffi._tvm import _NodeBase, tvm
from .._ffi.value import AssembleTensorValue, DeTuple, FromTVM, _make
from .base import register_mnm_node
from .context import Context


@register_mnm_node("mnm.value.Value")
class Value(_NodeBase):
    pass


@register_mnm_node("mnm.value.TensorValue")
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
        handle = self.dltensor_handle
        ctx = handle.contents.ctx

        return Context.create(ctx.device_type, ctx.device_id)

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
        if isinstance(ctx, str):
            ctx = Context(ctx)
        assert isinstance(ctx, Context), type(ctx)

        return AssembleTensorValue(ctx, dtype, shape, strides, data)

    @staticmethod
    def from_tvm(tvm_array):
        return FromTVM(tvm_array)

    @staticmethod
    def from_numpy(np_array):
        return TensorValue.from_tvm(tvm.ndarray.array(np_array))


@register_mnm_node("mnm.value.IntValue")
class IntValue(Value):

    def __init__(self, data):
        assert isinstance(data, int)
        self.__init_handle_by_constructor__(_make.IntValue, data)


@register_mnm_node("mnm.value.FloatValue")
class FloatValue(Value):

    def __init__(self, data):
        assert isinstance(data, float)
        self.__init_handle_by_constructor__(_make.FloatValue, data)


@register_mnm_node("mnm.value.BoolValue")
class BoolValue(Value):

    def __init__(self, data):
        assert isinstance(data, bool)
        self.__init_handle_by_constructor__(_make.BoolValue, data)


@register_mnm_node("mnm.value.StringValue")
class StringValue(Value):

    def __init__(self, data):
        assert isinstance(data, str)
        self.__init_handle_by_constructor__(_make.StringValue, data)


@register_mnm_node("mnm.value.TupleValue")
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
        return DeTuple(self)
