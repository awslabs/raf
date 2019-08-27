from ..._ffi.value import AssembleTensorValue, FromTVM
from ..base import register_mnm_node
from ..context import Context
from .value import Value
from mnm._ffi._tvm import tvm


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
