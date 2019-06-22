from . import _ffi
from . import context
from ._ffi._tvm import _NodeBase
from .base import register_mnm_node


@register_mnm_node("mnm.value.Value")
class Value(_NodeBase):
    pass


@register_mnm_node("mnm.value.TensorValue")
class TensorValue(Value):

    @property
    def dltensor_handle(self):
        return self.tensor.handle

    @property
    def data(self):
        handle = self.dltensor_handle
        return handle.contents.data

    @property
    def ctx(self):
        handle = self.dltensor_handle
        ctx = handle.contents.ctx
        return context.Context(ctx.device_type, ctx.device_id)

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
        return _ffi.value.AssembleTensorValue(ctx, dtype, shape, strides, data)
