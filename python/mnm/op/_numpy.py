import ctypes

import mnm._ffi.tensor as ffi
from mnm._core.context import Context
from mnm._core.value import TensorValue
from mnm._lib import _DLManagedTensor, _register_func, tvm

_DL_MANAGED_TENSOR_PTR = ctypes.POINTER(_DLManagedTensor)


@_register_func("mnm._numpy_array_deleter")
def _np_del(handle):
    handle = ctypes.cast(handle, _DL_MANAGED_TENSOR_PTR)
    void_p = handle.contents.manager_ctx
    pyobj = ctypes.cast(void_p, ctypes.py_object)
    ctypes.pythonapi.Py_DecRef(pyobj)


def np_to_tensor_value(npa, ctx=None):

    def _tensor_value(obj):
        ctx = Context("cpu")
        dtype = str(obj.dtype)
        shape = [x for x in obj.shape]
        strides = [x // obj.itemsize for x in obj.strides]
        data = obj.ctypes.data_as(ctypes.c_void_p)
        assert len(shape) == len(strides)

        return TensorValue.assemble(ctx=ctx, dtype=dtype, shape=shape, strides=strides, data=data)

    def _manager_ctx(obj):
        pyobj = ctypes.py_object(obj)
        void_p = ctypes.c_void_p.from_buffer(pyobj)
        ctypes.pythonapi.Py_IncRef(pyobj)

        return void_p

    if ctx is None:
        result = _tensor_value(npa)
        ffi.MarkNumpy(result._tensor, _manager_ctx(npa))

        return result

    ctx = Context(ctx)
    tvm_array = tvm.ndarray.array(npa, ctx=ctx)

    return TensorValue.from_tvm(tvm_array)
