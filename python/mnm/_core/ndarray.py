import ctypes

from mnm._core.core_utils import ctx2str, set_module, str2ctx
from mnm._core.value import TensorValue
from mnm._ffi.binding import (BindNDArray, BindSymbol, LookupBoundValue,
                              SetRequiresGrad, Backward, LookupGrad)
from mnm._ffi.tensor import MarkNumpy
from mnm._ffi.value import ToTVM
from mnm._lib import _DLManagedTensor, _register_func, relay, tvm_ndarray


@set_module("mnm")  # pylint: disable=invalid-name,too-many-instance-attributes
class ndarray:
    def __init__(  # pylint: disable=too-many-arguments
            self,
            shape,
            dtype=float,
            *,
            buffer=None,
            offset=0,
            strides=None,
            order=None,
            ctx=None,
            name=""):
        arg_0 = shape
        if isinstance(arg_0, relay.Var):
            self.__handle = arg_0
        elif isinstance(arg_0, ndarray):
            self.__handle = arg_0.__handle  # pylint: disable=protected-access
        else:
            import numpy as np  # pylint: disable=import-outside-toplevel
            if isinstance(arg_0, np.ndarray):
                npa = arg_0
            else:
                npa = np.ndarray(shape=shape,
                                 dtype=dtype,
                                 buffer=buffer,
                                 offset=offset,
                                 strides=strides,
                                 order=order)
            # NDArray is treated as relay.Constant
            self.__handle = BindNDArray(
                _np_to_tensor_value(npa, ctx=ctx), None, name)
        self.__requires_grad = False

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            if key == slice(None, None, None):
                import numpy as np  # pylint: disable=import-outside-toplevel
                assert isinstance(value, np.ndarray)
                assert value.shape == self.shape
                value = _np_to_tensor_value(value.astype(self.dtype),
                                            ctx=self.ctx)
                self.__handle = BindNDArray(
                    value, None, self.__handle.name_hint)
                return
        raise NotImplementedError

    @property
    def requires_grad(self):
        return self.__requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        assert isinstance(value, bool)
        self.__requires_grad = value
        SetRequiresGrad(self.__handle, value)

    @property
    def __handle(self):
        return self.__handle_

    @__handle.setter
    def __handle(self, handle):
        value = LookupBoundValue(handle)
        self.__handle_ = handle
        self.__value = value

    @property
    def __value(self):
        return self.__value_

    @__value.setter
    def __value(self, value):
        assert isinstance(value, TensorValue)
        self.__value_ = value
        # Figure out the underlying DLTensor
        dltensor = value._tensor.handle.contents  # pylint: disable=protected-access
        ndim = dltensor.ndim
        shape_handle = dltensor.shape
        strides_handle = dltensor.strides
        # Refresh cached values
        self.ctx = ctx2str(dltensor.ctx)
        self.dtype = dltensor.dtype
        self.ndim = ndim
        self.shape = tuple(shape_handle[i] for i in range(ndim))
        self.strides = tuple(strides_handle[i] for i in range(ndim))
        self.byte_offset = dltensor.byte_offset

    def __str__(self):
        fmt = "{}\n<NDArray [{}] @ {}, dtype={}>"
        shape = " x ".join(map(str, self.shape))
        npa = ToTVM(self.__value).asnumpy()
        return fmt.format(str(npa), shape, self.ctx, self.dtype)

    def __repr__(self):
        return str(self)

    def asnumpy(self):
        return ToTVM(self.__value).asnumpy()  # pylint: disable=protected-access

    @property
    def ctx(self):
        return self.__ctx

    @ctx.setter
    def ctx(self, ctx):
        self.__ctx = ctx

    @property
    def ndim(self):
        return self.__ndim

    @ndim.setter
    def ndim(self, ndim):
        self.__ndim = ndim

    @property
    def dtype(self):
        return str(self.__dtype)

    @dtype.setter
    def dtype(self, dtype):
        self.__dtype = dtype

    @property
    def shape(self):
        return self.__shape

    @shape.setter
    def shape(self, shape):
        self.__shape = shape

    @property
    def strides(self):
        return self.__strides

    @strides.setter
    def strides(self, strides):
        self.__strides = strides

    @property
    def byte_offset(self):
        return self.__byte_offset

    @byte_offset.setter
    def byte_offset(self, byte_offset):
        self.__byte_offset = byte_offset

    def to(self, *, ctx=None, dtype=None):  # pylint: disable=invalid-name
        npa = self.asnumpy()
        if dtype is not None:
            npa = npa.astype(dtype)
        if ctx is None:
            ctx = self.ctx
        return ndarray(BindNDArray(_np_to_tensor_value(npa, ctx=ctx), None, ""))

    def backward(self, gradient=None):
        if gradient is not None:
            assert isinstance(gradient, ndarray)
            gradient = gradient._ndarray__handle  # pylint: disable=protected-access
        Backward(self.__handle, gradient)

    @property
    def grad(self):
        if not self.requires_grad:
            raise ValueError("Cannot run backward() for NDArrays whose require_grad = False")
        return ndarray(LookupGrad(self.__handle))


class Symbol:  # pylint: disable=too-few-public-methods

    __slots__ = ["__handle"]

    def __init__(self):
        self.__handle = None

    @staticmethod
    def from_expr(expr):
        assert isinstance(expr, relay.Var)
        ret = Symbol()
        ret.__handle = expr  # pylint: disable=protected-access
        return ret

    @staticmethod
    def make_var(name_hint=""):
        ret = Symbol()
        ret.__handle = BindSymbol(None, name_hint)  # pylint: disable=protected-access
        return ret

    @staticmethod
    def make_tuple(symbols, name_hint=""):
        expr = relay.Tuple(symbols)
        ret = Symbol()
        ret.__handle = BindSymbol(expr, name_hint)  # pylint: disable=protected-access
        return ret

    def __getitem__(self, item, name_hint=""):
        if isinstance(item, int):
            expr = relay.TupleGetItem(self.__handle, item)
            ret = Symbol()
            ret.__handle = BindSymbol(expr, name_hint)  # pylint: disable=protected-access
            return ret
        raise NotImplementedError(
            "Only constant integers are supported for now.")


def _np_to_tensor_value(npa, ctx=None):
    def _tensor_value(obj):
        ctx = "cpu"
        dtype = str(obj.dtype)
        shape = list(obj.shape)
        strides = [x // obj.itemsize for x in obj.strides]
        data = obj.ctypes.data_as(ctypes.c_void_p)
        assert len(shape) == len(strides)
        return TensorValue.assemble(ctx=ctx,
                                    dtype=dtype,
                                    shape=shape,
                                    strides=strides,
                                    data=data)

    def _manager_ctx(obj):
        pyobj = ctypes.py_object(obj)
        void_p = ctypes.c_void_p.from_buffer(pyobj)
        ctypes.pythonapi.Py_IncRef(pyobj)
        return void_p

    if ctx is None:
        result = _tensor_value(npa)
        MarkNumpy(result._tensor, _manager_ctx(npa))  # pylint: disable=protected-access
        return result

    return TensorValue.from_tvm(tvm_ndarray(npa, ctx=str2ctx(ctx)))


@set_module("mnm")
def array(
        object,  # pylint: disable=too-many-arguments,redefined-builtin
        dtype=None,
        *,
        copy=True,
        order='K',
        subok=False,
        ndmin=0,
        ctx=None,
        name=""):
    import numpy as np  # pylint: disable=import-outside-toplevel
    npa = np.array(object,
                   dtype=dtype,
                   copy=copy,
                   order=order,
                   subok=subok,
                   ndmin=ndmin)
    return ndarray(BindNDArray(_np_to_tensor_value(npa, ctx=ctx), None, name))


_DL_MANAGED_TENSOR_PTR = ctypes.POINTER(_DLManagedTensor)


@_register_func("mnm._numpy_array_deleter")
def _np_del(handle):
    handle = ctypes.cast(handle, _DL_MANAGED_TENSOR_PTR)
    void_p = handle.contents.manager_ctx
    pyobj = ctypes.cast(void_p, ctypes.py_object)
    ctypes.pythonapi.Py_DecRef(pyobj)


@_register_func("mnm._ndarray_to_string")
def _print(var):
    return str(ndarray(var))


def get_ndarray_handle(a):
    return a._ndarray__handle  # pylint: disable=protected-access


def get_symbol_handle(a):
    return a._Symbol__handle  # pylint: disable=protected-access
