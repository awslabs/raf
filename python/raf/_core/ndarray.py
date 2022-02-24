# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-class-docstring,missing-function-docstring
"""Multi-dimension array representation"""
import ctypes

from raf._core.core_utils import dev2str, set_module, str2dev
from raf._core.value import TensorValue
from raf._ffi.binding import (
    BindNDArray,
    BindSymbol,
    RebindNDArray,
    LookupBoundValue,
    SetRequiresGrad,
    Backward,
    LookupGrad,
)
from raf._ffi.tensor import MarkNumpy
from raf._ffi.value import ToTVM
from raf._lib import _register_func, relay, tvm_ndarray
from raf._lib import TensorContainer as _DLManagedTensor


@set_module("raf")
class ndarray:  # pylint: disable=invalid-name,too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self,
        shape,
        dtype=float,
        *,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        device="cpu",
        name=""
    ):
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
                npa = np.ndarray(
                    shape=shape,
                    dtype=dtype,
                    buffer=buffer,
                    offset=offset,
                    strides=strides,
                    order=order,
                )
            # NDArray is treated as relay.Constant
            self.__handle = BindNDArray(_np_to_tensor_value(npa, device=device), None, name)
        self.__requires_grad = False

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            if key == slice(None, None, None):
                import numpy as np  # pylint: disable=import-outside-toplevel

                assert isinstance(value, np.ndarray)
                assert value.shape == self.shape
                value = _np_to_tensor_value(value.astype(self.dtype), device=self.device)
                self.__handle = BindNDArray(value, None, self.__handle.name_hint)
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
        self.device = dev2str(dltensor.device)
        self.dtype = dltensor.dtype
        self.ndim = ndim
        self.shape = tuple(shape_handle[i] for i in range(ndim))
        self.strides = tuple(strides_handle[i] for i in range(ndim))
        self.byte_offset = dltensor.byte_offset

    def __str__(self):
        fmt = "{}\n<NDArray [{}] @ {}, dtype={}>"
        shape = " x ".join(map(str, self.shape))
        npa = ToTVM(self.__value).numpy()
        return fmt.format(str(npa), shape, self.device, self.dtype)

    def __repr__(self):
        return str(self)

    def numpy(self):
        return ToTVM(self.__value).numpy()  # pylint: disable=protected-access

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, device):
        self.__device = device

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

    def to(self, *, device=None, dtype=None):  # pylint: disable=invalid-name
        npa = self.numpy()
        if dtype is not None:
            npa = npa.astype(dtype)
        if device is None:
            device = self.device
        ret = ndarray(BindNDArray(_np_to_tensor_value(npa, device=device), None, ""))
        ret.requires_grad = self.requires_grad
        return ret

    def backward(self, gradient=None):
        if gradient is not None:
            assert isinstance(gradient, ndarray)
            gradient = gradient._ndarray__handle  # pylint: disable=protected-access
        Backward(self.__handle, gradient)

    def update(self, value):
        RebindNDArray(self.__handle, value.__value, None)  # pylint: disable=protected-access
        # call custom setters to update value
        self.__handle = self.__handle
        self.requires_grad = self.requires_grad

    def __sub__(self, other):
        """x.__sub__(y) <=> x - y"""
        from raf._op.imp import subtract  # pylint: disable=import-outside-toplevel,cyclic-import

        return subtract(self, other)

    def __add__(self, other):
        """x.__add__(y) <=> x + y"""
        from raf._op.imp import add  # pylint: disable=import-outside-toplevel,cyclic-import

        return add(self, other)

    def __mul__(self, other):
        """x.__mul__(y) <=> x * y"""
        from raf._op.imp import multiply  # pylint: disable=import-outside-toplevel,cyclic-import

        return multiply(self, other)

    def __div__(self, other):
        """x.__div__(y) <=> x / y"""
        from raf._op.imp import divide  # pylint: disable=import-outside-toplevel,cyclic-import

        return divide(self, other)

    def __truediv__(self, other):
        """x.__div__(y) <=> x / y"""
        from raf._op.imp import divide  # pylint: disable=import-outside-toplevel,cyclic-import

        return divide(self, other)

    @property
    def grad(self):
        if not self.requires_grad:
            raise ValueError("Cannot run backward() for NDArrays whose require_grad = False")
        grad_var = LookupGrad(self.__handle)
        grad_val = LookupBoundValue(grad_var)
        if not isinstance(grad_val, TensorValue):
            return None
        return ndarray(grad_var)

    @staticmethod
    def from_tensor_value(value):
        assert isinstance(value, TensorValue)
        ret = ndarray(BindNDArray(value, None, ""))
        return ret


class Symbol:
    # pylint: disable=too-few-public-methods, protected-access
    __slots__ = ["__handle"]

    def __init__(self):
        self.__handle = None

    @staticmethod
    def from_expr(expr):
        assert isinstance(expr, relay.Var)
        ret = Symbol()
        ret.__handle = expr
        return ret

    @staticmethod
    def make_var(name_hint="", type_annotation=None):
        """
        Create a symbol

        Parameters
        ----------
        name_hint: string
            The name hint

        type_annotation: relay.Type
            The type annotation for the variable

        Returns:
            ret: Symbol
                 The Symbol with name_hint and type_annotation
        """
        ret = Symbol()
        ret.__handle = BindSymbol(None, name_hint, type_annotation)
        return ret

    @staticmethod
    def make_tuple(symbols, name_hint=""):
        def as_relay_expr(symbol):
            if isinstance(symbol, Symbol):
                return symbol.__handle
            if isinstance(symbol, relay.Expr):
                return symbol
            raise TypeError("Cannot convert to relay expr")

        symbol_handles = [as_relay_expr(symbol) for symbol in symbols]
        expr = relay.Tuple(symbol_handles)
        return Symbol.from_expr(BindSymbol(expr, name_hint, None))

    def __getitem__(self, item, name_hint=""):
        if isinstance(item, int):
            expr = relay.TupleGetItem(self.__handle, item)
            ret = Symbol()
            ret.__handle = BindSymbol(expr, name_hint, None)
            return ret
        raise NotImplementedError("Only constant integers are supported for now.")

    def __sub__(self, other):
        """x.__sub__(y) <=> x - y"""
        from raf._op.sym import subtract  # pylint: disable=import-outside-toplevel,cyclic-import

        return subtract(self, other)

    def __add__(self, other):
        """x.__add__(y) <=> x + y"""
        from raf._op.sym import add  # pylint: disable=import-outside-toplevel,cyclic-import

        return add(self, other)

    def __mul__(self, other):
        """x.__mul__(y) <=> x * y"""
        from raf._op.sym import multiply  # pylint: disable=import-outside-toplevel,cyclic-import

        return multiply(self, other)

    def __div__(self, other):
        """x.__div__(y) <=> x / y"""
        from raf._op.sym import divide  # pylint: disable=import-outside-toplevel,cyclic-import

        return divide(self, other)

    def __truediv__(self, other):
        """x.__div__(y) <=> x / y"""
        from raf._op.sym import divide  # pylint: disable=import-outside-toplevel,cyclic-import

        return divide(self, other)


def _np_to_tensor_value(npa, device="cpu"):
    def _tensor_value(obj):
        device = "cpu"
        dtype = str(obj.dtype)
        shape = list(obj.shape)
        strides = [x // obj.itemsize for x in obj.strides]
        data = obj.ctypes.data_as(ctypes.c_void_p)
        assert len(shape) == len(strides)
        return TensorValue.assemble(
            device=device, dtype=dtype, shape=shape, strides=strides, data=data
        )

    def _manager_ctx(obj):
        pyobj = ctypes.py_object(obj)
        void_p = ctypes.c_void_p.from_buffer(pyobj)
        ctypes.pythonapi.Py_IncRef(pyobj)
        return void_p

    if device is None:
        result = _tensor_value(npa)
        MarkNumpy(result._tensor, _manager_ctx(npa))  # pylint: disable=protected-access
        return result

    return TensorValue.from_tvm(tvm_ndarray(npa, device=str2dev(device)))


@set_module("raf")
def array(
    object,  # pylint: disable=too-many-arguments,redefined-builtin
    dtype=None,
    *,
    copy=True,
    order="K",
    subok=False,
    ndmin=0,
    device="cpu",
    name=""
):
    import numpy as np  # pylint: disable=import-outside-toplevel

    npa = np.array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)
    return ndarray(BindNDArray(_np_to_tensor_value(npa, device=device), None, name))


_DL_MANAGED_TENSOR_PTR = ctypes.POINTER(_DLManagedTensor)


@_register_func("raf._numpy_array_deleter")
def _np_del(handle):
    handle = ctypes.cast(handle, _DL_MANAGED_TENSOR_PTR)
    void_p = handle.contents.manager_ctx
    pyobj = ctypes.cast(void_p, ctypes.py_object)
    ctypes.pythonapi.Py_DecRef(pyobj)


@_register_func("raf._ndarray_to_string")
def _print(var):
    return str(ndarray(var))


def get_ndarray_handle(a):
    return a._ndarray__handle  # pylint: disable=protected-access


def get_symbol_handle(a):
    return a._Symbol__handle  # pylint: disable=protected-access
