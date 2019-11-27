import ctypes
import weakref

from mnm._core.core_utils import ctx2str, set_module, str2ctx
from mnm._core.value import TensorValue
from mnm._ffi.tensor import MarkNumpy
from mnm._ffi.value import BindValue, LookupBoundValue, ToTVM
from mnm._lib import _DLManagedTensor, _register_func, relay, tvm_array


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
            self.__handle = BindValue(_np_to_tensor_value(npa, ctx=ctx), name)

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
        return self.__dtype

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


@set_module("mnm")
class Parameter(ndarray):
    def __init__(
            self,
            shape,
            dtype=float,
            *,  # pylint: disable=too-many-arguments
            buffer=None,
            offset=0,
            strides=None,
            order=None,
            ctx=None,
            requires_grad=True,
            name=""):
        super(Parameter, self).__init__(shape=shape,
                                        dtype=dtype,
                                        buffer=buffer,
                                        offset=offset,
                                        strides=strides,
                                        order=order,
                                        ctx=ctx,
                                        name=name)
        self.__handle = BindValue(self._ndarray__value, name)  # pylint: disable=no-member
        self.__requires_grad = False
        self.__parents = weakref.WeakSet()
        self.requires_grad = requires_grad

    @property
    def requires_grad(self):
        return self.__requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        if not isinstance(value, bool):
            raise ValueError("Parameter's requires_grad should be boolean")
        if value == self.__requires_grad:
            return
        self.__requires_grad = value
        self._ndarray__handle, self.__handle = self.__handle, self._ndarray__handle
        for parent in list(self.__parents.data):
            parent = parent()
            if parent is None:
                continue
            parent._Model__invalidate_cache()  # pylint: disable=protected-access


class Symbol:  # pylint: disable=too-few-public-methods

    __slots__ = ["__expr"]

    def __init__(self):
        self.__expr = None

    @staticmethod
    def from_expr(expr):
        ret = Symbol()
        ret.__expr = expr  # pylint: disable=protected-access

        return ret

    @staticmethod
    def make_var(name_hint=""):
        ret = Symbol()
        ret.__expr = relay.Var(name_hint=name_hint)  # pylint: disable=protected-access

        return ret

    def __getitem__(self, item):
        if isinstance(item, int):
            ret = Symbol()
            ret.__expr = relay.TupleGetItem(self.__expr, item)  # pylint: disable=protected-access

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

    return TensorValue.from_tvm(tvm_array(npa, ctx=str2ctx(ctx)))


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
    return ndarray(BindValue(_np_to_tensor_value(npa, ctx=ctx), name))


_DL_MANAGED_TENSOR_PTR = ctypes.POINTER(_DLManagedTensor)


@_register_func("mnm._numpy_array_deleter")
def _np_del(handle):
    handle = ctypes.cast(handle, _DL_MANAGED_TENSOR_PTR)
    void_p = handle.contents.manager_ctx
    pyobj = ctypes.cast(void_p, ctypes.py_object)
    ctypes.pythonapi.Py_DecRef(pyobj)
