import ctypes
import weakref

from mnm._core.core_utils import ctx2str, set_module, str2ctx
from mnm._core.value import BoundExpr, TensorValue, Value
from mnm._ffi.ir._make import Constant as MakeConstExpr
from mnm._ffi.tensor import MarkNumpy
from mnm._ffi.value import ToTVM
from mnm._lib import _DLManagedTensor, _register_func, relay, tvm_array


@set_module("mnm")  # pylint: disable=invalid-name
class ndarray:

    def __init__(self, shape, dtype=float, *, # pylint: disable=too-many-arguments
                 buffer=None, offset=0,
                 strides=None, order=None, ctx=None):
        arg_0 = shape
        arg_1 = dtype

        if isinstance(arg_0, BoundExpr):
            self.__handle = arg_0
        elif isinstance(arg_0, ndarray):
            self.__handle = arg_0.__handle  # pylint: disable=protected-access
        elif isinstance(arg_0, Value) and isinstance(arg_1, relay.Expr):
            self.__handle = BoundExpr(value=arg_0, expr=arg_1)
        elif isinstance(arg_0, relay.Expr) and isinstance(arg_1, Value):
            self.__handle = BoundExpr(expr=arg_0, value=arg_1)
        else:
            import numpy as np  # pylint: disable=import-outside-toplevel

            if isinstance(arg_0, np.ndarray):
                npa = arg_0
            else:
                npa = np.ndarray(shape=shape, dtype=dtype, buffer=buffer,
                                 offset=offset, strides=strides, order=order)
            value = np_to_tensor_value(npa, ctx=ctx)
            self.__handle = BoundExpr(value=value, expr=MakeConstExpr(value))

    def __str__(self):
        fmt = "{}\n<NDArray [{}] @ {}, dtype={}>"

        return fmt.format(str(self.asnumpy()),
                          " x ".join(map(str, self.shape)),
                          self.ctx,
                          self.dtype)

    @property
    def __tensor(self):
        return self.__handle._value._tensor  # pylint: disable=protected-access

    def asnumpy(self):
        return ToTVM(self.__handle._value).asnumpy()  # pylint: disable=protected-access

    @property
    def ctx(self):
        return ctx2str(self.__tensor.handle.contents.ctx)

    @property
    def ndim(self):
        return int(self.__tensor.handle.contents.ndim)

    @property
    def dtype(self):
        return str(self.__tensor.handle.contents.dtype)

    @property
    def shape(self):
        shape_handle = self.__tensor.handle.contents.shape

        return tuple(shape_handle[i] for i in range(self.ndim))

    @property
    def strides(self):
        # TODO(@junrushao1994): check if they are in `numel` or `bytes`
        strides_handle = self.__tensor.handle.contents.strides_handle

        return tuple(strides_handle[i] for i in range(self.ndim))


@set_module("mnm")
class Parameter(ndarray):

    def __init__(self, shape, dtype=float, *, # pylint: disable=too-many-arguments
                 buffer=None, offset=0,
                 strides=None, order=None, ctx=None, requires_grad=True, name=""):
        super(Parameter, self).__init__(shape=shape, dtype=dtype, buffer=buffer,
                                        offset=offset, strides=strides, order=order,
                                        ctx=ctx)
        self.__handle = BoundExpr(expr=relay.Var(name_hint=name),
                                  value=self._ndarray__handle._value)  # pylint: disable=no-member
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

        if value != self.__requires_grad:
            self.__requires_grad = value
            self._ndarray__handle, self.__handle = self.__handle, self._ndarray__handle

            for parent in list(self.__parents.data):
                parent = parent()

                if parent is None:
                    continue
                parent._Model__invalidate_cache()  # pylint: disable=protected-access

    @requires_grad.deleter
    def requires_grad(self):
        raise NotImplementedError("Cannot delete the requires_grad from a parameter")


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
        raise NotImplementedError("Only constant integers are supported for now.")


def np_to_tensor_value(npa, ctx=None):

    def _tensor_value(obj):
        ctx = "cpu"
        dtype = str(obj.dtype)
        shape = list(obj.shape)
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
        MarkNumpy(result._tensor, _manager_ctx(npa))  # pylint: disable=protected-access

        return result

    return TensorValue.from_tvm(tvm_array(npa, ctx=str2ctx(ctx)))


@set_module("mnm")
def array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, ctx=None):  # pylint: disable=too-many-arguments,redefined-builtin
    import numpy as np  # pylint: disable=import-outside-toplevel
    npa = np.array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)
    value = np_to_tensor_value(npa, ctx=ctx)

    return ndarray(value, MakeConstExpr(value))


_DL_MANAGED_TENSOR_PTR = ctypes.POINTER(_DLManagedTensor)
@_register_func("mnm._numpy_array_deleter")
def _np_del(handle):
    handle = ctypes.cast(handle, _DL_MANAGED_TENSOR_PTR)
    void_p = handle.contents.manager_ctx
    pyobj = ctypes.cast(void_p, ctypes.py_object)
    ctypes.pythonapi.Py_DecRef(pyobj)
