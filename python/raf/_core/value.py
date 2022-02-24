# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-class-docstring,missing-function-docstring
"""Runtime value instances."""
from raf._core.core_utils import dev2str, register_node, str2dev
from raf._ffi import value as ffi
from raf._ffi.ir._make import Constant as make_const_expr
from raf._ffi.value import _make, ToTVM
from raf._lib import Object
from raf._lib import tvm_ndarray
from raf._lib import relay as _relay


@register_node("raf.value.Value")
class Value(Object):
    @staticmethod
    def as_const_expr(value):
        if isinstance(value, Value):
            return make_const_expr(value)
        if isinstance(value, bool):
            return make_const_expr(BoolValue(value))
        if isinstance(value, int):
            return make_const_expr(IntValue(value, "int64"))
        if isinstance(value, float):
            return make_const_expr(FloatValue(value, "float64"))
        if isinstance(value, str):
            return make_const_expr(StringValue(value))
        if isinstance(value, (list, tuple)):
            values = [Value.as_const_expr(v) for v in value]
            return make_const_expr(TupleValue(values))
        raise TypeError("Unsupported input type: {}".format(type(value)))


@register_node("raf.value.BaseTensorValue")
class BaseTensorValue(Value):
    pass


@register_node("raf.value.TensorValue")
class TensorValue(BaseTensorValue):
    # TODO(@junrushao1994): remove property decorators
    @property
    def dltensor_handle(self):
        return self._tensor.handle

    @property
    def data(self):
        handle = self.dltensor_handle
        return handle.contents.data

    @property
    def device(self):
        return dev2str(self.dltensor_handle.contents.device)

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
    def assemble(shape, dtype, device, strides=None, data=None):
        return ffi.AssembleTensorValue(str2dev(device), dtype, shape, strides, data)

    @staticmethod
    def from_tvm(array):
        return ffi.FromTVM(array)

    @staticmethod
    def from_numpy(np_array):
        return TensorValue.from_tvm(tvm_ndarray(np_array))

    def numpy(self):
        return ToTVM(self).numpy()


@register_node("raf.value.TensorTypeValue")
class TensorTypeValue(BaseTensorValue):
    # TODO(@hzfan): add constructors
    pass


@register_node("raf.value.IntValue")
class IntValue(Value):
    def __init__(self, data, dtype="int64"):
        self.__init_handle_by_constructor__(_make.IntValue, dtype, data)
        assert isinstance(data, int)


@register_node("raf.value.FloatValue")
class FloatValue(Value):
    def __init__(self, data, dtype="float32"):
        self.__init_handle_by_constructor__(_make.FloatValue, dtype, data)
        assert isinstance(data, float)


@register_node("raf.value.BoolValue")
class BoolValue(Value):
    def __init__(self, data):
        self.__init_handle_by_constructor__(_make.BoolValue, data)
        assert isinstance(data, bool)


@register_node("raf.value.StringValue")
class StringValue(Value):
    def __init__(self, data):
        self.__init_handle_by_constructor__(_make.StringValue, data)
        assert isinstance(data, str)


@register_node("raf.value.TupleValue")
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


@register_node("raf.value.ClosureValue")
class ClosureValue(Value):
    def __init__(self, env, func, bind=None):
        assert isinstance(env, dict)
        assert isinstance(func, _relay.Function)
        for (key, value) in env.items():
            assert isinstance(key, _relay.Var)
            assert isinstance(value, Value)
        self.__init_handle_by_constructor__(_make.ClosureValue, env, func, bind)


@register_node("raf.value.NoGradValue")
class NoGradValue(Value):
    def __init__(self):
        self.__init_handle_by_constructor__(_make.NoGradValue)
