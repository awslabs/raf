import numpy as np

import mnm._ffi.ir._make as _make
import mnm._ffi.ir.module as ffi
from mnm._core.core_utils import NodeBase, register_node

from .value import BoolValue, FloatValue, IntValue, TensorValue, Value

# TODO(@junrushao1994): move ConstantExpr to value.py
VALUE_MAKER = {
    int: IntValue,
    float: FloatValue,
    bool: BoolValue,
    np.ndarray: TensorValue.from_numpy
}


def ConstantExpr(value):
    if isinstance(value, Value):
        return _make.Constant(value)
    maker = VALUE_MAKER.get(type(value), None)

    if maker is not None:
        return _make.Constant(maker(value))
    raise NotImplementedError


@register_node("mnm.ir.Module")
class Module(NodeBase):

    GLOBAL = None

    def __init__(self, functions=None):
        if functions is None:
            functions = {}
        self.__init_handle_by_constructor__(_make.Module, functions)

    def __setitem__(self, var, func):
        ffi.Add(self, var, func)

    def __getitem__(self, var):
        return ffi.Lookup(self, var)


Module.GLOBAL = Module()
