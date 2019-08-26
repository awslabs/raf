from ..._ffi.value import _make
from ..base import register_mnm_node
from .value import Value


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
