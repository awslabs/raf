from ..._ffi.value import DeTuple, _make
from ..base import register_mnm_node
from .value import Value


@register_mnm_node("mnm.value.TupleValue")
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
        return DeTuple(self)
