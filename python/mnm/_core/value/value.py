from ..base import register_mnm_node
from ..._ffi._tvm import _NodeBase


@register_mnm_node("mnm.value.Value")
class Value(_NodeBase):
    pass
