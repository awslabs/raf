from ._ffi._tvm import _NodeBase
from ._ffi import _ndarray
from .base import register_mnm_node


class ndarray(_NodeBase):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()
