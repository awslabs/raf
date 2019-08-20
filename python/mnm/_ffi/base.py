from ._tvm import _register_node


def register_mnm_node(type_key=None):
    assert isinstance(type_key, str)
    return _register_node(type_key)
