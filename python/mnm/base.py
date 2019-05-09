from ._ffi import _tvm


def register_mnm_node(type_key=None):
    result = None
    if isinstance(type_key, type):
        result = _tvm._register_node(type_key.__name__)(type_key)
    elif isinstance(type_key, str):
        result = _tvm._register_node(type_key)
    else:
        raise ValueError("Unsupported type of type_key: ",
                         type(type_key).__name__)
    return result
