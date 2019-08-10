from ._ffi import _tvm


def register_mnm_node(type_key=None):
    assert isinstance(type_key, str)
    return _tvm._register_node(type_key)


def set_module(module):
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func
    return decorator
