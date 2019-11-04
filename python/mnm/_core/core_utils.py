from mnm._lib import _NodeBase as NodeBase
from mnm._lib import _register_node


def register_node(type_key=None):
    assert isinstance(type_key, str)

    return _register_node(type_key)


def set_module(module):
    def decorator(func):
        if module is not None:
            func.__module__ = module

        return func

    return decorator
