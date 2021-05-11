#pylint: disable=invalid-name, protected-access
"""Scope with certain backends enabled"""
import threading
import tvm

class Backend:
    """Backend scope to specify a list of preferred backends

    Parameters
    ----------
    backends : List[str]
        The list of preferred backends, with descending priority
    """
    valid_backends = ["tvmjit", "generated_cudnn", "cutlass", "cublas"]
    storage = threading.local()
    storage.scope = []

    def __init__(self, backends):
        super(Backend, self).__init__()
        if not set(Backend.valid_backends).issuperset(set(backends)):
            raise ValueError(
                f"{set(backends).difference(set(Backend.valid_backends))} are not valid backends")
        self.backends = backends

    def __enter__(self):
        Backend.storage.scope.append(self)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        Backend.storage.scope = Backend.storage.scope[:-1]


@tvm._ffi.register_func("backend.preferred_backends")
def preferred_backends():
    """Get a list of preferred backends, with descending priority"""
    if not Backend.storage.scope:
        return None
    return Backend.storage.scope[-1].backends
