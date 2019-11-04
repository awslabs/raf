import mnm._ffi.executor as ffi
from mnm._core.module import Module


class Interpreter(object):

    GLOBAL = None

    def __init__(self, module=None):
        if module is None:
            module = Module.GLOBAL
        self.module = module
        self.runner = ffi.CreateInterpreter(module)

    def __call__(self, expr):
        return self.runner(expr)


Interpreter.GLOBAL = Interpreter(Module.GLOBAL)
