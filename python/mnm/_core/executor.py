from .._ffi.executor import CreateInterpreter
from .ir import Module


class Interpreter(object):

    GLOBAL = None

    def __init__(self, module=None):
        if module is None:
            module = Module.GLOBAL
        self.module = module
        self.runner = CreateInterpreter(module)

    def __call__(self, expr):
        return self.runner(expr)


Interpreter.GLOBAL = Interpreter(Module.GLOBAL)
