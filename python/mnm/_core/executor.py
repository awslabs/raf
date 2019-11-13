import mnm._ffi.executor as ffi
from mnm._core.module import Module


class Interpreter:  # pylint: disable=too-few-public-methods

    def __init__(self, module=None):
        if module is None:
            module = Module.GLOBAL
            self.runner = ffi.CreateInterpreter(module, True)
        else:
            self.runner = ffi.CreateInterpreter(module, module is Module.GLOBAL)
        self.module = module

    def __call__(self, expr):
        return self.runner(expr)


Interpreter.GLOBAL = Interpreter(Module.GLOBAL)
