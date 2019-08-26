'''
from ._executor import CreateInterpreter
from .ir.module import Module
from _tvm import relay
from ._op import GetOp


class Interpreter(object):

    def __init__(self, module=None):
        if module is None:
            module = Module.get_global()
        self.module = module
        self.runner = CreateInterpreter(module)

    def call_function(self, global_var, args, attrs=None):
        return self.runner(relay.Call(global_var, args=args, attrs=attrs))

    def call_primitive(self, op, args, attrs):
        if isinstance(op, str):
            op = GetOp(op_name)
        return self.runner(relay.Call(op, args=args, attrs=attrs))


GLOBAL = Interpreter(Module.GLOBAL)
'''
