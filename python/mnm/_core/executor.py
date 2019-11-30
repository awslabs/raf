import mnm._ffi.executor as ffi

def interpret(expr, module=None):
    return ffi.Interpret(expr, module)
