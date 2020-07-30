"""Execute the program."""
import mnm._ffi.executor as ffi


def interpret(expr, module=None):
    """use interpreter to execute the program.

    Parameters
    ----------
    expr : relay.Call
        The function together with its arguments.
    module : mnm.ir.Module
        The module captures the global variables and functions.

    Returns
    -------
    ret: mnm.value.Value
        Executed results.
    """
    return ffi.Interpret(expr, module)
