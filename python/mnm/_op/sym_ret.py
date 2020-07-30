# pylint: disable=missing-module-docstring,missing-function-docstring
from mnm._core.ndarray import Symbol


def Any(x):  # pylint: disable=invalid-name
    return Symbol.from_expr(x)
