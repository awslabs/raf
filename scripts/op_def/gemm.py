# pylint: disable=too-few-public-methods,too-many-arguments,unused-argument,invalid-name
# pylint: disable=missing-class-docstring,missing-function-docstring
"""Matrix-multiplication operator definitions and argument data structures."""
from .base import Op, Tensor


class MatmulArgs:

    @staticmethod
    def f(x1: Tensor, x2: Tensor) -> Tensor:
        pass

    __ops__ = [
        Op("matmul"),
        Op("matmul_nt"),
        Op("matmul_tn"),
        Op("matmul_tt"),
    ]
