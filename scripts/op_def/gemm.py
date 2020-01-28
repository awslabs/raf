from .base import Op, Tensor

# pylint: disable=too-few-public-methods,too-many-arguments,unused-argument,invalid-name


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
