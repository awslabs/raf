from .base import Op, Tensor, TupleInt

# pylint: disable=too-few-public-methods,too-many-arguments,unused-argument,invalid-name


class BatchFlattenArgs:

    @staticmethod
    def f(x: Tensor) -> Tensor:
        ...

    __ops__ = [
        Op("batch_flatten"),
    ]


class ReshapeArgs:
    # Arguments:
    #   x: parameter
    #   g: gradient
    #   v: velocity
    #   mu: momentum
    #   lr: learning rate
    #
    # Update rule:
    #   v' = mu * v + g
    #   x' = x - lr * v'
    #
    # Update rule (Nesterov):
    #   v' = mu * v + lr * g
    #   x' = x - v'

    @staticmethod
    def f(x: Tensor,
          shape: TupleInt,
          ) -> Tensor:
        ...

    __ops__ = [
        Op("reshape"),
    ]
