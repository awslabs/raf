from .base import Op, Tensor, Tuple

# pylint: disable=too-few-public-methods,too-many-arguments,unused-argument,invalid-name


class SgdArgs:
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
          g: Tensor,
          v: Tensor,
          mu: float,
          lr: float,
          ) -> Tuple[Tensor, Tensor]:
        ...

    __ops__ = [
        Op("optim.sgd"),
    ]
