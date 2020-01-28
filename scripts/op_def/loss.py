from .base import Op, Tensor

# pylint: disable=too-few-public-methods,too-many-arguments,unused-argument,invalid-name


class LossArgs:

    __ops__ = [
        Op("nll_loss"),
        Op("nll_loss_dtrue"),
        Op("nll_loss_dpred"),
    ]

    @staticmethod
    def f(y_true: Tensor,
          y_pred: Tensor,
          ) -> Tensor:
        ...
