from .base import Any, Op, Tensor

# pylint: disable=too-few-public-methods,too-many-arguments,unused-argument,invalid-name


class TensorAttrsArgs:

    @staticmethod
    def f(x: Tensor) -> Any:
        ...

    __ops__ = [
        Op("tensor.shape", namespace="__hidden__"),
        Op("tensor.dtype", namespace="__hidden__"),
        Op("tensor.ctx", namespace="__hidden__"),
        Op("tensor.ndim", namespace="__hidden__"),
    ]


class BroadcastRelationsArgs:

    @staticmethod
    def f(source: Tensor,
          target: Tensor,
          ) -> Any:
        ...

    __ops__ = [
        Op("bcast_rel.bwd_axis", namespace="__hidden__"),
        Op("bcast_rel.bwd_keepdims", namespace="__hidden__"),
    ]
