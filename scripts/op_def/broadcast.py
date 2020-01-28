from .base import Op, ArrayLike

# pylint: disable=too-few-public-methods,too-many-arguments,unused-argument,invalid-name


class UnaryArgs:

    @staticmethod
    def f(x: ArrayLike) -> ArrayLike:
        ...

    __ops__ = [
        Op("copy"),
        Op("tanh"),
        Op("abs"),
        Op("ceil"),
        Op("cos"),
        Op("floor"),
        Op("log"),
        Op("sigmoid"),
        Op("negative"),
        Op("logical_not"),
        Op("relu"),
    ]


class UnaryDxArgs:

    @staticmethod
    def f(y: ArrayLike,
          dy: ArrayLike,
          x: ArrayLike,
          ) -> ArrayLike:
        ...

    __ops__ = [
        Op("relu_dx"),
        Op("tanh_dx"),
        Op("sigmoid_dx"),
    ]


class BinaryArgs:

    @staticmethod
    def f(x1: ArrayLike,
          x2: ArrayLike,
          ) -> ArrayLike:
        ...

    __ops__ = [
        Op("add"),
        Op("subtract"),
        Op("multiply"),
        Op("divide"),
        Op("mod"),
        Op("less"),
        Op("greater"),
        Op("less_equal"),
        Op("greater_equal"),
        Op("equal"),
        Op("not_equal"),
    ]

# TODO(@junrushao1994): implement `ufunc`s
