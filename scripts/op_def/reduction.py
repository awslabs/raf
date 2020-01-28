from .base import ArrayLike, BoolOrTupleInt, IntOrTupleIntOrNone, Op

# pylint: disable=too-few-public-methods,too-many-arguments,unused-argument,invalid-name


class ReductionArgs:
    # TODO(@junrushao1994): add dtype

    # axis = None: reduce all axes
    # axis = int | tuple[int]: reduce specific axis/axes

    # keepdims = true: keep all reduction dims
    # keepdims = false: keep no reduction dims
    # keepdims = tuple[int]: keep specific reduction dims

    @staticmethod
    def f(x: ArrayLike,
          axis: IntOrTupleIntOrNone = None,
          keepdims: BoolOrTupleInt = False,
          ) -> ArrayLike:
        ...

    __ops__ = [
        Op("sum"),
        # Op("all"),
        # Op("any"),
        # Op("max"),
        # Op("min"),
        # Op("argmax"),
        # Op("argmin"),
        # Op("prod"),
    ]
