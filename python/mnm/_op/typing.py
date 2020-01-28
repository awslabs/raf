from typing import NewType, Optional, Tuple, Union

Context = NewType('Context', object)

DType = NewType('DType', object)

Tensor = NewType('Tensor', object)

ArrayLike = NewType('ArrayLike', object)

TupleInt = Tuple[int, ...]

IntOrTupleInt = Union[int, TupleInt]

IntOrTupleIntOrNone = Optional[IntOrTupleInt]

BoolOrTupleInt = Union[bool, TupleInt]

__all__ = [
    "Context",
    "DType",
    "Tensor",
    "ArrayLike",
    "TupleInt",
    "IntOrTupleInt",
    "IntOrTupleIntOrNone",
    "BoolOrTupleInt",
]
