# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Types that used in operator definition/attributes."""
from typing import NewType, Optional, Tuple, Union

Context = NewType("Context", object)

DType = NewType("DType", object)

Tensor = NewType("Tensor", object)

ArrayLike = NewType("ArrayLike", object)

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
