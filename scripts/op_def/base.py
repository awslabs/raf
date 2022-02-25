# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-class-docstring,missing-function-docstring
"""Basic data structures for operator definition."""
from dataclasses import dataclass  # pylint: disable=import-error
from typing import Any, List, NewType, Optional, Tuple, Union

Annotation = NewType("Annotation", object)

__types__ = [
    # basic types:
    #   int,
    #   float,
    #   str,
    #   bool
    # advanced types
    "Context",
    "DType",
    "Tensor",
    "ArrayLike",
    "TupleInt",
    "IntOrTupleInt",
    "IntOrTupleIntOrNone",
    "BoolOrTupleInt",
]

__all__ = [
    # dataclass
    "Arg",
    "Schema",
    "Op",
    "API",
    # from typing
    "Any",
    "Tuple",
    # special value
    "NO_DEFAULT",
] + __types__


@dataclass
class Arg:
    name: str
    type: Annotation
    default: Any


@dataclass
class Schema:
    name: str
    args: List[Arg]
    ret_type: Annotation
    module: str


@dataclass
class Op:
    name: str
    namespace: str = ""
    schema: Schema = None


@dataclass
class API:
    name: str
    path: str
    lineno: int


class _NoDefaultType:  # pylint: disable=too-few-public-methods
    pass


NO_DEFAULT = _NoDefaultType()


# Context: nullable
#     str
#     DLContext
Context = NewType("Context", object)

# DType: nullable
#     str
#     DLDataType
DType = NewType("DType", object)

# Tensor: nullable
#     ndarray
Tensor = NewType("Tensor", object)

# ArrayLike: nullable
#     scalar
#     ndarray
ArrayLike = NewType("ArrayLike", object)

# TupleInt: not nullable
#     tuple of integers
TupleInt = Tuple[int, ...]

# IntOrTupleInt: not nullable
#     integers
#     tuple of integers
IntOrTupleInt = Union[int, TupleInt]

# IntOrTupleIntOrNone: nullable
#     None
#     integers
#     tuple of integers
IntOrTupleIntOrNone = Optional[IntOrTupleInt]

# BoolOrTupleInt: nullable
#     None
#     True/False
#     tuple of integers
BoolOrTupleInt = Union[bool, TupleInt]
