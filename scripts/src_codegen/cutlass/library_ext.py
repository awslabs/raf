# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The extended library. Origin: cutlass/tools/library/scripts/library.py"""
import enum
from library import *


class EpilogueFunctorExt(enum.Enum):
    LinearCombinationRelu = enum_auto()
    LinearCombinationGELU = enum_auto()


EpilogueFunctorTag.update(
    {
        EpilogueFunctorExt.LinearCombinationRelu: "cutlass::epilogue::thread::LinearCombinationRelu",
        EpilogueFunctorExt.LinearCombinationGELU: "cutlass::epilogue::thread::LinearCombinationGELU",
    }
)

EpilogueFunctorNames = {
    EpilogueFunctor.LinearCombination: "",
    EpilogueFunctor.LinearCombinationClamp: "clamp",
    EpilogueFunctorExt.LinearCombinationRelu: "relu",
    EpilogueFunctorExt.LinearCombinationGELU: "gelu",
}
