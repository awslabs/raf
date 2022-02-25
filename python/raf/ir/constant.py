# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access,too-many-return-statements,too-many-branches
"""Utils to create constant expr."""
import numpy as _np

from raf._core import ndarray as _nd
from raf._core import value as _value
from raf._ffi.ir._make import Constant


def to_value(value, dtype=None):
    """Convert to value.

    Parameters
    ----------
    value: Union[bool, int, float, str, numpy.ndarray, raf.nd.ndarray, list, tuple]
        The constant value.

    dtype: str, optional
        The data type of the resulting constant.

    Note
    ----
    When dtype is None, we use the following rule:

    - int maps to "int64"
    - float maps to "float32"
    - bool maps to "bool"
    """
    # tuple value
    if isinstance(value, (tuple, list)):
        return _value.TupleValue([to_value(e, dtype) for e in value])

    # tensor value
    if isinstance(value, _np.ndarray):
        return _value.TensorValue.from_numpy(value)
    if isinstance(value, _nd.ndarray):
        return value._ndarray__value

    # scalar value
    if isinstance(value, str):
        return _value.StringValue(value)

    if not dtype:
        if isinstance(value, bool):
            dtype = "bool"
        elif isinstance(value, (int, _np.int64)):
            dtype = "int64"
        elif isinstance(value, _np.int32):
            dtype = "int32"
        elif isinstance(value, (float, _np.float32)):
            dtype = "float32"
        elif isinstance(value, _np.float64):
            dtype = "float64"
        elif isinstance(value, _np.float16):
            dtype = "float16"
        else:
            raise ValueError("Unknown value type: %s" % type(value))

    if dtype == "bool":
        return _value.BoolValue(value)
    if dtype.startswith("int"):
        return _value.IntValue(value, dtype=dtype)
    # float dtype
    assert dtype.startswith("float"), "Unknown dtype: %s" % dtype
    return _value.FloatValue(value, dtype=dtype)


def const(value, dtype=None):
    """Create a constant expr.

    Parameters
    ----------
    value: Union[bool, int, float, str, numpy.ndarray, raf.nd.ndarray, list, tuple, None]
        The constant value. If value is None, return a constant with nullptr.

    dtype: str, optional
        The data type of the resulting constant.

    Note
    ----
    When dtype is None, we use the following rule:

    - int maps to "int64"
    - float maps to "float32"
    - bool maps to "bool"
    """
    if value is None:
        return Constant(None)
    return Constant(to_value(value, dtype))
