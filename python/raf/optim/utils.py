# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimizer utilities."""
import math

import numpy as np

from raf._lib import relay
from raf._ffi.ir.constant import ExtractValue
from raf._ffi.binding import LookupBoundExpr
from raf._core.value import NoGradValue
from raf._core.ndarray import ndarray, get_symbol_handle


def has_grad(dx):
    """Check if dx is NoGradValue"""

    def simplify(x):
        if isinstance(x, relay.Var):
            return simplify(LookupBoundExpr(x))
        if isinstance(x, relay.TupleGetItem):
            tup = simplify(x.tuple_value)
            if isinstance(tup, relay.Tuple):
                return simplify(tup[x.index])
        return x

    dx = simplify(get_symbol_handle(dx))
    if isinstance(dx, relay.Constant):
        dx = ExtractValue(dx)
        return not isinstance(dx, NoGradValue)
    return True


def split_ndarray_with_padding(inp, n_part):
    """
    Split the first axis of the ndarray to N parts evenly. If the first axis
    is not diviable, then zero-padding will be applied before splitting.
    Note that the given ndarray has to be on CPU.
    """
    if isinstance(inp, ndarray):
        assert inp.device == "cpu", "The array must be on CPU, but on %s" % inp.device
        inp = inp.numpy()
    assert isinstance(inp, np.ndarray)
    part_first_dim_size = math.ceil(inp.shape[0] / n_part)
    pad_first_dim_size = part_first_dim_size * n_part

    if pad_first_dim_size > inp.shape[0]:
        pad_width = [(0, 0) for _ in inp.shape]
        pad_width[0] = (0, pad_first_dim_size - inp.shape[0])
        inp = np.pad(inp, pad_width)
    return np.split(inp, n_part)
