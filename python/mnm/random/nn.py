# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring,too-many-arguments
"""NN-specific random initializers."""
import math

import numpy as np

from .np import normal, uniform


def _calc_fan_in_out(shape):
    ndim = len(shape)
    if ndim < 2:
        raise ValueError("Please provide shape with at least 2 dims")
    if ndim == 2:
        fan_out, fan_in = shape
        return fan_in, fan_out
    fan_out, fan_in = shape[:2]
    reception = int(np.prod(shape[2:]))
    return fan_in * reception, fan_out * reception


_GAIN_MAP = {
    "sigmoid": 1.0,
    "tanh": 5.0 / 3.0,
    "relu": math.sqrt(2.0),
}


def _calc_gain(nonlinearity, param=None):
    if nonlinearity in _GAIN_MAP:
        return _GAIN_MAP[nonlinearity]
    if nonlinearity == "leaky_relu":
        neg_slope = 0.01 if param is None else param
        return math.sqrt(2.0 / (1.0 + neg_slope * neg_slope))
    raise NotImplementedError("gain for nonlinearity: " + str(nonlinearity))


def xavier_uniform(shape, gain=1.0, name="", dtype="float32", device="cpu"):
    fan_in, fan_out = _calc_fan_in_out(shape)
    a = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return uniform(low=-a, high=a, shape=shape, name=name, dtype=dtype, device=device)


def xavier_normal(shape, gain=1.0, name="", dtype="float32", device="cpu"):
    fan_in, fan_out = _calc_fan_in_out(shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal(mean=0.0, std=std, shape=shape, name=name, dtype=dtype, device=device)


def kaiming_uniform(
    shape,
    a=0,
    mode="fan_in",
    nonlinearity="leaky_relu",
    name="",  # pylint: disable=too-many-arguments
    dtype="float32",
    device="cpu",
):
    fan_in, fan_out = _calc_fan_in_out(shape)
    if mode == "fan_in":
        fan = fan_in
    elif mode == "fan_out":
        fan = fan_out
    else:
        raise ValueError("Cannot recognize mode in kaiming_uniform", mode)
    gain = _calc_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform(low=-bound, high=bound, shape=shape, name=name, dtype=dtype, device=device)


def kaiming_normal(
    shape,
    a=0,
    mode="fan_in",
    nonlinearity="leaky_relu",
    name="",  # pylint: disable=too-many-arguments
    dtype="float32",
    device="cpu",
):
    fan_in, fan_out = _calc_fan_in_out(shape)
    if mode == "fan_in":
        fan = fan_in
    elif mode == "fan_out":
        fan = fan_out
    else:
        raise ValueError("Cannot recognize mode in kaiming_uniform", mode)
    gain = _calc_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal(0, std, name=name, dtype=dtype, device=device)
