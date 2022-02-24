# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring
"""Random tensor samplers from numpy."""
import numpy as np

from raf._core.ndarray import ndarray


def _wrap(np_ndarray, name="", dtype="float32", device="cpu"):
    ret = np_ndarray
    if not isinstance(ret, np.ndarray):
        ret = np.array(ret)
    ret = ret.astype(dtype)
    return ndarray(ret, name=name, device=device, dtype=dtype)


def uniform(
    low=0.0, high=1.0, shape=None, name="", device="cpu", dtype="float32"
):  # pylint: disable=too-many-arguments
    return _wrap(np.random.uniform(low=low, high=high, size=shape), name, dtype, device)


def normal(
    mean=0.0, std=1.0, shape=None, name="", device="cpu", dtype="float32"
):  # pylint: disable=too-many-arguments
    return _wrap(np.random.normal(loc=mean, scale=std, size=shape), name, dtype, device)


def zeros_(
    shape=None, name="", device="cpu", dtype="float32"
):  # pylint: disable=too-many-arguments
    return _wrap(np.zeros(shape), name, dtype, device)


def ones_(shape=None, name="", device="cpu", dtype="float32"):  # pylint: disable=too-many-arguments
    return _wrap(np.ones(shape), name, dtype, device)
