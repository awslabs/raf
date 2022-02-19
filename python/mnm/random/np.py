# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=missing-function-docstring
"""Random tensor samplers from numpy."""
import numpy as np

from mnm._core.ndarray import ndarray


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
