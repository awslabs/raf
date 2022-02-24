# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-class-docstring,missing-function-docstring
"""Module that consists of global variables and functions."""
import raf._ffi.ir.module as ffi
from raf._lib import IRModule  # pylint: disable=unused-import


def get_global():
    return ffi.Global()
