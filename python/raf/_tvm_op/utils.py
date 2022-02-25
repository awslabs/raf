# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for processing TVM ops."""
# pylint: disable=protected-access
import os

import tvm


@tvm._ffi.register_func("raf._tvm_op.utils.export_library")
def export_library(mod, path):
    """Export a built TVM runtime module to be a shared library (.so) file.

    Parameters
    ----------
    mod : tvm.runtime.Module
        The TVM runtime module to be exported.
    path : str
        The path to the shared library file.

    Returns
    -------
    bool
        Whether the export was successful.
    """
    mod.export_library(path)
    return os.path.exists(path)


@tvm._ffi.register_func("raf._tvm_op.utils.load_module")
def load_module(path):
    """Load a module from a .so file.

    Parameters
    ----------
    path : str
        The path to the .so file.

    Returns
    -------
    tvm.runtime.module.Module
        The loaded module.
    """
    if not os.path.exists(path):
        raise RuntimeError("Module file does not exist {}".format(path))
    return tvm.runtime.module.load_module(path)
