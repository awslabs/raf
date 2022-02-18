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

"""Utilities for processing TVM ops."""
# pylint: disable=protected-access
import os

import tvm


@tvm._ffi.register_func("mnm._tvm_op.utils.export_library")
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


@tvm._ffi.register_func("mnm._tvm_op.utils.load_module")
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
