# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dialect related functions"""
import threading

from .. import build as _build
from .._ffi import op as _ffi
from .._lib import Object
from .._core.core_utils import register_node


def enabled(dialect, device):
    """Check if a dialect is enabled on a device.

    Parameters
    ----------
    dialect : str
        The dialect name.
    device : device.Device
        The device.

    Returns
    -------
    ret : bool
        Whether the dialect is enabled.
    """
    _ffi.DialectEnabled(dialect, device.device_type)


def register_pattern(pattern, dialect, plevel, name=""):
    """Register a dialect fusion pattern.

    Parameters
    ----------
    pattern : DFPattern
        The fusion pattern.

    dialect : str
        The dialect to use for the pattern.

    plevel : int
        The priority level.

    name : str
        The pattern name.
    """
    if _build.build_with(dialect):
        _ffi.AddDialectPattern(pattern, dialect, plevel, name)


@register_node("raf.op.DialectPreference")
class DialectPreference(Object):
    """Dialect scope to specify a list of preferred dialects

    Parameters
    ----------
    dialects : List[str]
        The list of preferred dialects, with descending priority
    """

    valid_dialects = _ffi.GetAllDialects()
    storage = threading.local()

    def __init__(self, dialects):
        if not set(DialectPreference.valid_dialects).issuperset(set(dialects)):
            raise ValueError(
                f"{set(dialects).difference(set(DialectPreference.valid_dialects))} "
                "are not valid backends"
            )
        self.__init_handle_by_constructor__(_ffi.DialectPreference, dialects)

    def __enter__(self):
        _ffi.DialectPrefEnterScope(self)
        return self

    def __exit__(self, ptype, value, traceback):
        _ffi.DialectPrefExitScope(self)
