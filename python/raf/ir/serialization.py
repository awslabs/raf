# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""IR serialization."""
from raf._ffi.ir.serialization import SaveJSON, LoadJSON


def save_json(node):
    """Save object as json string. It takes care of extended IR.

    Parameters
    ----------
    node : Object
        A TVM object to be saved.

    Returns
    -------
    json_str : str
        Saved json string.
    """
    return SaveJSON(node)


def load_json(json):
    """Load json string into object. It takes care of extended IR.

    Parameters
    ----------
    json: str
        Saved json string.

    Returns
    -------
    node : Object
        A loaded TVM object
    """
    return LoadJSON(json)
