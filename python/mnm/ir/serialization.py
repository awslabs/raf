"""IR serialization."""
from mnm._ffi.ir.serialization import SaveJSON


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
