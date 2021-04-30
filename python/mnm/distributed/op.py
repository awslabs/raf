# pylint: disable=protected-access, invalid-name
"""Collective communication operators"""
from .._op import sym
from .context import get_context

def allreduce(x):
    """General allreduce operators, take tensor or list of tensors as input."""
    if not isinstance(x, (tuple, list)):
        x = [x]
    return sym._allreduce(x)


def allgather(x, axis):
    """It performs concatenation across replicas.

    Parameters
    ----------
    x : Tensor | [Tensor]
        The tensor(s) to be concatenated across replicas
    axis : int
        The axis over which concatenation is to be performed

    Returns
    -------
    ret: Tensor | [Tensor]
        Concatenation results
    """
    def swap_axis(x):
        return x if axis == 0 else sym.swap_axis(x, axis1=0, axis2=axis)

    is_list = isinstance(x, (tuple, list))
    dctx = get_context()
    if not is_list:
        x = [x]
    l = len(x)

    # pack the list of tensors into a single tensor
    x = [swap_axis(i) for i in x]
    if l > 1:
        sx = [sym.shape(sym.repeat(i, axis=0, repeats=dctx.size)) for i in x]
        x = [sym.reshape(i, (-1,)) for i in x]
        indices_or_sections = sym.concatenate_dx(x, axis=0)
        x = sym.concatenate(x, axis=0)
    else:
        x = x[0]

    # broadcast the packed tensor
    x = sym._allgather(x, axis=0)

    # unpack the tensor
    if l > 1:
        x = sym.reshape(x, (dctx.size, -1))
        x = sym.split(x, indices_or_sections=indices_or_sections, axis=1)
        x = [sym.reshape(x[i], sx[i]) for i in range(l)]
    else:
        x = [x]
    x = [swap_axis(x[i]) for i in range(l)]

    if not is_list:
        x = x[0]
    return x
