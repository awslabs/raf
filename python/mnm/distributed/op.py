# pylint: disable=protected-access, invalid-name
"""Collective communication operators"""
from .._op import sym
from .context import get_context

def allreduce(x, computation="sum"):
    """General allreduce operators, take tensor or list of tensors as input."""
    if not isinstance(x, (tuple, list)):
        x = [x]
    return sym.comm_allreduce(x, computation)


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
    x = sym.comm_allgather(x, axis=0)

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


def reduce_scatter(x):
    """Performs reduction then scatter

    Parameters
    ----------
    x : List[Tensor]
        A list of tensors of equal shape
        replica i receives reduction of x[i] over all replicas

    Returns
    -------
    ret: Tensor
        reduction result of x[rank] over all replicas,
        where rank represents rank number of the current process
    """
    return sym.comm_reduce_scatter(x)


def send(x, peer):
    """ Send x to peer.
    This operation is blocking for GPU.

    Parameters
    ----------
    x : Tensor
        The tensor to be sent

    peer : int
        The send destination

    Returns
    -------
    ret: Tensor
        a tensor of zero dimension, which is equivalent to "no return value"
    """
    return sym.comm_send(x, peer=peer)


def recv(peer, shape, dtype):
    """ Receive a tensor from peer
    This operation is blocking for GPU.

    Parameters
    ----------
    peer : int
        The peer who sends the tensor

    shape : Tuple[int]
        The shape of the tensor to be received

    dtype : String
        The dtype of the tensor to be received

    Returns
    -------
    ret: Tensor
        the received tensor
    """
    return sym.comm_recv(peer=peer, shape=shape, dtype=dtype)
