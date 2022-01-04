# pylint: disable=protected-access, invalid-name
"""Collective communication operators"""
from .._op import sym
from .context import get_context


def allreduce(x, computation="sum", rank_list=None):
    """General allreduce operators, take tensor or list of tensors as input."""
    if not isinstance(x, (tuple, list)):
        x = [x]
    return sym._allreduce(x, computation, rank_list)


def allgather(x, axis, rank_list=None):
    """It performs concatenation across replicas.

    Parameters
    ----------
    x : Tensor | [Tensor]
        The tensor(s) to be concatenated across replicas
    axis : int
        The axis over which concatenation is to be performed
    rank_list: [int]
        The list of ranks to communicate. If the rank list leaves empty,
        all the nodes will perform the calculation. Note that this operator
        should be invoked by all machines at the first time, including the nodes
        that is not in the rank list since creating a new sub-communicator will
        invoke MPI_Bcast(MPI_COMM_WORLD) for now.


    Returns
    -------
    ret: Tensor | [Tensor]
        Concatenation results
    """

    def swap_axis(x):
        return x if axis == 0 else sym.swap_axis(x, axis1=0, axis2=axis)

    is_list = isinstance(x, (tuple, list))

    dctx = get_context()
    if rank_list:
        if dctx.rank in rank_list:
            size = len(rank_list)
        else:
            size = 1
    else:
        size = dctx.size

    if not is_list:
        x = [x]
    l = len(x)

    # pack the list of tensors into a single tensor
    x = [swap_axis(i) for i in x]
    if l > 1:
        sx = [sym.shape(sym.repeat(i, axis=0, repeats=size)) for i in x]
        x = [sym.reshape(i, (-1,)) for i in x]
        indices_or_sections = sym.concatenate_dx(x, axis=0)
        x = sym.concatenate(x, axis=0)
    else:
        x = x[0]

    # broadcast the packed tensor
    x = sym._allgather(x, 0, rank_list)

    # unpack the tensor
    if l > 1:
        x = sym.reshape(x, (size, -1))
        x = sym.split(x, indices_or_sections=indices_or_sections, axis=1)
        x = [sym.reshape(x[i], sx[i]) for i in range(l)]
    else:
        x = [x]
    x = [swap_axis(x[i]) for i in range(l)]

    if not is_list:
        x = x[0]
    return x


def reduce(x, root, computation="sum"):
    """Performs reduce operation. Collect data to root rank

    Parameters
    ----------
    x : Tensor or list of Tensor
        Tensor(s) to be reduced
    root: int
        The root rank
    computation: string
        The reduction operation, default is sum

    Returns
    -------
    ret: Tensor
        reduction result
    """
    if not isinstance(x, (tuple, list)):
        x = [x]
    return sym._reduce(x, root, computation)


def reduce_scatter(x, computation="sum"):
    """Performs reduction then scatter

    Parameters
    ----------
    x : List[Tensor]
        A list of tensors of equal shape
        replica i receives reduction of x[i] over all replicas
    computation: string
        The reduction operation, default is sum

    Returns
    -------
    ret: Tensor
        reduction result of x[rank] over all replicas,
        where rank represents rank number of the current process
    """
    return sym._reduce_scatter(x, computation)


def broadcast(x, root):
    """Performs broadcast

    Parameters
    ----------
    x : List[Tensor]
        A list of tensors on rank root to broadcast
    root : int
        root rank

    Returns
    -------
    ret: List[Tensor]
        broadcast-ed results
    """
    if not isinstance(x, (tuple, list)):
        x = [x]
    return sym._broadcast(x, root)


def send(x, peer, token=None):
    """Send x to peer.
    This operation is blocking for GPU.

    Parameters
    ----------
    x : Tensor
        The tensor to be sent

    peer : int
        The send destination

    token : OptionalTensor
        A frame of data that introduces data dependency so that send will not be reordered

    Returns
    -------
    ret: Tensor
        a tensor of zero dimension, which is equivalent to "no return value"
    """
    return sym._send(x, peer=peer, token=token)


def recv(peer, shape, dtype, token=None):
    """Receive a tensor from peer
    This operation is blocking for GPU.

    Parameters
    ----------
    peer : int
        The peer who sends the tensor

    shape : Tuple[int]
        The shape of the tensor to be received

    dtype : String
        The dtype of the tensor to be received

    token : OptionalTensor
        A frame of data that introduces data dependency so that recv will not be reordered

    Returns
    -------
    ret: Tensor
        the received tensor
    """
    return sym._recv(peer=peer, shape=shape, dtype=dtype, token=token)
