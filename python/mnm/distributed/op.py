"""Collective communication operators"""
from .._op.sym import _allreduce


def allreduce(x):
    """General allreduce operators, take tensor or list of tensors as input."""
    if not isinstance(x, (tuple, list)):
        x = [x]
    return _allreduce(x)
