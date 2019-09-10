""" Utilities for py-op interfaces. """

def int2tuple(x):
    assert isinstance(x, (int, tuple))
    return (x, x) if isinstance(x, int) else x
