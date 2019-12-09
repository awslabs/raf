import numpy as np

import mnm


def _wrap(np_ndarray, name=""):
    ret = np_ndarray
    if not isinstance(ret, np.ndarray):
        ret = np.array(ret)
    return mnm.ndarray(ret, name=name)  # pylint: disable=unexpected-keyword-arg


def uniform(low=0.0, high=1.0, shape=None, name=""):
    return _wrap(np.random.uniform(low=low, high=high, size=shape), name)


def normal(mean=0.0, std=1.0, shape=None, name=""):
    return _wrap(np.random.normal(loc=mean, scale=std, size=shape), name)
