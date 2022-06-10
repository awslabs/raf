# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name,protected-access, import-outside-toplevel
"""Common utilities for testing"""
import logging
import functools
import random
import os
import sys
import re
import numpy as np
import raf
from raf import distributed as dist
from .._op.dialect import DialectPreference


def check_type(expr, typ):
    """Helper function to check expr.checked_type == type"""
    checked_type = expr.checked_type
    if checked_type != typ:
        raise RuntimeError(f"Type mismatch {checked_type} vs {typ}")


def get_testable_devices():
    """Helper function to get testable devices"""
    ret = ["cpu"] if "RAF_DISABLE_CPU_TEST" not in os.environ else []
    if raf.build.with_cuda():
        ret.append("cuda")
    return ret


def get_arr_addr(arr):
    """Helper function to get the address of array"""
    # pylint: disable=protected-access
    if isinstance(arr, raf.ndarray):
        arr = arr._ndarray__value
    assert isinstance(arr, raf._core.value.TensorValue)
    return raf._ffi.value.ToTVM(arr).handle.contents.data


def numpy(x):
    """Helper function to convert x to numpy"""
    import torch

    if isinstance(x, (raf.ndarray, raf._core.value.TensorValue)):
        return x.numpy()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, "asnumpy"):
        return x.asnumpy()
    if np.isscalar(x):
        return np.array(x)
    assert isinstance(x, np.ndarray), f"{type(x)} is not supported"
    return x


def check(m_x, m_y, *, rtol=1e-5, atol=1e-5, dump_name_when_error=None):
    """Helper function to check if m_x and m_y are equal"""
    m_x = numpy(m_x)
    m_y = numpy(m_y)
    try:
        np.testing.assert_allclose(m_x, m_y, rtol=rtol, atol=atol)
    except Exception as err:  # pylint: disable=broad-except
        if dump_name_when_error is not None:
            m_x.tofile(dump_name_when_error + "_x.npy", sep=",")
            m_y.tofile(dump_name_when_error + "_y.npy", sep=",")
        raise Exception(err)


def to_torch_dev(device_str):
    """Change device string form `cuda(id)` to pytorch style `cuda:id`"""
    tokens = re.search(r"(\w+).?(\d?)", device_str)
    dev_type = tokens.groups()[0]
    dev_id = int(tokens.groups()[1]) if tokens.groups()[1] else 0
    return "%s:%d" % (dev_type, dev_id)


def randn(shape, *, device="cpu", dtype="float32", positive=False, requires_grad=False):
    """Helper function to generate a pair of raf and numpy arrays"""
    x = np.random.randn(*shape)
    if positive:
        x = np.abs(x) + 1e-5
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = raf.array(n_x, device=device)
    m_x.requires_grad = requires_grad
    return m_x, n_x


def randint(shape, *, low=0, high=None, device="cpu", dtype="int64"):
    """Helper function to generate a pair of raf and numpy arrays with int"""
    x = np.random.randint(low, high, shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = raf.array(n_x, device=device)
    return m_x, n_x


def randn_torch(
    shape, *, device="cpu", dtype="float32", requires_grad=False, mean=0.0, std=1.0, positive=False
):
    """Helper function to generate a pair of raf and torch arrays"""
    import torch

    x = np.random.randn(*shape) * std + mean
    if positive:
        x = np.abs(x) + 1e-5
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = raf.array(n_x, device=device)
    m_x.requires_grad = requires_grad
    t_x = torch.tensor(
        n_x, requires_grad=requires_grad, device=to_torch_dev(device)
    )  # pylint: disable=not-callable
    return m_x, t_x


def randn_mxnet(
    shape, *, device="cpu", dtype="float32", requires_grad=False, mean=0.0, std=1.0, positive=False
):
    """Helper function to generate a pair of raf and mxnet arrays"""
    import mxnet as mx

    x = np.random.randn(*shape) * std + mean
    if positive:
        x = np.abs(x) + 1e-5
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = raf.array(n_x, device=device)
    m_x.requires_grad = requires_grad
    mx_x = mx.nd.array(n_x, dtype=dtype, ctx=mx.cpu())
    if requires_grad:
        mx_x.attach_grad()
    return m_x, mx_x


def one_hot_torch(size, num_classes, device="cpu"):
    """Helper function to generate one hot tensors in raf and torch"""
    import torch

    size = tuple(size) if isinstance(size, (list, tuple)) else (size,)
    targets = np.random.randint(0, num_classes, size=size)
    m_x = raf.array(targets, device=device)
    t_x = torch.tensor(
        targets, requires_grad=False, device=to_torch_dev(device)
    )  # pylint: disable=not-callable
    assert tuple(m_x.shape) == size
    assert tuple(t_x.shape) == size
    return m_x, t_x


def one_hot_mxnet(size, num_classes, device="cpu"):
    """Helper function to generate one hot tensors in raf and mxnet"""
    import mxnet as mx

    size = tuple(size) if isinstance(size, (list, tuple)) else (size,)
    targets = np.random.randint(0, num_classes, size=size)
    raf_x = raf.array(targets, device=device)
    mx_x = mx.nd.array(targets, ctx=mx.cpu())  # pylint: disable=not-callable
    assert tuple(raf_x.shape) == size
    assert tuple(mx_x.shape) == size
    return raf_x, mx_x


def t2m_param(param, device="cuda"):
    """Helper function to convert torch parameter to raf ndarray"""
    return raf.ndarray(  # pylint: disable=unexpected-keyword-arg
        param.detach().cpu().numpy(), device=device
    )


def default_logger():
    """A logger used to output seed information to logs."""
    logger = logging.getLogger(__name__)
    # getLogger() lookups will return the same logger, but only add the handler once.
    if len(logger.handlers) == 0:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        if logger.getEffectiveLevel() == logging.NOTSET:
            logger.setLevel(logging.INFO)
    return logger


def with_seed(seed=None):
    """
    A decorator for test functions that manages rng seeds.

    Parameters
    ----------

    seed : the seed to pass to np.random and random


    This tests decorator sets the np and python random seeds identically
    prior to each test, then outputs those seeds if the test fails or
    if the test requires a fixed seed (as a reminder to make the test
    more robust against random data).

    @with_seed()
    def test_ok_with_random_data():
        ...

    @with_seed(1234)
    def test_not_ok_with_random_data():
        ...

    Use of the @with_seed() decorator for all tests creates
    tests isolation and reproducability of failures.  When a
    test fails, the decorator outputs the seed used.
    """

    def test_helper(orig_test):
        @functools.wraps(orig_test)
        def test_new(*args, **kwargs):
            if seed is not None:
                this_test_seed = seed
                log_level = logging.INFO
            else:
                this_test_seed = np.random.randint(0, np.iinfo(np.int32).max)
                log_level = logging.DEBUG
            post_test_state = np.random.get_state()
            np.random.seed(this_test_seed)
            random.seed(this_test_seed)
            logger = default_logger()
            # 'pytest --logging-level=DEBUG' shows this msg even with an ensuing core dump.
            pre_test_msg = (
                "Setting test np/python random seeds, use seed={}" " to reproduce."
            ).format(this_test_seed)
            on_err_test_msg = ("Error seen with seeded test, use seed={}" " to reproduce.").format(
                this_test_seed
            )
            logger.log(log_level, pre_test_msg)
            try:
                ret = orig_test(*args, **kwargs)
            except:
                # With exceptions, repeat test_msg at WARNING level to be sure it's seen.
                if log_level < logging.WARNING:
                    logger.warning(on_err_test_msg)
                raise
            finally:
                # Provide test-isolation for any test having this decorator
                np.random.set_state(post_test_state)
            return ret

        return test_new

    return test_helper


def with_dialect(dialect):
    """
    A decorator to specify available dialects

    Parameters
    ----------
    dialect : Union[str, List[str]]
    """

    def decorator(wrapped):
        @functools.wraps(wrapped)
        def wrapper(*args, **kwargs):
            dialects = [dialect] if isinstance(dialect, str) else dialect
            with DialectPreference(dialects):
                return wrapped(*args, **kwargs)

        return wrapper

    return decorator


def get_dist_comm_info(verbose=False):
    """Helper function to get the distributed communicator info.

    Parameters
    ----------
    verbose: bool
        Whether to print the distributed communicator information.

    Returns
    -------
    Tuple[int, int, int]
        A tuple of (total rank, self rank, self local rank)
    """
    comm = dist.get_communicator()
    root_rank = comm.root_rank
    rank = comm.rank
    size = comm.size
    local_rank = comm.local_rank
    local_size = comm.local_size

    if verbose and rank == 0:
        node_info = f"root_rank={root_rank},rank={rank}, \
        size={size},local_rank={local_rank}, local_size={local_size} "
        print(node_info)
    return size, rank, local_rank


def skip_dist_test(min_rank_num=1, require_exact_rank=False):
    """Helper function to determine whether to skip the unit tests for distributed training.

    Parameters
    ----------
    min_rank_num: int
        The minimial rank number required to run the test.

    require_exact_rank: bool
        Whether to require the exact number of rank to run the test.

    Returns
    -------
    bool
        Whether to skip the test.
    """
    if not raf.build.with_distributed():
        return True

    size, _, _ = get_dist_comm_info()
    if require_exact_rank:
        return size != min_rank_num
    return size < min_rank_num
