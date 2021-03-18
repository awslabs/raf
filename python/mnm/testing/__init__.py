#pylint: disable=invalid-name,protected-access
"""Utilities for testing and benchmarks"""
import logging
import functools
import random
import sys
import numpy as np
import mxnet as mx
import torch
import mnm
import tvm

from .._core.module import IRModule
from .._core.executor import VMExecutor, VMCompiler
from .._ffi import ir
from .._ffi import pass_
from ..model.trace import _get_func_inputs


def run_infer_type(expr):
    """Helper function to infer the type of the given expr """
    if isinstance(expr, IRModule):
        return pass_.InferType(expr)
    mod = IRModule.from_expr(expr)
    mod = pass_.InferType(mod)
    return mod["main"]


def check_type(expr, typ):
    """Helper function to check expr.checked_type == typ"""
    checked_type = expr.checked_type
    if checked_type != typ:
        raise RuntimeError(f"Type mismatch {checked_type} vs {typ}")


def get_device_list():
    """Helper function to get all available contexts"""
    ret = ["cpu"]
    if mnm.build.with_cuda():
        ret.append("cuda")
    return ret


def get_arr_addr(arr):
    """Helper function to get the address of array"""
    # pylint: disable=protected-access
    if isinstance(arr, mnm.ndarray):
        arr = arr._ndarray__value
    assert isinstance(arr, mnm._core.value.TensorValue)
    return mnm._ffi.value.ToTVM(arr).handle.contents.data


def asnumpy(x):
    """Helper function to convert x to numpy"""
    if isinstance(x, (mnm.ndarray, mnm._core.value.TensorValue)):
        return x.asnumpy()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, mx.nd.NDArray):
        return x.asnumpy()
    if np.isscalar(x):
        return np.array(x)
    assert isinstance(x, np.ndarray), f"{type(x)} is not supported"
    return x


def check(m_x, m_y, *, rtol=1e-5, atol=1e-5):
    """Helper function to check if m_x and m_y are equal"""
    m_x = asnumpy(m_x)
    m_y = asnumpy(m_y)
    np.testing.assert_allclose(m_x, m_y, rtol=rtol, atol=atol)


def randn(shape, *, device="cpu", dtype="float32", positive=False, requires_grad=False):
    """Helper function to generate a pair of mnm and numpy arrays"""
    x = np.random.randn(*shape)
    if positive:
        x = np.abs(x) + 1e-5
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, device=device)
    m_x.requires_grad = requires_grad
    return m_x, n_x


def randint(shape, *, low=0, high=None, device="cpu", dtype="int64"):
    """Helper function to generate a pair of mnm and numpy arrays with int"""
    x = np.random.randint(low, high, shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, device=device)
    return m_x, n_x


def randn_torch(shape, *, device="cpu", dtype="float32", requires_grad=False, mean=0.0, std=1.0,
                positive=False):
    """Helper function to generate a pair of mnm and torch arrays"""
    x = np.random.randn(*shape) * std + mean
    if positive:
        x = np.abs(x) + 1e-5
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, device=device)
    m_x.requires_grad = requires_grad
    t_x = torch.tensor(n_x, requires_grad=requires_grad, device=device)  # pylint: disable=not-callable
    return m_x, t_x


def randn_mxnet(shape, *, device="cpu", dtype="float32", requires_grad=False, mean=0.0, std=1.0,
                positive=False):
    """Helper function to generate a pair of mnm and mxnet arrays"""
    x = np.random.randn(*shape) * std + mean
    if positive:
        x = np.abs(x) + 1e-5
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, device=device)
    m_x.requires_grad = requires_grad
    mx_x = mx.nd.array(n_x, dtype=dtype, ctx=mx.cpu())
    if requires_grad:
        mx_x.attach_grad()
    return m_x, mx_x


def one_hot_torch(batch_size, num_classes, device="cpu", dtype="float32"):
    """Helper function to generate one hot tensors in mnm and torch"""
    targets = np.random.randint(0, num_classes, size=batch_size)
    m_x = np.zeros([batch_size, num_classes], dtype=dtype)
    m_x[range(batch_size), targets] = 1
    m_x = mnm.array(m_x, device=device)
    t_x = torch.tensor(targets, requires_grad=False, device=device)  # pylint: disable=not-callable
    assert list(m_x.shape) == [batch_size, num_classes]
    assert list(t_x.shape) == [batch_size]
    return m_x, t_x

def one_hot_mxnet(batch_size, num_classes, device="cpu", dtype="float32"):
    """Helper function to generate one hot tensors in mnm and torch"""
    targets = np.random.randint(0, num_classes, size=batch_size)
    m_x = np.zeros([batch_size, num_classes], dtype=dtype)
    m_x[range(batch_size), targets] = 1
    mnm_x = mnm.array(m_x, device=device)

    mx_x = mx.nd.array(targets, ctx=mx.cpu())  # pylint: disable=not-callable
    assert list(mnm_x.shape) == [batch_size, num_classes]
    assert list(mx_x.shape) == [batch_size]
    return mnm_x, mx_x

def t2m_param(param, device="cuda"):
    """Helper function to convert torch parameter to mnm ndarray"""
    return mnm.ndarray(param.detach().cpu().numpy(), device=device)  # pylint: disable=unexpected-keyword-arg


def run_vm_model(model, device, args, optimize=None):
    """Helper function to execute model with VM"""
    record = model._internal(*args)
    mod = record.mod
    if optimize:
        mod = optimize(mod)
    inputs = _get_func_inputs(record, args, {}, get_handle=False)
    executor = VMExecutor(mod, device)
    out = executor.make_executor()(*inputs)
    return out


def compile_vm_model(model, device, args):
    """Helper function to compile model into VM bytecode"""
    mod = model._internal(*args).mod
    executor = VMExecutor(mod, device)
    return executor.executable.bytecode


def lower_vm_model(model, target_name, args):
    """Helper function to lower model into optimized relay"""
    mod = model._internal(*args).mod
    compiler = VMCompiler()
    mod, _ = compiler.optimize(mod, target_name)
    # TODO (janimesh) - Revisit where the output is used
    return mod['main']


def default_logger():
    """A logger used to output seed information to logs."""
    logger = logging.getLogger(__name__)
    # getLogger() lookups will return the same logger, but only add the handler once.
    if len(logger.handlers) == 0:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
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
            pre_test_msg = ('Setting test np/python random seeds, use seed={}'
                            ' to reproduce.').format(this_test_seed)
            on_err_test_msg = ('Error seen with seeded test, use seed={}'
                               ' to reproduce.').format(this_test_seed)
            logger.log(log_level, pre_test_msg)
            try:
                orig_test(*args, **kwargs)
            except:
                # With exceptions, repeat test_msg at WARNING level to be sure it's seen.
                if log_level < logging.WARNING:
                    logger.warning(on_err_test_msg)
                raise
            finally:
                # Provide test-isolation for any test having this decorator
                np.random.set_state(post_test_state)
        return test_new
    return test_helper
