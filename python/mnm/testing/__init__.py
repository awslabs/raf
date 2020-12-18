#pylint: disable=invalid-name,protected-access
"""Utilities for testing and benchmarks"""
import logging
import functools
import random
import sys
import numpy as np
import torch
import mnm
import tvm

from .._core.module import Module
from .._core.executor import VMExecutor, VMCompiler
from .._ffi import ir
from .._ffi import pass_
from ..model.trace import _get_func_inputs


def run_infer_type(func):
    """Helper function to infer the type of the given function"""
    main = tvm.relay.GlobalVar("main")
    mod = ir._make.Module({main: func})
    mod = pass_.InferType(mod)
    return mod["main"]


def check_type(expr, typ):
    """Helper function to check expr.checked_type == typ"""
    checked_type = expr.checked_type
    if checked_type != typ:
        raise RuntimeError(f"Type mismatch {checked_type} vs {typ}")


def get_ctx_list():
    """Helper function to get all available contexts"""
    ret = ["cpu"]
    if mnm.build.with_cuda():
        ret.append("cuda")
    return ret


def check(m_x, m_y, *, rtol=1e-5, atol=1e-5):
    """Helper function to check if m_x and m_y are equal"""
    def _convert(x):
        if isinstance(x, (mnm.ndarray, mnm._core.value.TensorValue)):
            return x.asnumpy()
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if np.isscalar(x):
            return np.array(x)
        assert isinstance(x, np.ndarray), f"{type(x)} is not supported"
        return x
    m_x = _convert(m_x)
    m_y = _convert(m_y)
    np.testing.assert_allclose(m_x, m_y, rtol=rtol, atol=atol)


def randn(shape, *, ctx="cpu", dtype="float32", positive=False):
    """Helper function to generate a pair of mnm and numpy arrays"""
    x = np.random.randn(*shape)
    if positive:
        x = np.abs(x) + 1e-5
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    return m_x, n_x


def randint(shape, *, low=0, high=None, ctx="cpu", dtype="int64"):
    """Helper function to generate a pair of mnm and numpy arrays with int"""
    x = np.random.randint(low, high, shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    return m_x, n_x


def randn_torch(shape, *, ctx="cpu", dtype="float32", requires_grad=True, mean=0.0, std=1.0,
                positive=False):
    """Helper function to generate a pair of mnm and torch arrays"""
    x = np.random.randn(*shape) * std + mean
    if positive:
        x = np.abs(x) + 1e-5
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    t_x = torch.tensor(n_x, requires_grad=requires_grad, device=ctx)  # pylint: disable=not-callable
    return m_x, t_x


def run_vm_model(model, ctx, args, optimize=None):
    """Helper function to execute model with VM"""
    record = model._internal(*args)
    func = record.func
    if optimize:
        func = optimize(func)
    mod = Module()
    mod[tvm.ir.GlobalVar('main')] = func
    inputs = _get_func_inputs(record, args, {}, get_handle=False)
    executor = VMExecutor(mod, ctx)
    out = executor.make_executor()(*inputs)
    return out


def compile_vm_model(model, ctx, args):
    """Helper function to compile model into VM bytecode"""
    mod = Module()
    func = model._internal(*args).func
    mod[tvm.ir.GlobalVar('main')] = func
    executor = VMExecutor(mod, ctx)
    return executor.executable.bytecode


def lower_vm_model(model, target_name, args):
    """Helper function to lower model into optimized relay"""
    mod = Module()
    func = model._internal(*args).func
    gvar = tvm.ir.GlobalVar('main')
    mod[gvar] = func
    compiler = VMCompiler()
    mod, _ = compiler.optimize(mod, target_name)
    return mod[gvar]


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
