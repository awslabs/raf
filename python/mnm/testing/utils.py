"""testing utilities for models"""
import tvm

import mnm
from mnm.testing import run_infer_type
from mnm.model.trace import _get_func_inputs
from mnm._core.executor import VMExecutor
from mnm._core.profiler_vm import VMProfilerExecutor
from mnm._core.module import Module
from mnm._core.core_utils import get_chained_attr


def get_param(model, name):
    """get parameter from model"""
    if isinstance(name, str):
        name = name.split('.')
    ret = get_chained_attr(model, name)
    if ret is None:
        raise AttributeError(f"No attribute {name}")
    return ret


def set_param(model, name, value):
    """set the value of a parameter"""
    if isinstance(name, str):
        name = name.split('.')
    assert len(name) > 0
    ins = get_param(model, name[:-1])
    setattr(ins, name[-1], value)


# TODO(@hzfan): remove this after we have PassContext
def ir_simplify(func):
    # pylint: disable=protected-access
    """simplify func by simplifying tuples and eliminating dead code"""
    func = run_infer_type(func)
    func = run_infer_type(mnm._ffi.pass_.InlineLet(func))
    func = run_infer_type(mnm._ffi.pass_.DeadCodeElimination(func))
    return func


# TODO(@hzfan): remove this after we have PassContext
def ir_fusion(func):
    """fuse ops"""
    # pylint: disable=protected-access
    func = ir_simplify(func)
    func = run_infer_type(func)
    func = mnm._ffi.pass_.FuseOps(func, 3)
    func = run_infer_type(func)
    return func


def get_vm_executor(model, device, args, optimize=None, sch_file=None):
    """get vm executor"""
    # pylint: disable=protected-access
    record = model._internal(*args)
    func = record.func
    if optimize:
        func = optimize(func)
    mod = Module()
    mod[tvm.ir.GlobalVar('main')] = func
    inputs = _get_func_inputs(record, args, {}, get_handle=False)
    executor = VMExecutor(mod, device)
    return executor.make_executor(sch_file=sch_file), inputs


def get_vm_profiler(model, device, args, optimize=None):
    """get vm profiler"""
    # pylint: disable=invalid-name, protected-access
    record = model._internal(*args)
    func = record.func
    if optimize:
        func = optimize(func)
    mod = Module()
    mod[tvm.ir.GlobalVar('main')] = func
    inputs = _get_func_inputs(record, args, {}, get_handle=False)
    executor = VMProfilerExecutor(mod, device)
    return executor, inputs
