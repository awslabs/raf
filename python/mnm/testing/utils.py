"""testing utilities for models"""
import mnm
from mnm.ir import MNMSequential
from mnm.testing import run_infer_type
from mnm.model.trace import _get_func_inputs
from mnm._core.executor import VMExecutor
from mnm._core.profiler_vm import VMProfilerExecutor
from mnm._core.core_utils import get_chained_attr
from mnm._ffi import pass_


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
def ir_simplify(mod):
    # pylint: disable=protected-access
    """simplify mod by simplifying tuples and eliminating dead code"""
    seq = MNMSequential([pass_.InferType(), pass_.InlineLet(),
                         pass_.InferType(),
                         pass_.DeadCodeElimination(), pass_.InferType()])
    return seq(mod)


# TODO(@hzfan): remove this after we have PassContext
def ir_fusion(mod, fuse_opt_level=3):
    """fuse ops"""
    # pylint: disable=protected-access
    mod = ir_simplify(mod)
    mod = run_infer_type(mod)
    mod = mnm._ffi.pass_.FuseOps(mod, fuse_opt_level)
    mod = run_infer_type(mod)
    return mod


def get_vm_executor(model, device, args, optimize=None, sch_file=None):
    """get vm executor"""
    # pylint: disable=protected-access
    record = model._internal(*args)
    mod = record.mod
    if optimize:
        mod = optimize(mod)
    inputs = _get_func_inputs(record, args, {}, get_handle=False)
    executor = VMExecutor(mod, device)
    return executor.make_executor(sch_file=sch_file), inputs


def get_vm_profiler(model, device, args, optimize=None):
    """get vm profiler"""
    # pylint: disable=invalid-name, protected-access
    record = model._internal(*args)
    mod = record.mod
    if optimize:
        mod = optimize(mod)
    inputs = _get_func_inputs(record, args, {}, get_handle=False)
    executor = VMProfilerExecutor(mod, device)
    return executor, inputs
