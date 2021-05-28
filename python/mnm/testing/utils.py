"""Testing utilities for models and executor."""
# pylint: disable=invalid-name,protected-access,too-many-arguments
import mnm
from mnm.model.trace import _get_func_inputs
from mnm._core.executor import VMExecutor, VMCompiler
from mnm._core.profiler_vm import VMProfilerExecutor
from mnm._core.core_utils import get_chained_attr
from .._core.module import IRModule
from .._ffi import pass_


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


# TODO: Remove this after all its use cases are migrated to pass manager with proper requirements
def run_infer_type(expr):
    """Helper function to infer the type of the given expr """
    if isinstance(expr, IRModule):
        return pass_.InferType()(expr)
    mod = IRModule.from_expr(expr)
    mod = pass_.InferType()(mod)
    return mod["main"]


def get_vm_executor(model, device, args, opt_level=2, fuse_level=3, sch_file=None):
    """get vm executor"""
    # pylint: disable=protected-access
    record = model._internal(*args)
    mod = record.mod
    inputs = _get_func_inputs(record, args, {}, get_handle=False)
    with mnm.ir.PassContext(opt_level=opt_level, config={"mnm.fuse_level": fuse_level}):
        executor = VMExecutor(mod, device)
    return executor.make_executor(sch_file=sch_file), inputs


def get_vm_profiler(model, device, args, fuse_level=3):
    """get vm profiler"""
    # pylint: disable=invalid-name, protected-access
    record = model._internal(*args)
    mod = record.mod
    inputs = _get_func_inputs(record, args, {}, get_handle=False)
    with mnm.ir.PassContext(config={"mnm.fuse_level": fuse_level}):
        executor = VMProfilerExecutor(mod, device)
    return executor, inputs


def run_vm_model(model, device, args, opt_level=2, fuse_level=3):
    """Helper function to execute model with VM"""
    vm, inputs = get_vm_executor(model, device, args, opt_level, fuse_level)
    out = vm(*inputs)
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
