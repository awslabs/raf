"""Testing utilities for models and executor."""
# pylint: disable=invalid-name,protected-access,too-many-arguments
import mnm
from mnm.model.trace import _get_func_inputs
from mnm._core.executor import VMExecutor
from mnm._core.vm import VMCompiler
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


def get_vm_executor(mod, device, opt_level=2, fuse_level=3, **options):
    """Get VM executor"""
    # pylint: disable=protected-access
    options.setdefault("stream_schedule_policy", "sequential")
    options.setdefault("sch_file", None)
    options.setdefault("pass_seq", None)
    options.setdefault("reuse_storage", False)

    config = {
        "mnm.fuse_level": fuse_level,
        "mnm.stream_schedule.policy": options["stream_schedule_policy"],
        "mnm.memory_plan.reuse_storage": options["reuse_storage"],
    }
    pass_seq = options['pass_seq']
    with mnm.ir.PassContext(opt_level=opt_level, config=config):
        mod = mnm._ffi.pass_.InferType()(mod)
        if pass_seq is not None:
            mod = pass_seq(mod)
        executor = VMExecutor(mod, device)
    return executor.make_executor(sch_file=options['sch_file'])


def run_vm_model(model, device, args, opt_level=2, fuse_level=3, **options):
    """Helper function to execute model with VM"""
    args, kwargs = ([], args) if isinstance(args, dict) else (args, {})
    record = model._internal(*args, **kwargs)
    mod = record.mod
    inputs = _get_func_inputs(record, args, kwargs, get_handle=False)
    vm = get_vm_executor(mod, device, opt_level, fuse_level, **options)
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
