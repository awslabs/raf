"""Testing utilities for models and executor."""
# pylint: disable=invalid-name,protected-access,too-many-arguments
import mnm
from mnm.model.trace import _get_func_inputs
from mnm._core.executor import VMExecutor
from mnm._core.vm import VMCompiler
from mnm._core.core_utils import get_chained_attr
from .._core.module import IRModule
from .._ffi import pass_
from .._lib import tvm


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


class DialectChecker(tvm.relay.ExprVisitor):
    """
    Check if all ops in the expr belong to the given dialect list.
    """

    def __init__(self, dialects):
        super(DialectChecker, self).__init__()
        assert isinstance(dialects, (str, list, tuple))
        self.dialects = [dialects] if isinstance(dialects, str) else dialects

    def visit_call(self, call):
        if isinstance(call.op, tvm.relay.Function):
            assert call.op.attrs.Primitive == 1
            assert call.op.attrs.Dialect in self.dialects
        else:  # op
            match = False
            for dialect in self.dialects:
                if dialect in call.op.name:
                    match = True
                    break
            assert match
        super().visit_call(call)


def _get_vm_executor(mod, device, opt_level=2, disable_fusion=False, **options):
    """Get VM executor"""
    # pylint: disable=protected-access
    options.setdefault("stream_schedule_policy", "sequential")
    options.setdefault("sch_file", None)
    options.setdefault("pass_seq", None)
    options.setdefault("reuse_storage", False)

    config = {
        "mnm.stream_schedule.policy": options["stream_schedule_policy"],
        "mnm.memory_plan.reuse_storage": options["reuse_storage"],
    }
    pass_seq = options['pass_seq']
    disabled_pass = []
    if disable_fusion:
        disabled_pass += ["FuseDialect", "FuseTVM"]
    with mnm.ir.PassContext(opt_level=opt_level, config=config, disabled_pass=disabled_pass):
        mod = mnm._ffi.pass_.InferType()(mod)
        if pass_seq is not None:
            mod = pass_seq(mod)
        executor = VMExecutor(mod, device)
    return executor


def get_vm_executor(mod, device, opt_level=2, disable_fusion=False, **options):
    """Get VM executor"""
    executor = _get_vm_executor(mod, device, opt_level, disable_fusion, **options)
    return executor.make_executor(sch_file=options.get('sch_file', None))


def get_vm_profiler(mod, device, opt_level=2, disable_fusion=False, warmup=5, number=10, repeat=10,
                    **options):
    """Get VM Profiler"""
    executor = _get_vm_executor(mod, device, opt_level, disable_fusion, **options)
    return executor.make_profiler(warmup, number, repeat, sch_file=options.get('sch_file', None))


def run_vm_model(model, device, args, opt_level=2, disable_fusion=False, **options):
    """Helper function to execute model with VM"""
    args, kwargs = ([], args) if isinstance(args, dict) else (args, {})
    record = model._internal(*args, **kwargs)
    mod = record.mod
    inputs = _get_func_inputs(record, args, kwargs, get_handle=False)
    vm = get_vm_executor(mod, device, opt_level, disable_fusion, **options)
    out = vm(*inputs)
    return out


def profile_vm_model(model, device, args, opt_level=2, disable_fusion=False, warmup=5, number=10,
                     repeat=10, **options):
    """Helper function to profile model executed by VM"""
    args, kwargs = ([], args) if isinstance(args, dict) else (args, {})
    record = model._internal(*args, **kwargs)
    mod = record.mod
    inputs = _get_func_inputs(record, args, kwargs, get_handle=False)
    vm = get_vm_profiler(mod, device, opt_level, disable_fusion, warmup, number, repeat, **options)
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
