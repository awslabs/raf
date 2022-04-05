# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Testing utilities for models and executor."""
# pylint: disable=invalid-name,protected-access,too-many-arguments
import raf
from raf.model.trace import _get_func_inputs
from raf._core.executor import VMExecutor
from raf._core.vm import VMCompiler
from raf._core.core_utils import get_chained_attr
from .common import check
from .._core.module import IRModule
from .._ffi import pass_
from .._lib import tvm


def get_param(model, name):
    """get parameter from model"""
    if isinstance(name, str):
        name = name.split(".")
    ret = get_chained_attr(model, name)
    if ret is None:
        raise AttributeError(f"No attribute {name}")
    return ret


def set_param(model, name, value):
    """set the value of a parameter"""
    if isinstance(name, str):
        name = name.split(".")
    assert len(name) > 0
    ins = get_param(model, name[:-1])
    setattr(ins, name[-1], value)


# TODO: Remove this after all its use cases are migrated to pass manager with proper requirements
def run_infer_type(expr):
    """Helper function to infer the type of the given expr"""
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
    options.setdefault("anf_only", False)
    options.setdefault("sch_file", None)
    options.setdefault("pass_seq", None)

    config = {
        "raf.stream_schedule.policy": options["stream_schedule_policy"],
        "raf.vm.optimize.anf_only": options["anf_only"],
    }
    pass_seq = options["pass_seq"]
    disabled_pass = []
    if disable_fusion:
        disabled_pass += ["FuseDialect", "FuseTVM"]
    with raf.ir.PassContext(opt_level=opt_level, config=config, disabled_pass=disabled_pass):
        mod = raf._ffi.pass_.InferType()(mod)
        if pass_seq is not None:
            mod = pass_seq(mod)
        executor = VMExecutor(mod, device)
    return executor


def get_vm_executor(mod, device, opt_level=2, disable_fusion=False, **options):
    """Get VM executor"""
    executor = _get_vm_executor(mod, device, opt_level, disable_fusion, **options)
    return executor.make_executor(sch_file=options.get("sch_file", None))


def run_vm_executor(executor, record, args, device):
    """Form VM inputs with inputs and paramters and run the executor."""
    tvm_device = tvm.nd.device(device)
    vm_inputs = _get_func_inputs(record, args, {}, get_handle=False)
    out = executor(*vm_inputs)
    tvm_device.sync()
    return out


def get_vm_profiler(
    mod, device, opt_level=2, disable_fusion=False, warmup=5, number=10, repeat=10, **options
):
    """Get VM Profiler"""
    executor = _get_vm_executor(mod, device, opt_level, disable_fusion, **options)
    return executor.make_profiler(warmup, number, repeat, sch_file=options.get("sch_file", None))


def run_vm_model(model, device, args, opt_level=2, disable_fusion=False, **options):
    """Helper function to execute model with VM"""
    args, kwargs = ([], args) if isinstance(args, dict) else (args, {})
    record = model._internal(*args, **kwargs)
    mod = record.mod
    inputs = _get_func_inputs(record, args, kwargs, get_handle=False)
    vm = get_vm_executor(mod, device, opt_level, disable_fusion, **options)
    out = vm(*inputs)
    return out


def profile_vm_model(
    model,
    device,
    args,
    opt_level=2,
    disable_fusion=False,
    warmup=5,
    number=10,
    repeat=10,
    **options,
):
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
    return mod["main"]


def run_model(model, args, device, check_result=True):
    """Helper function to run the model using both interpreter and VM, and check if their
    results are the same. Note that some ops (e.g., reduce, send/recv) may only produce
    valid results at the target device. In this case, check_result should be skipped on
    other devices.
    """
    out1 = model(*args)
    ret = out1
    out2 = run_vm_model(model, device, args)
    if check_result:
        if not isinstance(out1, (tuple, tvm.ir.container.Array, raf._core.value.TupleValue)):
            out1 = [out1]
            out2 = [out2]
        for o1, o2 in zip(out1, out2):
            try:
                check(o1, o2)
            except AssertionError as e:
                raise AssertionError(
                    "Inconsistent results between interpreter and VM at %s" % device
                ) from e
    return ret
