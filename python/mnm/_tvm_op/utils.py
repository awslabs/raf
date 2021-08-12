"""Utilities for TVM ops."""
from mnm._lib import tvm
from tvm import relay
from tvm.auto_scheduler.compute_dag import ComputeDAG
from tvm.auto_scheduler.relay_integration import traverse_to_get_io_tensors

@tvm._ffi.register_func("mnm._tvm_op.utils.calc_flops") # pylint: disable=protected-access
def calc_flops(func, tvm_target):
    """Calculate the compute FLOPS of the given call node.
    """
    engine = relay.backend.compile_engine.get()
    cached_func = engine.lower(func, tvm_target)

    io_tensors, _, _ = traverse_to_get_io_tensors(cached_func.outputs)
    dag = ComputeDAG(io_tensors)
    return int(dag.flop_ct)
