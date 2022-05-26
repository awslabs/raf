# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for processing TVM ops."""
# pylint: disable=protected-access
import os

import numpy as np

import tvm


@tvm._ffi.register_func("raf._tvm_op.utils.export_library")
def export_library(mod, path):
    """Export a built TVM runtime module to be a shared library (.so) file.

    Parameters
    ----------
    mod : tvm.runtime.Module
        The TVM runtime module to be exported.
    path : str
        The path to the shared library file.

    Returns
    -------
    bool
        Whether the export was successful.
    """
    mod.export_library(path)
    return os.path.exists(path)


@tvm._ffi.register_func("raf._tvm_op.utils.load_module")
def load_module(path):
    """Load a module from a .so file.

    Parameters
    ----------
    path : str
        The path to the .so file.

    Returns
    -------
    tvm.runtime.module.Module
        The loaded module.
    """
    if not os.path.exists(path):
        raise RuntimeError("Module file does not exist {}".format(path))
    return tvm.runtime.module.load_module(path)


def profile_schedule(**params):
    """A lightwight tuner for TOPI schedules. It is similar to AutoTVM but very lightweight.
    It can be used as follows:

    ```python
    @profile_schedule(num_thread=[8, 16, 32, 64])
    def _schedule_cuda(outs, **kwargs):
        num_thread = kwargs.get("num_thread", 32) # Get tuned value or default.
        ...
        return sch

    @schedule_sum.register(["cuda", "gpu"])
    def schedule_cuda(attrs, outs, target):
        with target:
            return _schedule_cuda(outs)
    ```

    The above code snippet profiles 4 schedules with different num_thread and returns
    the best one. Since we directly use tvm.build to compile and evaluate the schedule
    without heavy RFC mechanism and reuse the random data inputs, this is lightwieght
    compared to AutoTVM and auto-schedule and can be used for JIT compilation. However,
    develoeprs should control the tuning space to avoid long JIT time. It is recommended
    to have <10 tuning space when using this function.
    """
    enable = os.environ.get("RAF_JIT_TUNE", False)

    def _wrapper(sch_func):
        def _profile(outs):
            if not enable:
                return sch_func(outs)

            outs = [outs] if isinstance(outs, tvm.te.tensor.Tensor) else outs

            # Collect arguments.
            args_set = set()

            def collect_args(tensor):
                operator = tensor.op
                if isinstance(operator, tvm.te.PlaceholderOp):
                    args_set.add(tensor)
                else:
                    for inp_tensor in operator.input_tensors:
                        collect_args(inp_tensor)

            for out in outs:
                collect_args(out)

            # Generate random input data for profiling.
            tvm_target = tvm.target.Target.current()
            tvm_device = tvm.device(str(tvm_target), 0)  # FIXME
            args = list(args_set)
            args_data = []
            for arg in args:
                shape = [s.value for s in arg.shape]
                args_data.append(
                    tvm.nd.array(np.random.uniform(size=shape).astype(arg.dtype), tvm_device)
                )

            # Profiling
            def profile_param(param_list, param_dict):
                if not param_list:
                    sch = sch_func(outs, **param_dict)
                    try:
                        func = tvm.build(sch, args, tvm_target)
                        evaluator = func.time_evaluator(func.entry_name, tvm_device, number=5)
                        latency = evaluator(*args_data).median
                    except Exception:  # pylint: disable=broad-except
                        latency = float("inf")
                    return sch, latency
                else:
                    best_sch_n_latency = None
                    key, vals = param_list[0]
                    for val in vals:
                        param_dict[key] = val
                        sch, latency = profile_param(param_list[1:], param_dict)
                        if best_sch_n_latency is None or latency < best_sch_n_latency[1]:
                            best_sch_n_latency = (sch, latency)
                    assert best_sch_n_latency is not None
                    return best_sch_n_latency

            sch, _ = profile_param(list(params.items()), {})
            del args_data
            return sch

        return _profile

    return _wrapper
