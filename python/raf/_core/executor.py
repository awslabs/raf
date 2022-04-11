# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""RAF executor."""
# pylint: disable=no-else-return,unidiomatic-typecheck,undefined-variable,invalid-name
# pylint: disable=protected-access
import os
import tvm
from tvm import auto_scheduler, autotvm
from tvm.auto_scheduler.dispatcher import ApplyHistoryBest
from .. import _ffi
from . import vm
from .device import Device


def interpret(expr, module=None):
    """use interpreter to execute the program.

    Parameters
    ----------
    expr : relay.Call
        The function together with its arguments.
    module : raf.ir.IRModule
        The module captures the global variables and functions.

    Returns
    -------
    ret: raf.value.Value
        Executed results.
    """
    return _ffi.executor.Interpret(expr, module)


class MetaFallbackContext(ApplyHistoryBest):
    """
    The RAF fallback dispatch context, which queries the builtin schedules and outputs
    the message when missed. This is used as the root context for RAF.

    Parameters
    ----------
    verbose: int
        The verbose level (0-3) for missing schedules. Default 2.
        verbose 0 disables all messages; verbose 1 only shows the warning when
        builtin schedule files are missing; verbose 2 prints warnings for ops
        that missed schedules; verbose 3 is based on verbose 2 but with the compute DAG of the op.
    """

    def __init__(self, verbose=2):
        # Load the builtin schedules
        fallback_sch_log = None
        if "RAF_SCH_FILE" in os.environ and os.path.exists(os.environ["RAF_SCH_FILE"]):
            fallback_sch_log = os.environ["RAF_SCH_FILE"]

        if verbose > 0:
            if fallback_sch_log is not None:
                print(f"RAF schedule file is pointed to {fallback_sch_log}")
            else:
                print('No pretuned schedules because "RAF_SCH_FILE" is not set or does not exist')

        super().__init__(fallback_sch_log, include_compatible=True)

        self.verbose = verbose

        # The schedule missing message memory to avoid duplications.
        self.memory = set()

    def query(self, target, workload_key, has_complex_op, dag, func_name):
        # pylint: disable=too-many-arguments
        # Query the builtin schedules.
        ret = self._query_inside(target, workload_key, func_name)
        if ret is not None:
            return ret

        key = (str(target), workload_key)
        if key in self.memory:
            return None

        # Print the message due to no valid schedule.
        if has_complex_op and self.verbose >= 2:
            msg = f"Cannot find tuned schedule for op={func_name}, target={target}"
            if self.verbose == 3:
                msg += (
                    f", workload_key={workload_key}\n"
                    f"Compute DAG info:\n{dag}"
                    f"-----------------------------------\n"
                )
            if msg not in self.memory:
                self.memory.add(msg)
                print(msg)

        self.memory.add(key)
        return None


def init_auto_scheduler_dispatch_context():
    """Initialize auto scheduler dispatch context."""
    verbose = int(os.environ["RAF_SCH_VERBOSE"]) if "RAF_SCH_VERBOSE" in os.environ else 0
    env = MetaFallbackContext(verbose=verbose)
    env.__enter__()


init_auto_scheduler_dispatch_context()

# pylint: disable=too-few-public-methods
class VMExecutor:
    """
    An implementation of the executor interface for the RAF VM.

    Parameters
    ----------
    mod : :py:class:`~Module`
        The module to support the execution.

    device : str
        The runtime context to run the code on.

    enable_cuda_graph : bool
        Whether to use CUDA graph.

    dryrun: bool
        Whether to create a dryrun VM that skips the op execution.
    """

    def __init__(self, mod, device, enable_cuda_graph=False, dryrun=False):
        if mod is None:
            raise RuntimeError("Must provide module to get VM executor.")
        if "gpu" not in device and "cuda" not in device:
            enable_cuda_graph = False
        self.device = Device(device)
        self.executable = vm.compile(mod, self.device)
        self.vm = vm.VirtualMachine(
            self.executable, self.device, enable_cuda_graph=enable_cuda_graph, dryrun=dryrun
        )

    @staticmethod
    def _make_vm_helper(maker, sch_file=None):
        """
        Get a wrapper that runs given maker function. The wrapper would configure the relay auto
        scheduler to use the tuning records in given schedule file.

        Parameters
        ----------
        maker : Callable
            The maker function to be wrapped.

        sch_file : str
            The schedule file that contains the tuning records.

        Returns
        -------
        result: Callable
            The wrapped function.
        """
        auto_scheduler_dispatch_context = auto_scheduler.ApplyHistoryBest(
            sch_file, include_compatible=True
        )

        def _vm_wrapper(*args, **kwargs):
            # Backup current configurations
            old_autotvm_silent = autotvm.GLOBAL_SCOPE.silent
            autotvm.GLOBAL_SCOPE.silent = True

            with auto_scheduler_dispatch_context:
                with tvm.transform.PassContext(
                    config={"relay.backend.use_auto_scheduler": True},
                    disabled_pass={"AutoSchedulerLayoutRewrite"},
                ):
                    ret = maker(*args, **kwargs)

            # Recover the configurations
            autotvm.GLOBAL_SCOPE.silent = old_autotvm_silent
            return ret

        return _vm_wrapper

    def make_profiler(self, warmup=5, number=10, repeat=10, sch_file=None):
        """Create a VM profiler that measure the latency of model execution.

        It uses the following procedure to measure the end-to-end latency of the model:
        ```
            Warmup the model by running `warmup` times.
            results = []
            for i in [0, repeat):
              sync()
              t1 = cur_time()
              for j in [0, number):
                run the model
              sync()
              t2 = cur_time()
              results.append((t2 - t1) / number)
            return results
        ```
        where sync() would do a device synchronization, and cur_time() get the current time.

        To measure the average latency of N runs, we can use whether `number=1, repeat=N` or
        `number=N, repeat=1`, where the former one would synchronize the device for each run, and
        the latter one would not.

        Parameters
        ----------
        warmup : int
            The number of runs used to warmup. Default 5.

        number : int
            The number of times to run the generated code for taking average. We call these runs as
            one repeat of measurement. Default 10.

        repeat : int
            The number of times to repeat the measurement. In total, the generated code will be run
            (warmup + number x repeat) times, where the first “warmup” results will be discarded.
            The returned result contains repeat costs, each of which is an average of number costs.
            Default 10.

        sch_file: Optional[str]
            The tuned schedule file path.

        Returns
        -------
        result : List[float]
            The list of latency for each repeat in milliseconds, where len(result) == repeat.
        """

        def _maker(*args, **kwargs):
            return self.vm.profile(*args, **kwargs, warmup=warmup, number=number, repeat=repeat)

        return self._make_vm_helper(_maker, sch_file)

    def make_executor(self, sch_file=None):
        """Create a VM executor.

        Parameters
        ----------
        sch_file: Optional[str]
            The tuned schedule file path.

        Returns
        -------
        executor: Callable
            The VM executor
        """

        def _maker(*args, **kwargs):
            return self.vm.run(*args, **kwargs)

        return self._make_vm_helper(_maker, sch_file)
