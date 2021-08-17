"""Meta executor."""
# pylint: disable=no-else-return,unidiomatic-typecheck,undefined-variable,invalid-name
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
    module : mnm.ir.IRModule
        The module captures the global variables and functions.

    Returns
    -------
    ret: mnm.value.Value
        Executed results.
    """
    return _ffi.executor.Interpret(expr, module)


class MetaFallbackContext(ApplyHistoryBest):
    """
    The Meta fallback dispatch context, which queries the builtin schedules and outputs
    the message when missed. This is used as the root context for Meta.

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
        fallback_sch_log = ""
        if "MNM_HOME" in os.environ:
            fallback_sch_log = os.path.join(os.environ["MNM_HOME"], "sch/latest.json")
        elif verbose > 0:
            print('Cannot find Meta builtin schedules because "MNM_HOME" is not set')

        if verbose > 0 and not os.path.exists(fallback_sch_log):
            print("Cannot find Meta builtin schedules in %s" % fallback_sch_log)
        super(MetaFallbackContext, self).__init__(fallback_sch_log, include_compatible=True)

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
            msg = (
                f"Cannot find tuned schdule for op={func_name}, target={target}"
            )
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


# pylint: disable=too-few-public-methods
class VMExecutor:
    """
    An implementation of the executor interface for the Meta VM.

    Parameters
    ----------
    mod : :py:class:`~Module`
        The module to support the execution.

    device : str
        The runtime context to run the code on.

    enable_cuda_graph : bool
        Whether to use CUDA graph.
    """

    def __init__(self, mod, device, enable_cuda_graph=False):
        if mod is None:
            raise RuntimeError("Must provide module to get VM executor.")
        if "gpu" not in device and "cuda" not in device:
            enable_cuda_graph = False
        self.mod = mod
        self.device = Device(device)
        self.executable = vm.compile(mod, self.device)
        self.vm = vm.VirtualMachine(self.executable, self.device,
                                    enable_cuda_graph=enable_cuda_graph)
        self.auto_scheduler_fallback_context = None

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
        if self.auto_scheduler_fallback_context is None:
            verbose = int(os.environ["MNM_SCH_VERBOSE"]) if "MNM_SCH_VERBOSE" in os.environ else 2
            self.auto_scheduler_fallback_context = MetaFallbackContext(verbose=verbose)
        auto_scheduler_dispatch_context = \
            auto_scheduler.ApplyHistoryBest(sch_file, include_compatible=True)

        def _vm_wrapper(*args, **kwargs):
            # Backup current configurations
            old_auto_scheduler_fallback_context = auto_scheduler.DispatchContext.current
            old_autotvm_silent = autotvm.GLOBAL_SCOPE.silent

            auto_scheduler.DispatchContext.current = self.auto_scheduler_fallback_context
            autotvm.GLOBAL_SCOPE.silent = True

            with auto_scheduler_dispatch_context:
                with tvm.transform.PassContext(
                        config={"relay.backend.use_auto_scheduler": True},
                        disabled_pass={"AutoSchedulerLayoutRewrite"},
                ):
                    ret = self.vm.run(*args, **kwargs)

            # Recover the configurations
            autotvm.GLOBAL_SCOPE.silent = old_autotvm_silent
            auto_scheduler.DispatchContext.current = old_auto_scheduler_fallback_context
            return ret
        return _vm_wrapper
