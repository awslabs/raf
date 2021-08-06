"""The Meta VM debug tools."""
# pylint: disable=super-init-not-called,too-few-public-methods
from .. import _ffi
from . import executor
from . import vm

class VMMemoryProfiler(vm.VirtualMachine):
    """VM memory profiler."""
    def __init__(self, exe, device):
        self.module = _ffi.vm.VMMemoryProfiler(exe.module)
        self._exec = exe
        self._set_devices = self.module["set_devices"]
        self._prepare_context = self.module["prepare_context"]
        self._trace = self.module["trace"]
        self._get_result = self.module["get_result"]
        self._set_devices(device)

    def trace(self, *args, func_name="main", **kwargs):
        """Trace the memory footprint.

        Parameters
        ----------
        args : list[mnm.ndarray] or list[np.ndarray]
            The arguments to the function.

        func_name : str
            The name of function to run.

        kwargs: dict of str to mnm.ndarray or np.ndarray
            Named arguments to the function.

        Returns
        -------
        result : Dict[str, FloatImm]
            The per device memory footprint in MBs.
        """
        # pylint: disable=arguments-differ
        ctx = self.prepare_context(func_name, *args, **kwargs)
        return self._trace(ctx)

    run = trace

    def get_result(self):
        """Get the memory trace result.

        Returns
        -------
        ret: str
            The memory trace that shows the instant memory footprint over op execution.
        """
        return self._get_result()

class VMDebugger(vm.VirtualMachine):
    """VM debugger to debug the intermediate results."""
    def __init__(self, exe, device):
        self.module = _ffi.vm.VMDebugger(exe.module)
        self._exec = exe
        self._set_devices = self.module["set_devices"]
        self._prepare_context = self.module["prepare_context"]
        self._run = self.module["run"]
        self._get_interm_tensors = self.module["get_interm_tensors"]
        self._reset = self.module["reset"]
        self._set_devices(device)

    def get_interm_tensors(self):
        """Get the intermediate results.

        Returns
        -------
        ret : Array[Str], Array[Array[Value]], Array[Value]
            op names, op inputs, op outputs
        """
        res = self._get_interm_tensors()
        names, ins, outs = res["names"], res["inputs"], res["outputs"]
        return names, ins, outs

    def reset(self):
        """Reset the states."""
        self._reset()

class VMDebugExecutor(executor.VMExecutor):
    """
    An implementation of the executor interface for the Meta VmMemoryProfiler and VmDebugger.

    Parameters
    ----------
    mod : :py:class:`~Module`
        The module to support the execution.

    device : str
        The runtime device to run the code on.

    option : str
        Available options:
        - "profile_memory": use to trace memory
        - "debug": use to debug the intermediate results
    """
    def __init__(self, mod, device, option):
        super(VMDebugExecutor, self).__init__(mod, device)
        assert option in ["profile_memory", "debug"], "Unknown option: %s" % option
        if option == "profile_memory":
            self.vm = VMMemoryProfiler(self.executable, self.device)
            self.get_result = self.vm.get_result
        else: # debug
            self.vm = VMDebugger(self.executable, self.device)
            self.get_interm_tensors = self.vm.get_interm_tensors
            self.reset = self.vm.reset
