"""
The Relay Virtual Machine profiler.

Provides extra APIs for profiling vm execution.
"""
from .. import _ffi
from . import executor


class VirtualMachineProfiler(executor.VirtualMachine):
    """Relay profile VM runtime."""

    def __init__(self, exe, device, enable_cuda_graph=False):
        super(VirtualMachineProfiler, self).__init__(exe, device, enable_cuda_graph)
        self.module = _ffi.vm.VirtualMachineProfiler(exe.module, enable_cuda_graph)
        self._set_devices = self.module["set_devices"]
        self._prepare_context = self.module["prepare_context"]
        self._get_stat = self.module["get_stat"]
        self._reset = self.module["reset"]
        self._run = self.module["run"]
        self._set_devices(device)

    def get_stat(self, sort_by_time=True):
        """Get the statistics of executed ops.

        Parameters
        ----------
        sort_by_time: bool
           Set to indicate the returned results are sorted by execution time in
           the descending order. It will be printed in the random order otherwise.

        Returns
        -------
        ret : str
            The execution statistics in string.
        """
        return self._get_stat(sort_by_time)

    def reset(self):
        """Reset statistics"""
        self._reset()


class VMProfilerExecutor(executor.VMExecutor):
    """
    An implementation of the executor interface for
    the Meta VMProfiler.

    Parameters
    ----------
    mod : :py:class:`~Module`
        The module to support the execution.

    device : str
        The runtime device to run the code on.
    """
    def __init__(self, mod, device, enable_cuda_graph=False):
        super(VMProfilerExecutor, self).__init__(mod, device, enable_cuda_graph)
        self.vm = VirtualMachineProfiler(self.executable, self.device, enable_cuda_graph)

    def reset(self):
        """Reset statistics"""
        self.vm.reset()

    def get_stat(self, sort_by_time=True):
        """Get the statistics of executed ops.

        Parameters
        ----------
        sort_by_time: bool
           Set to indicate the returned results are sorted by execution time in
           the descending order. It will be printed in the random order otherwise.

        Returns
        -------
            The execution statistics in string.
        """
        return self.vm.get_stat(sort_by_time)
