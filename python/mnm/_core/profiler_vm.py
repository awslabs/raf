"""
The Relay Virtual Machine profiler.

Provides extra APIs for profiling vm execution.
"""
# pylint: disable=too-many-instance-attributes
from .. import _ffi
from . import executor


class VirtualMachineProfiler(executor.VirtualMachine):
    """Relay profile VM runtime."""

    def __init__(self, exe, device, enable_cuda_graph=False, cache_interm_tensors=False):
        super(VirtualMachineProfiler, self).__init__(exe, device, enable_cuda_graph)
        self.module = _ffi.vm.VirtualMachineProfiler(exe.module, enable_cuda_graph,
                                                     cache_interm_tensors)
        self._set_devices = self.module["set_devices"]
        self._prepare_context = self.module["prepare_context"]
        self._get_stat = self.module["get_stat"]
        self._get_interm_tensors = self.module["get_interm_tensors"]
        self._profile_memory = self.module["profile_memory"]
        self._reset = self.module["reset"]
        self._run = self.module["run"]
        self._set_devices(device)

    def get_stat(self, sort_by_time=True, show_shape=True):
        """Get the statistics of executed ops.

        Parameters
        ----------
        sort_by_time: bool
            Set to indicate the returned results are sorted by execution time in
            the descending order. It will be printed in the random order otherwise.

        show_shape: bool
            Whether to display input and output shapes of executed ops. Default True.

        Returns
        -------
        ret : str
            The execution statistics in string.
        """
        return self._get_stat(sort_by_time, show_shape)

    def get_interm_tensors(self):
        """Get the intermediate results

        Returns
        -------
        ret : Array[Str], Array[Array[Value]], Array[Value]
            op names, op inputs, op outputs
        """
        res = self._get_interm_tensors()
        names, ins, outs = res["names"], res["inputs"], res["outputs"]
        return names, ins, outs

    def reset(self):
        """Reset statistics"""
        self._reset()

    def profile_memory(self, *args, func_name="main", **kwargs):
        """Profile the total memory footprint in MBs. Note that we now
        only profile memory allocated by AllocStorage bytecode instruction.
        The memory allocated for op workspace is ignored for simplify.
        Since workspace allocation is relatively rare, the impact on
        the overall memory footprint is limited.

        Parameters
        ----------
        args : list[mnm.ndarray] or list[np.ndarray]
            The arguments to the function.

        func_name : str
            The name of function.

        kwargs: dict of str to mnm.ndarray or np.ndarray
            Named arguments to the function.

        Returns
        -------
        result : float
            The total memory footprint in MBs.
        """
        ctx = self.prepare_context(func_name, *args, **kwargs)
        return self._profile_memory(ctx)


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

    enable_cuda_graph : bool
        Whether to enable cuda graph
    """
    def __init__(self, mod, device, enable_cuda_graph=False, cache_interm_tensors=False):
        super(VMProfilerExecutor, self).__init__(mod, device, enable_cuda_graph)
        self.vm = VirtualMachineProfiler(self.executable, self.device, enable_cuda_graph,
                                         cache_interm_tensors)

    def reset(self):
        """Reset statistics"""
        self.vm.reset()

    def get_stat(self, sort_by_time=True, show_shape=True):
        """Get the statistics of executed ops.

        Parameters
        ----------
        sort_by_time: bool
            Set to indicate the returned results are sorted by execution time in
            the descending order. It will be printed in the random order otherwise.

        show_shape: bool
            Whether to display input and output shapes of executed ops. Default True.

        Returns
        -------
            The execution statistics in string.
        """
        return self.vm.get_stat(sort_by_time, show_shape)

    def get_interm_tensors(self):
        """Get the intermediate results

        Returns
        -------
        ret : Array[Str], Array[Array[Value]], Array[Value]
            op names, op inputs, op outputs
        """
        return self.vm.get_interm_tensors()

    def profile_memory(self, *args, func_name="main", **kwargs):
        """Profile the total memory footprint in MBs. Note that we now
        only profile memory allocated by AllocStorage bytecode instruction.
        The memory allocated for op workspace is ignored for simplify.
        Since workspace allocation is relatively rare, the impact on
        the overall memory footprint is limited.

        Parameters
        ----------
        args : list[mnm.ndarray] or list[np.ndarray]
            The arguments to the function.

        func_name : str
            The name of function.

        kwargs: dict of str to mnm.ndarray or np.ndarray
            Named arguments to the function.

        Returns
        -------
        result : float
            The total memory footprint in MBs.
        """
        return self.vm.profile_memory(*args, func_name=func_name, **kwargs)
