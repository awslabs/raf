# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The RAF VM debug tools."""
# pylint: disable=super-init-not-called,too-few-public-methods
from .. import _ffi
from . import executor
from . import vm


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
    An implementation of the executor interface for the RAF VmDryRunner and VmDebugger.

    Parameters
    ----------
    mod : :py:class:`~Module`
        The module to support the execution.

    device : str
        The runtime device to run the code on.
    """

    def __init__(self, mod, device):
        super(VMDebugExecutor, self).__init__(mod, device)
        self.vm = VMDebugger(self.executable, self.device)
        self.get_interm_tensors = self.vm.get_interm_tensors
        self.reset = self.vm.reset
