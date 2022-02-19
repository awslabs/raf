# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""The Meta VM debug tools."""
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
    An implementation of the executor interface for the Meta VmDryRunner and VmDebugger.

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
