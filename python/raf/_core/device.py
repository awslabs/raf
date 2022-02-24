# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device."""
import re

from .core_utils import register_node, DEVICE_TYPE_MAP
from .._lib import Object
from .._ffi import device as ffi


@register_node("raf.device.Device")
class Device(Object):
    """Construct a Device object.

    Parameters
    ----------
    device_str: str
        The device string such as "cpu", "cuda", or the device string with ID,
        such as "cuda(1)".
    """

    def __init__(self, device_str):
        tokens = re.search(r"(\w+).?(\d?)", device_str)
        if not tokens or len(tokens.groups()) != 2:
            raise ValueError("Invalid device string: %s" % device_str)

        # Process device type.
        device_type_str = tokens.groups()[0]

        if device_type_str not in DEVICE_TYPE_MAP:
            raise ValueError(
                "Unrecognized device type: %s. Supported types:\n%s"
                % (device_type_str, ",".join(DEVICE_TYPE_MAP.keys()))
            )
        device_type = DEVICE_TYPE_MAP[device_type_str]

        # Process device ID.
        device_id = int(tokens.groups()[1]) if tokens.groups()[1] else 0

        self.__init_handle_by_constructor__(ffi.Device, device_type, device_id)

    def tvm_target(self):
        """Get the corresponding TVM target object of this device.

        Returns
        -------
        target: tvm.target.Target
            The TVM target object of this device.
        """
        return ffi.GetTVMTarget(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __enter__(self):
        ffi.DeviceEnterScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        ffi.DeviceExitScope(self)

    @staticmethod
    def current(allow_none=True):
        """Returns the current device.

        Parameters
        ----------
        allow_none : bool
            When the current device is not set, return None if True;
            otherwise throw an error.

        Returns
        -------
        device: Union[Device, None]
            Return the current device, or None if the current device is not set
            and allow_none=True.
        """
        dev = ffi.DeviceCurrent(allow_none)
        return dev if dev.device_type != 0 or dev.device_id != -1 else None


def device(device_str):
    """Create a device."""
    return Device(device_str)


def cpu(device_id=0):
    """Create a CPU device object."""
    return Device(f"cpu({device_id})")


def cuda(device_id=0):
    """Create a CPU device object."""
    return Device(f"cuda({device_id})")
