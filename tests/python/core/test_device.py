# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import tvm
from raf._core import device
from raf._core.device import Device
from raf._core.core_utils import DEVICE_TYPE_MAP


def verify_device(dev_str):
    dev = Device(dev_str)
    if dev_str.find("(") != -1:
        dev_type_str = dev_str[: dev_str.find("(")]
        dev_id = int(dev_str[dev_str.find("(") + 1 : dev_str.find(")")])
    else:
        dev_type_str = dev_str
        dev_id = 0
    assert str(dev) == f"Device({dev_type_str}({dev_id}))"
    assert dev.device_type == DEVICE_TYPE_MAP[dev_type_str]
    assert dev.device_id == dev_id
    assert str(dev.tvm_target()) == str(
        tvm.target.Target(dev_type_str if dev_type_str != "cpu" else "llvm")
    )


def test_single_device():
    verify_device("cpu")
    verify_device("cpu(0)")
    verify_device("cuda(1)")

    with pytest.raises(ValueError):
        Device("invalid_string*")

    with pytest.raises(ValueError):
        Device("unsupported_device")

    with pytest.raises(ValueError):
        Device("unsupported_device(1)")

    assert Device("cpu") == Device("cpu(0)")
    assert Device("cpu") != Device("cpu(1)")


def test_helpers():
    assert device.cpu() == Device("cpu")
    assert device.cuda(1) == Device("cuda(1)")


def test_scope():
    assert Device.current() is None
    with Device("cuda(0)"):
        assert Device.current() == Device("cuda(0)")
        with Device("cpu(0)"):
            assert Device.current() == Device("cpu(0)")
        assert Device.current() == Device("cuda(0)")
        with Device("cuda(1)"):
            assert Device.current() == Device("cuda(1)")
        assert Device.current() == Device("cuda(0)")
    assert Device.current() is None


if __name__ == "__main__":
    pytest.main([__file__])
