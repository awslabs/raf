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

import pytest
import tvm
from mnm._core import device
from mnm._core.device import Device
from mnm._core.core_utils import DEVICE_TYPE_MAP


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
