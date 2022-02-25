# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name, protected-access
import pytest
import raf


def test_match_constant():
    c = raf.ir.const(1)
    value = raf._core.value.IntValue(1)
    pat = raf.ir.dataflow_pattern.is_constant(value)
    assert raf.ir.dataflow_pattern.match(pat, c)


def test_no_match_constant():
    c = raf.ir.const(1.0)
    value = raf._core.value.IntValue(1)
    pat = raf.ir.dataflow_pattern.is_constant(value)
    assert not raf.ir.dataflow_pattern.match(pat, c)


if __name__ == "__main__":
    pytest.main([__file__])
