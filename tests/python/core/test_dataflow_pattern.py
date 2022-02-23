# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name, protected-access
import pytest
import mnm


def test_match_constant():
    c = mnm.ir.const(1)
    value = mnm._core.value.IntValue(1)
    pat = mnm.ir.dataflow_pattern.is_constant(value)
    assert mnm.ir.dataflow_pattern.match(pat, c)


def test_no_match_constant():
    c = mnm.ir.const(1.0)
    value = mnm._core.value.IntValue(1)
    pat = mnm.ir.dataflow_pattern.is_constant(value)
    assert not mnm.ir.dataflow_pattern.match(pat, c)


if __name__ == "__main__":
    pytest.main([__file__])
