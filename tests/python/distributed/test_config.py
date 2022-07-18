# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from raf import distributed as dist


def test_dumps():
    dcfg = dist.get_config()
    expected = {"enable_data_parallel": 0, "zero_opt_level": 0, "enable_auto_dp_profiling": 0}
    actual = dcfg.dumps()
    for key, val in expected.items():
        assert actual[key] == val


if __name__ == "__main__":
    pytest.main([__file__])
