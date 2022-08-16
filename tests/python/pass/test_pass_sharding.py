# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring, missing-class-docstring, invalid-name, protected-access
import pytest
from raf.distributed.sharding import make_replicated_spec, make_shard_spec, make_unset_spec
from tvm.ir import structural_equal


def test_shardspec():
    a = make_shard_spec([4], ranks=4)
    b = make_shard_spec([4], ranks=4)
    assert structural_equal(a, b)

    c = make_shard_spec([2, 2], ranks=4)
    assert not structural_equal(a, c)

    d = make_shard_spec([4], ranks=8)
    assert not structural_equal(a, d)

    e = make_unset_spec()
    f = make_unset_spec()
    assert structural_equal(e, f)
    assert not structural_equal(a, e)

    g = make_shard_spec([4], [4], ranks=4)
    h = make_replicated_spec(ndim=1, ranks=4)
    assert not structural_equal(a, g)
    assert structural_equal(g, h)

    i = make_shard_spec([4], ranks=4, mutable=False)
    assert not structural_equal(a, i)


if __name__ == "__main__":
    pytest.main([__file__])
