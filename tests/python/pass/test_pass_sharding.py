# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring, missing-class-docstring, invalid-name, protected-access
import raf
import pytest
import numpy as np
from raf.distributed.sharding import (
    ShardSpec,
    BaseShardSpec,
    ShardOpCallAttrs,
)
from raf._ffi.pass_ import (
    AnnotateShardOpCall,
    ToGraphNormalForm,
    ExpandShardOpCall,
    InferType,
    InferShardSpec,
)
from raf._lib import relay
from raf.distributed.sharding import make_replicated_spec, make_shard_spec, make_unset_spec
from tvm.ir import structural_equal
from tvm.relay.analysis.analysis import post_order_visit


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

def test_infer_hint_without_prev_spec():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            z = raf.add(x, y)
            a = raf.relu(z)
            b = raf.relu(a)
            return b

    model = Model()
    m_x = raf.array(np.arange(16, dtype="float").reshape((4, 4)))
    m_y = raf.array(np.zeros(16, dtype="float").reshape((4, 4)))
    record = model._internal(m_x, m_y)
    mod_before = record.mod
    mod_before = InferType()(mod_before)

    print(m_x)
    call_list = []
    post_order_visit(
        mod_before["main"].body,
        lambda op: call_list.append(op) if isinstance(op, relay.Call) else None,
    )

    attrs_map = {
        call_list[1]: ShardOpCallAttrs([make_unset_spec()], [make_shard_spec([4, 1], ranks=4, mutable=False)]),
        call_list[2]: ShardOpCallAttrs([make_unset_spec()], [make_replicated_spec(2, mutable=False)])
    }

    mod0 = AnnotateShardOpCall(attrs_map)(mod_before)
    mod1 = ToGraphNormalForm()(mod0)
    mod2 = InferType()(mod1)
    print("after 1st infer type")
    print(raf._ffi.ir.AsText(mod2))

    mod3 = InferShardSpec()(mod2)
    print("after infer shard spec")
    print(raf._ffi.ir.AsText(mod3))

    mod4 = InferType()(mod3)
    print("after 2nd infer type")
    print(raf._ffi.ir.AsText(mod4))

    mod5 = ExpandShardOpCall()(mod4)
    print("after expand shard opcall")
    print(raf._ffi.ir.AsText(mod5))

def test_infer_hint_inserting_reshard():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            z = raf.add(x, y)
            a = raf.relu(z)
            return a

    model = Model()
    m_x = raf.array(np.arange(16, dtype="float").reshape((4, 4)))
    m_y = raf.array(np.zeros(16, dtype="float").reshape((4, 4)))
    record = model._internal(m_x, m_y)
    mod_before = record.mod
    mod_before = InferType()(mod_before)

    print(m_x)
    call_list = []
    post_order_visit(
        mod_before["main"].body,
        lambda op: call_list.append(op) if isinstance(op, relay.Call) else None,
    )

    spec = make_shard_spec([2, 2], [1, 2], 4, mutable=False)

    attrs_map = {
        call_list[0]: ShardOpCallAttrs([make_unset_spec(), make_unset_spec()], [make_unset_spec()]),
        call_list[1]: ShardOpCallAttrs([make_unset_spec()], [spec]),
    }

    mod0 = AnnotateShardOpCall(attrs_map)(mod_before)
    mod1 = ToGraphNormalForm()(mod0)
    mod2 = InferType()(mod1)
    print("after 1st infer type")
    print(raf._ffi.ir.AsText(mod2))

    mod3 = InferShardSpec()(mod2)
    print("after infer shard spec")
    print(raf._ffi.ir.AsText(mod3))

    mod4 = InferType()(mod3)
    print("after 2nd infer type")
    print(raf._ffi.ir.AsText(mod4))

    mod5 = ExpandShardOpCall()(mod4)
    print("after expand shard opcall")
    print(raf._ffi.ir.AsText(mod5))


if __name__ == "__main__":
    test_infer_hint_inserting_reshard()
    # test_infer_hint_without_prev_spec()
    # pytest.main([__file__])
