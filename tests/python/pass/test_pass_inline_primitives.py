# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name,protected-access,no-self-use
import pytest
import raf
from raf._ffi import pass_
from raf.testing import randn
from tvm import relay


class InlineChecker(relay.ExprVisitor):
    def __init__(self):
        super(InlineChecker, self).__init__()
        self.flag = True

    def visit_let(self, let):
        val = let.value
        if isinstance(val, relay.Function) and val.attrs and val.attrs.Primitive == 1:
            self.flag = False
        super().visit_let(let)


def check_inline(func):
    checker = InlineChecker()
    checker.visit(func)
    return checker.flag


def optimize(mod):
    mod = pass_.ToGraphNormalForm()(mod)
    mod = pass_.ToBasicBlockNormalForm()(mod)
    mod = pass_.FuseTVM()(mod)
    mod = pass_.ToANormalForm()(mod)
    mod = pass_.InferType()(mod)
    return mod


def test_simple():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf.relu(x + x)

    model = Model()
    m_x, _ = randn((3, 4))
    mod = model._internal(m_x).mod
    mod = optimize(mod)
    assert not check_inline(mod["main"])

    mod = pass_.InlinePrimitives()(mod)
    assert check_inline(mod["main"])


if __name__ == "__main__":
    pytest.main([__file__])
