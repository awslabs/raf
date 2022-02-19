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

# pylint: disable=invalid-name,protected-access,no-self-use
import pytest
import mnm
from mnm._ffi import pass_
from mnm.testing import randn
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
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return mnm.relu(x + x)

    model = Model()
    m_x, _ = randn((3, 4))
    mod = model._internal(m_x).mod
    mod = optimize(mod)
    assert not check_inline(mod["main"])

    mod = pass_.InlinePrimitives()(mod)
    assert check_inline(mod["main"])


if __name__ == "__main__":
    pytest.main([__file__])
