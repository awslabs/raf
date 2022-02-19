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

# pylint: disable=attribute-defined-outside-init,no-self-use,protected-access
import pytest
import numpy as np

import mnm
from mnm.testing import randn, get_testable_devices, check, run_vm_model
from mnm._core.ndarray import ndarray
from mnm._core.core_utils import get_chained_attr
from mnm._op import sym
from mnm.model.trace import trace_mutate_attr
from tvm import relay


@pytest.mark.parametrize("device", get_testable_devices())
def test_tup_inputs(device):
    class MNMTupleTest(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, tup):
            x = mnm.add(tup[0], tup[1])
            return x

    shape = (2, 2)
    m_model = MNMTupleTest()
    m_x, n_x = randn(shape, device=device)
    m_y, n_y = randn(shape, device=device)
    m_z = m_model((m_x, m_y))
    n_z = n_x + n_y
    check(m_z, n_z)
    m_z = run_vm_model(m_model, device, [(m_x, m_y)])
    check(m_z, n_z)


# TODO(@hzfan): use BatchNorm in python/mnm/model/nn.py
class BatchNorm(mnm.Model):
    # pylint: disable=attribute-defined-outside-init, too-many-instance-attributes
    def build(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.running_var = self.running_mean = None
        if affine:
            self.w = self.b = None
        self.reset()

    def reset(self):
        n_f = self.num_features
        self.running_mean = ndarray(
            np.zeros(n_f, dtype="float32"),
            name="running_mean",
            device=get_chained_attr(self, ["running_mean", "device"], "cpu"),
        )
        self.running_var = ndarray(
            np.ones(n_f, dtype="float32"),
            name="running_var",
            device=get_chained_attr(self, ["running_var", "device"], "cpu"),
        )
        if self.affine:
            self.w = ndarray(
                np.ones(n_f, dtype="float32"),
                name="w",
                device=get_chained_attr(self, ["w", "device"], "cpu"),
            )
            self.b = ndarray(
                np.zeros(n_f, dtype="float32"),
                name="b",
                device=get_chained_attr(self, ["b", "device"], "cpu"),
            )

    # pylint: enable=attribute-defined-outside-init

    @mnm.model.trace
    def forward(self, x):
        ret = sym.batch_norm_train(
            x=x,
            w=self.w,
            b=self.b,
            running_mean=self.running_mean,
            running_var=self.running_var,
            eps=self.eps,
            momentum=self.momentum,
        )
        trace_mutate_attr(self, "running_mean", ret[1])
        trace_mutate_attr(self, "running_var", ret[2])
        return ret[0]

    @mnm.model.trace
    def forward_infer(self, x):
        ret = sym.batch_norm_infer(
            x=x,
            w=self.w,
            b=self.b,
            running_mean=self.running_mean,
            running_var=self.running_var,
            eps=self.eps,
            momentum=self.momentum,
        )
        return ret


def test_mutate_bn():
    # pylint: disable=line-too-long
    # fn (%x: Tensor[(2, 3, 4, 5), float32], %batch_norm.b: Tensor[(3), float32], %batch_norm.running_mean: Tensor[(3), float32], %batch_norm.running_var: Tensor[(3), float32], %batch_norm.w: Tensor[(3), float32]) {
    #     let %a1 = mnm.op.batch_norm_train(%x, %batch_norm.running_mean, %batch_norm.running_var, %batch_norm.w, %batch_norm.b, -114514, -114514);
    #     let %a2 = %a1.0;
    #     let %a3 = mnm.op.relu(%a2);
    #     let %a4 = %a1.1;
    #     let %a5 = %a1.2;
    #     let %a6 = (%a3, %a4, %a5);
    #     %a6
    # }
    class Test(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self, num_features, eps=1e-5, momentum=0.1, affine=True):
            self.batch_norm = BatchNorm(num_features, eps, momentum, affine)

        @mnm.model.trace
        def forward(self, x):
            x = self.batch_norm(x)
            x = mnm.relu(x)
            return x

    shape = (2, 3, 4, 5)
    dtype = "float32"
    data = mnm.array(np.ones(shape), dtype=dtype)
    model = Test(num_features=3)
    mod = model._internal(data).mod
    mod = mnm._ffi.pass_.InferType()(mod)
    ret_type = mod["main"].checked_type.ret_type
    assert isinstance(ret_type, relay.ty.TupleType)
    assert len(ret_type.fields) == 3


def test_mutate_grad():
    # pylint: disable=too-many-locals, too-many-arguments, attribute-defined-outside-init
    class Model(mnm.Model):
        def build(self, shape):
            self.shape = shape
            self.reset()

        def reset(self):
            self.x = mnm.array(np.random.randn(*self.shape))

        @mnm.model.trace
        def forward(self):  # pylint: disable=no-self-use
            return mnm.relu(self.x)

    # fn (%dy: Tensor[(2, 3, 4), float64], %model.x: Tensor[(2, 3, 4), float64]) {
    #     let %a1 = mnm.op.relu(%model.x);
    #     let %a2 = mnm.op.relu_dx(%model.x, %a1, %dy);
    #     let %a3 = mnm.op.subtract(%model.x, %a2, -114514, -114514);
    #     let %a4 = (%a1, %a3);
    #     %a4
    # }
    class SGD(mnm.Model):
        def build(self, model):
            self.model = model

        def reset(self):
            self.model.reset()

        @mnm.model.trace
        def forward(self, dy):
            out = self.model()
            # grad of model, which will be replaced by AutoDiff
            dx = mnm.relu_dx(self.model.x, out, dy)
            # update params
            new_x = mnm.subtract(self.model.x, dx)
            trace_mutate_attr(self.model, "x", new_x)
            return out

    shape = [2, 3, 4]
    param = np.random.randn(*shape)
    dy = mnm.array(np.random.randn(*shape))
    model = Model(shape)
    sgd = SGD(model)
    # IR with SGD
    model.x = mnm.array(param)
    out_1 = sgd(dy)
    new_x_1 = model.x
    # IR without SGD
    model.x = mnm.array(param)
    model.x.requires_grad = True
    out_2 = model()
    out_2.backward(dy)
    new_x_2 = mnm.subtract(model.x, model.x.grad)
    # check forward
    check(out_1, out_2)
    # check backward
    check(new_x_1, new_x_2)
    # check ir
    mod = sgd._internal(dy).mod
    mod = mnm._ffi.pass_.InferType()(mod)
    ret_type = mod["main"].checked_type.ret_type
    assert isinstance(ret_type, relay.ty.TupleType)
    assert len(ret_type.fields) == 2


if __name__ == "__main__":
    pytest.main([__file__])
