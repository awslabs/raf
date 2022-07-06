# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements
import pytest
import numpy as np

import raf
from raf import distributed as dist
from raf.testing import randn, run_infer_type
from raf._ffi.pass_ import InferType, AutoDiff, AutoDataParallel
from raf.ir import RAFSequential
import tvm
from tvm import relay


def one_hot(batch_size, num_classes, device="cuda"):
    targets = np.random.randint(0, num_classes, size=batch_size)
    m_x = raf.array(targets, device=device)
    assert list(m_x.shape) == [
        batch_size,
    ]
    return m_x


# pylint: disable=unused-variable
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize(
    "config",
    [
        (2, 2, 2),
    ],
)
def test_dp(config):
    dcfg = dist.get_config()
    dcfg.enable_data_parallel = True
    comm = dist.get_communicator()
    device = f"cuda({comm.local_rank})"
    const, _ = randn([config[0], config[1]], device=device)
    nccl_version = raf.build.with_nccl()

    class TestModel(raf.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            self.c = const

        # pylint: enable=attribute-defined-outside-init

        @raf.model.trace
        def forward(self, x, y_true):
            y_pred = self.forward_infer(x)
            loss = raf.nll_loss(y_true=y_true, y_pred=y_pred)
            return loss

        @raf.model.trace
        def forward_infer(self, x):
            out = raf.matmul(x, self.c)
            return out

    def expected():
        shape = [config[0], config[1]]

        # Params
        x = raf.ir.var("x", relay.TensorType(shape))
        c = raf.ir.var("c", relay.TensorType(shape))
        y_true = raf.ir.var(
            "y_true",
            relay.TensorType(
                [
                    2,
                ],
                dtype="int64",
            ),
        )

        # Forward IR components
        expr_a1 = raf.ir.op.matmul(x, c)
        var_a1 = raf.ir.var("a1")

        expr_a2 = raf.ir.op.nll_loss(y_true, var_a1)
        var_a2 = raf.ir.var("a2")

        # Backward IR components
        dy = raf.ir.var(
            "dy",
            relay.TensorType(
                [
                    1,
                ],
                dtype="float32",
            ),
        )
        var_closure = raf.ir.var("closure")

        expr_x1 = raf.ir.op.nll_loss_dpred(dy, y_true, var_a1)
        var_x0 = raf.ir.var("x0")

        expr_x2 = raf.ir.op.matmul_nt(var_x0, c)
        var_x1 = raf.ir.var("x1")

        allreduce_in = raf.ir.var("allreduce_in")
        expr_t = relay.Tuple([var_x1])

        expr_x3 = raf.ir.op.matmul_tn(x, var_x0)
        var_x2 = raf.ir.var("x2")

        allreduce_in1 = raf.ir.var("allreduce_in1")
        expr_t1 = relay.Tuple([var_x2])

        expr_x4 = raf.ir.op.zeros_like(y_true)
        var_x3 = raf.ir.var("x3")

        allreduce_in2 = raf.ir.var("allreduce_in2")
        expr_t2 = relay.Tuple([var_x3])

        if nccl_version > 21000:
            expr_g = raf.ir.op._allreduce(allreduce_in, "avg")
            var_g = raf.ir.var("g")

            expr_g1 = raf.ir.op._allreduce(allreduce_in1, "avg")
            var_g1 = raf.ir.var("g1")

            expr_g2 = raf.ir.op._allreduce(allreduce_in2, "avg")
            var_g2 = raf.ir.var("g2")
        else:
            fdeno = raf.ir.const(float(comm.size), dtype="float32")
            ideno = raf.ir.const(comm.size, dtype="int64")

            expr_g = raf.ir.op._allreduce(allreduce_in)
            var_g_sum = raf.ir.var("g_sum")
            expr_avg = raf.ir.op.divide(var_g_sum, fdeno)
            var_g = raf.ir.var("g")

            expr_g1 = raf.ir.op._allreduce(allreduce_in1)
            var_g1_sum = raf.ir.var("g_sum1")
            expr_avg1 = raf.ir.op.divide(var_g1_sum, fdeno)
            var_g1 = raf.ir.var("g1")

            expr_g2 = raf.ir.op._allreduce(allreduce_in2)
            var_g2_sum = raf.ir.var("g_sum2")
            expr_avg2 = raf.ir.op.divide(var_g2_sum, ideno)
            var_g2 = raf.ir.var("g2")

        expr_x5 = relay.Tuple([var_g, var_g2, var_g1])
        var_x5 = raf.ir.var("x5")

        # Forward IR components
        expr_ret = relay.Tuple([var_a2, var_closure])
        var_ret = raf.ir.var("ret")
        if nccl_version >= 21000:
            # Construct Backward IR as a closure
            let8 = relay.Let(var_x5, expr_x5, var_x5)
            let7 = relay.Let(var_g2, expr_g2, let8)
            let_ad2 = relay.Let(allreduce_in2, expr_t2, let7)
            let6 = relay.Let(var_x3, expr_x4, let_ad2)
            let5 = relay.Let(var_g1, expr_g1, let6)
            let_ad1 = relay.Let(allreduce_in1, expr_t1, let5)
            let4 = relay.Let(var_x2, expr_x3, let_ad1)
            let3 = relay.Let(var_g, expr_g, let4)
            let_ad = relay.Let(allreduce_in, expr_t, let3)
            let2 = relay.Let(var_x1, expr_x2, let_ad)
            let1 = relay.Let(var_x0, expr_x1, let2)
            closure_func = relay.Function([dy], let1)
        else:
            # Construct Backward IR as a closure
            let8 = relay.Let(var_x5, expr_x5, var_x5)
            let_div2 = relay.Let(var_g2, expr_avg2, let8)
            let7 = relay.Let(var_g2_sum, expr_g2, let_div2)
            let_ad2 = relay.Let(allreduce_in2, expr_t2, let7)
            let6 = relay.Let(var_x3, expr_x4, let_ad2)
            let_div1 = relay.Let(var_g1, expr_avg1, let6)
            let5 = relay.Let(var_g1_sum, expr_g1, let_div1)
            let_ad1 = relay.Let(allreduce_in1, expr_t1, let5)
            let4 = relay.Let(var_x2, expr_x3, let_ad1)
            let_div = relay.Let(var_g, expr_avg, let4)
            let3 = relay.Let(var_g_sum, expr_g, let_div)
            let_ad = relay.Let(allreduce_in, expr_t, let3)
            let2 = relay.Let(var_x1, expr_x2, let_ad)
            let1 = relay.Let(var_x0, expr_x1, let2)
            closure_func = relay.Function([dy], let1)

        # Construct Forward IR
        let10 = relay.Let(var_ret, expr_ret, var_ret)
        let0 = relay.Let(var_closure, closure_func, let10)

        let_1 = relay.Let(var_a2, expr_a2, let0)
        let_2 = relay.Let(var_a1, expr_a1, let_1)

        return relay.Function([x, y_true, c], let_2)

    m_model = TestModel()
    m_model.to(device=device)
    m_model.train_mode()

    m_x, _ = randn([config[0], config[1]], device=device, requires_grad=True)
    m_y = one_hot(batch_size=2, num_classes=config[2], device=device)
    m_x.requires_grad = True
    m_y.requires_grad = True

    record = m_model._internal(m_x, m_y)
    mod_before = record.mod
    passes = [
        InferType(),
        AutoDiff(record.requires_grads),
        InferType(),
        AutoDataParallel(),
        InferType(),
    ]
    seq = RAFSequential(passes)
    mod = seq(mod_before)
    func_after = mod["main"]
    func_expected = expected()
    func_expected = run_infer_type(func_expected)
    text = func_after.astext()
    assert "raf.op._allreduce" in text
    assert tvm.ir.structural_equal(func_after, func_expected)
    dcfg.enable_data_parallel = False


if __name__ == "__main__":
    pytest.main([__file__])
