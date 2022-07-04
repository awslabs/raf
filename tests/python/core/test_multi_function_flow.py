# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access, attribute-defined-outside-init, too-many-locals
# pylint: disable=too-many-statements, no-self-use, too-many-arguments
import pytest
import raf
from raf._core.ndarray import get_ndarray_handle
from raf.ir import ScopeBuilder
from raf.model.trace import _get_func_inputs
from raf._core.device import Device
from raf._core.executor import VMExecutor
from raf._ffi.model import RunModel
from raf._ffi.memory_pool import InitPool
from raf.testing import get_testable_devices, check, randn
from raf.model.trace import _unwrap

import tvm
from tvm import relay

# Run a module through the VM, return the result in numpy
def run_vm(dev, model_or_mod, args, use_multi_func=False):

    if not isinstance(model_or_mod, tvm.IRModule):
        record = model_or_mod._internal(*args)
        args = _get_func_inputs(record, args, {}, get_handle=False)
        mod = record.mod
    else:
        mod = model_or_mod

    InitPool(Device(dev), "no_pool")
    pass_config = {"raf.memory_schedule": True, "raf.memory_budget": int(13e9)}
    pass_config["raf.use_multi_func"] = use_multi_func

    with tvm.transform.PassContext(
        opt_level=3, config=pass_config, disabled_pass=["FuseTVM", "FuseDialect"]
    ):
        res = VMExecutor(mod, dev).make_executor()(*args)

    return res.numpy()


@pytest.mark.parametrize("device", get_testable_devices())
def test_simple_convnet(device):
    """Simple convnet to test the multi-function flow"""

    ishape = (16, 16, 64, 64)
    pooled_ishape = (16, 16, 32, 32)
    dense_ishape = (16, 16)
    wgtshape = (16, 16, 3, 3)
    dense_wgt_shape = (1, 16)

    # Get operators
    conv2d_op = raf._ffi.op.GetOp("raf.op.conv2d")
    conv2d_call = lambda x, w: relay.Call(
        conv2d_op,
        [
            x,
            w,
            raf.ir.const([1]),
            raf.ir.const([1]),
            raf.ir.const([1]),
            raf.ir.const(1),
            raf.ir.const("NCHW"),
            raf.ir.const("OIHW"),
            raf.ir.const("NCHW"),
        ],
    )
    max_pool2d_op = raf._ffi.op.GetOp("raf.op.max_pool2d")
    max_pool2d_call = lambda x, winsize: relay.Call(
        max_pool2d_op,
        [
            x,
            raf.ir.const(winsize),  # kernel size
            raf.ir.const(winsize),  # stride
            raf.ir.const([0]),  # padding
            raf.ir.const([1]),  # dilation
            raf.ir.const(False),
            raf.ir.const(True),
            raf.ir.const("NCHW"),
        ],
    )

    def get_mod_multi_func():
        """
        Create a simple convnet model with two types of layers. These two
        layers have different input shapes. This test is without autodiff.
        More comprehensive testing with autodiff is postponed.

        fn(input, wgt0, wgt1, wgt2, wgt3, wgt4) {

            // Layer 0
            let layer0 = fn(inp0, wgt0) {
                let conv_out0 = conv2d(inp0, wgt0)
                let relu_out0 = relu(conv_out0)
                relu_out0
            }

            // Layer 1
            let layer1 = fn(inp1, wgt1) {
                let conv_out1 = conv2d(inp1, wgt1)
                let relu_out1 = relu(conv_out1)
                relu_out1
            }

            // First two layers
            let a0 = layer0(input, wgt0)
            let a1 = layer0(a0, wgt1)

            // Go through a pooling layer
            let a1_pooled = maxpool_2d(a1, [2, 2])

            // Next two layers
            let a2 = layer1(a1_pooled, wgt2)
            let a3 = layer1(a2, wgt3)

            // Pool again and go through a dense layer
            let a3_pooled = maxpool_2d(a3, [32, 32]) // Pool each channel to 1 pixel
            let a3_pooled_reshaped = reshape(a3_pooled, [16, 16])
            let out = linear(a3_pooled_reshaped, wgt4)
            out
        }

        """
        data = raf.ir.var("data", shape=ishape)
        wgt0 = raf.ir.var("wgt0", shape=wgtshape)
        wgt1 = raf.ir.var("wgt1", shape=wgtshape)
        wgt2 = raf.ir.var("wgt2", shape=wgtshape)
        wgt3 = raf.ir.var("wgt3", shape=wgtshape)
        wgt4 = raf.ir.var("wgt4", shape=dense_wgt_shape)

        sb = ScopeBuilder()
        # Layer 0
        inp_l0 = raf.ir.var("inp0", shape=ishape)
        wgt_l0 = raf.ir.var("wgt0", shape=wgtshape)
        sb_l0 = ScopeBuilder()
        conv_out0 = sb_l0.let("conv_out0", conv2d_call(inp_l0, wgt_l0))
        relu_out0 = sb_l0.let("relu_out0", raf.ir.op.relu(conv_out0))
        sb_l0.ret(relu_out0)
        func_l0 = relay.Function([inp_l0, wgt_l0], sb_l0.get())
        layer0 = sb.let("layer0", func_l0)

        # Layer 1
        inp_l1 = raf.ir.var("inp1", shape=pooled_ishape)
        wgt_l1 = raf.ir.var("wgt1", shape=wgtshape)
        sb_l1 = ScopeBuilder()
        conv_out1 = sb_l1.let("conv_out1", conv2d_call(inp_l1, wgt_l1))
        relu_out1 = sb_l1.let("relu_out1", raf.ir.op.relu(conv_out1))
        sb_l1.ret(relu_out1)
        func_l1 = relay.Function([inp_l1, wgt_l1], sb_l1.get())
        layer1 = sb.let("layer1", func_l1)

        # Call layer0 twice
        a_0 = sb.let("a0", relay.Call(layer0, [data, wgt0]))
        a_1 = sb.let("a1", relay.Call(layer0, [a_0, wgt1]))

        # Max-pool
        a1_pooled = sb.let("a1_pooled", max_pool2d_call(a_1, (2, 2)))

        # Call layer1 twice
        a_2 = sb.let("a2", relay.Call(layer1, [a1_pooled, wgt2]))
        a_3 = sb.let("a3", relay.Call(layer1, [a_2, wgt3]))

        # Finish
        a3_pooled = sb.let("a3_pooled", max_pool2d_call(a_3, (32, 32)))  # shape=(16, 16, 1, 1)
        a3_pooled_reshaped = sb.let(
            "a3_pooled_reshaped", raf.ir.op.reshape(a3_pooled, dense_ishape)
        )
        outp = sb.let("outp", raf.ir.op.dense(a3_pooled_reshaped, wgt4))
        sb.ret(outp)
        func = relay.Function([data, wgt0, wgt1, wgt2, wgt3, wgt4], sb.get())
        return tvm.IRModule.from_expr(func)

    def get_mod_flat():
        """
        The flat version of the module above.

        fn(input, wgt0, wgt1, wgt2, wgt3, wgt4) {
            // First two layers
            let conv_out0 = conv2d(input, wgt0)
            let relu_out0 = relu(conv_out0)
            let conv_out1 = conv2d(relu_out0, wgt1)
            let relu_out1 = relu(conv_out1)

            // Go through a pooling layer
            let relu_out1_pooled = maxpool_2d(relu_out1, [2, 2])

            // Next two layers
            let conv_out2 = conv2d(relu_out1_pooled, wgt2)
            let relu_out2 = relu(conv_out2)
            let conv_out3 = conv2d(relu_out2, wgt3)
            let relu_out3 = relu(conv_out3)

            // Pool again and go through a dense layer
            let relu_out3_pooled = maxpool_2d(relu_out3, [32, 32]) // Pool each channel to 1 pixel
            let relu_out3_pooled_reshaped = reshape(relu_out3_pooled, [16, 16])
            let out = linear(relu_out3_pooled_reshaped, wgt4)
            out
        }
        """
        data = raf.ir.var("data", shape=ishape)
        wgt0 = raf.ir.var("wgt0", shape=wgtshape)
        wgt1 = raf.ir.var("wgt1", shape=wgtshape)
        wgt2 = raf.ir.var("wgt2", shape=wgtshape)
        wgt3 = raf.ir.var("wgt3", shape=wgtshape)
        wgt4 = raf.ir.var("wgt4", shape=dense_wgt_shape)

        sb = ScopeBuilder()
        conv_out0 = sb.let("conv_out0", conv2d_call(data, wgt0))
        relu_out0 = sb.let("relu_out0", raf.ir.op.relu(conv_out0))
        conv_out1 = sb.let("conv_out1", conv2d_call(relu_out0, wgt1))
        relu_out1 = sb.let("relu_out1", raf.ir.op.relu(conv_out1))
        relu_out1_pooled = sb.let("relu_out1_pooled", max_pool2d_call(relu_out1, (2, 2)))
        conv_out2 = sb.let("conv_out2", conv2d_call(relu_out1_pooled, wgt2))
        relu_out2 = sb.let("relu_out2", raf.ir.op.relu(conv_out2))
        conv_out3 = sb.let("conv_out3", conv2d_call(relu_out2, wgt3))
        relu_out3 = sb.let("relu_out3", raf.ir.op.relu(conv_out3))
        relu_out3_pooled = sb.let(
            "relu_out3_pooled", max_pool2d_call(relu_out3, (32, 32))
        )  # shape=(16, 16, 1, 1)
        relu_out3_pooled_reshaped = sb.let(
            "relu_out3_pooled_reshaped", raf.ir.op.reshape(relu_out3_pooled, dense_ishape)
        )
        outp = sb.let("outp", raf.ir.op.dense(relu_out3_pooled_reshaped, wgt4))
        sb.ret(outp)
        func = relay.Function([data, wgt0, wgt1, wgt2, wgt3, wgt4], sb.get())
        return tvm.IRModule.from_expr(func)

    m_x, _ = randn(ishape, device=device)
    m_wgt0, _ = randn(wgtshape, device=device)
    m_wgt1, _ = randn(wgtshape, device=device)
    m_wgt2, _ = randn(wgtshape, device=device)
    m_wgt3, _ = randn(wgtshape, device=device)
    m_wgt4, _ = randn(dense_wgt_shape, device=device)
    all_inputs = [m_x, m_wgt0, m_wgt1, m_wgt2, m_wgt3, m_wgt4]

    # Try the VM path
    multi_func_res = run_vm(device, get_mod_multi_func(), all_inputs, use_multi_func=True)
    flat_res = run_vm(device, get_mod_flat(), all_inputs)

    check(multi_func_res, flat_res)

    # Try the interpreter path
    interp_all_inputs = [get_ndarray_handle(inp) for inp in all_inputs]
    multi_func_res = _unwrap(RunModel(get_mod_multi_func(), interp_all_inputs))
    flat_res = _unwrap(RunModel(get_mod_flat(), interp_all_inputs))
    check(multi_func_res, flat_res)


if __name__ == "__main__":
    pytest.main([__file__])
