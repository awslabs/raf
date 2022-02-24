# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use, protected-access, attribute-defined-outside-init
import pytest
import raf
from raf._lib import tvm
from raf._core.module import IRModule
from raf._core.device import Device
from raf.testing import get_testable_devices, randn


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("shape", [[3, 3], [4, 4]])
def test_memory_alloc(device, shape):
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            y = raf.add(x, x)
            z = raf.add(x, y)
            return z

    model_before = Model()
    model_before.infer_mode()
    m_x, _ = randn(shape, device=device)
    func = model_before._internal(m_x).mod["main"]
    mod = IRModule.from_expr(func)
    mod = raf._ffi.pass_.InferType()(mod)
    with Device(device if device != "cpu" else "llvm"):
        mod = raf._ffi.pass_.ManifestAlloc()(mod)
    mod = raf._ffi.pass_.InferType()(mod)
    text = mod["main"].astext()
    assert "alloc_storage" in text
    assert "alloc_tensor" in text
    assert "invoke_op" in text


def test_dynamic_model():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            y = raf.argwhere(x)
            y = raf.argwhere(y)
            y = raf.split(y, 2)
            y = raf.add(y[0], y[1])
            y = raf.abs(y)
            return y

    model = Model()
    m_x, _ = randn((2, 2))
    mod = model._internal(m_x).mod
    with tvm.transform.PassContext():
        mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
        mod = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = raf._ffi.pass_.InferType()(mod)
        mod = raf._ffi.pass_.FuseTVM()(mod)
        mod = raf._ffi.pass_.ToANormalForm()(mod)
        mod = raf._ffi.pass_.InlinePrimitives()(mod)
        mod = raf._ffi.pass_.InferType()(mod)
        mod = raf._ffi.pass_.ManifestAlloc()(mod)
    text = raf.ir.AsText(mod["main"])
    assert (
        "\n".join(text.splitlines()[:-4])
        == """#[version = "0.0.5"]
fn (%x: Tensor[(2, 2), float32]) -> Tensor[(meta[tir.Div][0], 2), int32] {
  let %x_0 = raf.op.vm.alloc_storage(int64(32), int64(64), int32(1), int32(0), str"int32");
  let %x_1 = raf.op.vm.alloc_tensor(%x_0, [4, 2], str"int32", [4, 2]);
  let %x_2 = raf.op.vm.alloc_storage(int64(16), int64(64), int32(1), int32(0), str"int64");
  let %x_3 = raf.op.vm.alloc_tensor(%x_2, [2], str"int64", [2]);
  let %x_4 = raf.op.upper_bound.argwhere;
  let %x_5 = (%x,);
  let %x_6 = (%x_1, %x_3);
  let %x_7 = raf.op.vm.invoke_op(%x_4, %x_5, %x_6);
  let %x1 = raf.op.vm.set_shape(%x_1, %x_3);
  let %x_8 = (%x1,);
  let %x_9 = raf.op.upper_bound.argwhere;
  let %x_10 = raf.op.vm.infer_type(%x_9, %x_8);
  let %x_11 = %x_10.1;
  let %x_12 = %x_11.0;
  let %x_13 = %x_11.1;
  let %x_14 = raf.op.vm.alloc_storage(%x_13, int64(64), int32(1), int32(0), str"int32");
  let %x_15 = raf.op.vm.alloc_tensor(%x_14, %x_12, str"int32", %x_12);
  let %x_16 = %x_10.2;
  let %x_17 = %x_16.0;
  let %x_18 = %x_16.1;
  let %x_19 = raf.op.vm.alloc_storage(%x_18, int64(64), int32(1), int32(0), str"int64");
  let %x_20 = raf.op.vm.alloc_tensor(%x_19, %x_17, str"int64", %x_17);
  let %x_21 = (%x_15, %x_20);
  let %x_22 = raf.op.vm.invoke_op(%x_9, %x_8, %x_21);
  let %x2 = raf.op.vm.set_shape(%x_15, %x_20);
  let %x_23 = nullptr;
  let %x_24 = nullptr;
  let %x_25 = (%x2, %x_23, %x_24);
  let %x_26 = fn (%p0: Tensor[(?, 2), int32], %p1: (), %p2: (), Primitive=1, Dialect="tvm") -> Tensor[(meta[tir.Div][0], 2), int32] {
    %0 = raf.op.tvm.split(%p0, int64(2), int64(0)) /* ty=(Tensor[(meta[tir.Div][0], 2), int32], Tensor[(meta[tir.Div][1], 2), int32]) */;
    %1 = %0.0;
    %2 = %0.1;
    %3 = raf.op.tvm.add(%1, %2, %p1, %p2) /* ty=Tensor[(meta[tir.Div][0], 2), int32] */;
    raf.op.tvm.abs(%3) /* ty=Tensor[(meta[tir.Div][0], 2), int32] */
  };
  let %x_27 = raf.op.vm.infer_type(%x_26, %x_25);
  let %x_28 = %x_27.1;
  let %x_29 = %x_28.0;
  let %x_30 = %x_28.1;
  let %x_31 = raf.op.vm.alloc_storage(%x_30, int64(64), int32(1), int32(0), str"int32");
  let %x_32 = raf.op.vm.alloc_tensor(%x_31, %x_29, str"int32", %x_29);
  let %x_33 = %x_27.0;
  let %x_34 = (%x_32,);
  let %x_35 = raf.op.vm.invoke_op(%x_33, %x_25, %x_34);
  let %x4 = %x_32;
  %x4
}"""
    )


def test_reshape():
    shape = [3, 4, 5]

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            y = raf.reshape(x, (12, 5))
            y = raf.expand_dims(y, axis=0)
            y = raf.squeeze(y)
            y = raf.relu(y)
            y = raf.reshape(y, (3, 4, 5))
            return y

    model = Model()
    m_x, _ = randn(shape, device="cpu")
    func = model._internal(m_x).mod["main"]
    mod = IRModule.from_expr(func)
    mod = raf._ffi.pass_.InferType()(mod)
    with Device("cpu"):
        mod = raf._ffi.pass_.ManifestAlloc()(mod)
    text = mod["main"].astext()
    assert text.count("vm.set_shape") == 4
    assert "reshape" not in text
    assert "expand_dims" not in text
    assert "squeeze" not in text


def test_device():
    shape = [5, 5]

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self):
            return raf.zeros(shape, device="cuda(1)")

    model = Model()
    func = model._internal().mod["main"]
    mod = IRModule.from_expr(func)
    mod = raf._ffi.pass_.InferType()(mod)
    with Device("cpu"):
        mod = raf._ffi.pass_.ManifestAlloc()(mod)
    text = raf.ir.AsText(mod["main"])

    # ManifestAlloc should allocate the tensor on the specific device
    # for init ops and memory related ops. In this case, we expect int32(2) and int32(1),
    # which mean cuda(1).
    assert 'raf.op.vm.alloc_storage(int64(100), int64(64), int32(2), int32(1), str"int32")' in text


if __name__ == "__main__":
    pytest.main([__file__])
