    # pylint: disable=invalid-name,protected-access,too-many-locals
import pytest
import numpy as np
import tvm

import mnm
from mnm._core.module import IRModule
from mnm._lib import relay
from mnm.testing import resnet_cifar10 as resnet
from mnm.testing import check, run_vm_model, with_seed


def test_simple_dedeplicate1():
    x = mnm.ir.var("x")
    y1 = mnm.ir.op.relu(x)
    y2 = mnm.ir.op.relu(y1)
    y3 = mnm.ir.op.relu(y2)
    y4 = mnm.ir.op.relu(y3)
    f = relay.Function([x], y4)
    mod = IRModule.from_expr(f)
    new_mod = mnm._ffi.pass_.Deduplicate(0, False, True, None)(mod)
    assert mnm.ir.AsText(new_mod).count("relu") == 2


def test_simple_dedeplicate2():
    x = mnm.ir.var("x")
    y1 = mnm.ir.op.relu(x)
    y2 = mnm.ir.op.relu(y1)
    y3 = mnm.ir.op.add(y2, y1)
    y4 = mnm.ir.op.relu(y3)
    y5 = mnm.ir.op.relu(y4)
    y6 = mnm.ir.op.multiply(y5, y4)
    f = relay.Function([x], y6)
    mod = IRModule.from_expr(f)
    mod = IRModule.from_expr(f)
    new_mod = mnm._ffi.pass_.Deduplicate(0, False, False, None)(mod)
    text = mnm.ir.AsText(new_mod)
    assert text.count("relu") == 2
    assert text.count(".0;") == 2 and text.count(".1;") == 2


def test_simple_no_dedeplicate1():
    x = mnm.ir.var("x")
    y1 = mnm.ir.op.relu(x)
    y2 = mnm.ir.op.relu(y1)
    y3 = mnm.ir.op.add(y2, y1)
    y4 = mnm.ir.op.relu(y3)
    y5 = mnm.ir.op.relu(y4)
    y6 = mnm.ir.op.multiply(y5, y4)
    f = relay.Function([x], y6)
    mod = IRModule.from_expr(f)
    mod = IRModule.from_expr(f)
    new_mod = mnm._ffi.pass_.Deduplicate(0, False, True, None)(mod)
    assert tvm.ir.structural_equal(mod, new_mod)


def test_simple_no_dedeplicate2():
    x = mnm.ir.var("x")
    a = mnm.ir.op.relu(x)
    b = mnm.ir.op.relu(x)
    c = mnm.ir.op.add(a, b)
    d = mnm.ir.op.add(b, c)
    f = relay.Function([x], d)
    mod = IRModule.from_expr(f)
    new_mod = mnm._ffi.pass_.Deduplicate(0, False, True, None)(mod)
    assert tvm.ir.structural_equal(mod, new_mod)


@pytest.mark.parametrize("must_dominate", [True, False])
def test_resnet_infer(must_dominate):
    x = np.random.randn(1, 3, 32, 32)
    m_x = mnm.array(x, dtype="float32")
    model = resnet.MNMResNet50([3, 4, 6, 3])
    model.infer_mode()
    infer_mod = model._internal(m_x).mod
    infer_mod = mnm._ffi.pass_.ToGraphNormalForm()(infer_mod)
    infer_mod = mnm._ffi.pass_.FoldConstant()(infer_mod)
    infer_mod = mnm._ffi.pass_.InferType()(infer_mod)
    ref_y = model(m_x)

    new_mod = mnm._ffi.pass_.Deduplicate(0, True, must_dominate, None)(infer_mod)
    assert " = fn" in mnm.ir.AsText(new_mod)
    new_mod = mnm._ffi.pass_.ToANormalForm()(new_mod)
    new_model = mnm.frontend.FrameworkModel(new_mod, new_mod, model.state(), {})
    y = new_model(m_x)
    check(y, ref_y)


@with_seed(1)
@pytest.mark.parametrize("must_dominate", [True, False])
def test_resnet_train(must_dominate):
    x = np.random.randn(1, 3, 32, 32)
    y = np.random.randn(1,)
    dy = np.ones((), "float32")
    m_x = mnm.array(x, dtype="float32")
    m_y = mnm.array(y, dtype='int64')
    m_dy = mnm.array(dy)
    model = resnet.MNMResNet50([3, 4, 6, 3])
    model.train_mode()
    new_model = mnm.optim.optim.with_autodiff(model)
    ad_mod = new_model._internal(m_y, m_x, m_y).mod
    ad_mod = mnm._ffi.pass_.ToGraphNormalForm()(ad_mod)
    ad_mod = mnm._ffi.pass_.FoldConstant()(ad_mod)
    ad_mod = mnm._ffi.pass_.InferType()(ad_mod)

    new_mod = mnm._ffi.pass_.Deduplicate(0, True, must_dominate, None)(ad_mod)
    assert " = fn" in mnm.ir.AsText(new_mod)

    ad_mod = mnm._ffi.pass_.ToANormalForm()(ad_mod)
    new_model = mnm.frontend.FrameworkModel(ad_mod, ad_mod, new_model.state(), {})
    ref_out = run_vm_model(new_model, "cpu", [m_dy, m_x, m_y])

    new_mod = mnm._ffi.pass_.ToANormalForm()(new_mod)
    new_model.infer_mode()
    new_model = mnm.frontend.FrameworkModel(new_mod, new_mod, new_model.state(), {})
    out = new_model(m_dy, m_x, m_y)
    vm_out = run_vm_model(new_model, "cpu", [m_dy, m_x, m_y])

    for i, j in zip(ref_out[0], out[0]):
        check(i, j)
    for i, j in zip(ref_out[1], out[1]):
        check(i, j)
    for i, j in zip(ref_out[0], vm_out[0]):
        check(i, j)
    for i, j in zip(ref_out[1], vm_out[1]):
        check(i, j)


if __name__ == "__main__":
    pytest.main([__file__])
