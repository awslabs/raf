# pylint: disable=protected-access, redefined-builtin, too-many-locals, line-too-long, too-many-statements
import pytest
import numpy as np
import mnm
from mnm._ffi.pass_ import AnnotateStream, ToGraphNormalForm, InferType
from mnm._core.ir_ext import extended_var
from mnm._lib import tvm
from mnm._lib import relay as _relay
from mnm.testing import run_infer_type

def test_simple_diamond():
    # pylint: disable=invalid-name, no-self-use, unused-variable

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            a_1 = mnm.relu(x)
            a_2 = mnm.abs(a_1)
            a_3 = mnm.relu(a_2)
            a_4 = mnm.abs(a_1)
            a_5 = mnm.relu(a_4)
            out = mnm.add(a_3, a_5)
            return out

    def expected():
        # IR printed by mnm.ir.AsText(module):
        #[version = "0.0.5"]
        # def @main(%x: Tensor[(10, 10), float64]) -> Tensor[(10, 10), float64] {
        #   %0 = mnm.op.stream_start(%x, int32(0) /* ty=int32 */) /* ty=Tensor[(10, 10), float64] */;
        #   %1 = mnm.op.relu(%0) /* ty=Tensor[(10, 10), float64] */;
        #   %2 = mnm.op.abs(%1) /* ty=Tensor[(10, 10), float64] */;
        #   %3 = mnm.op.stream_start(%1, int32(1) /* ty=int32 */) /* ty=Tensor[(10, 10), float64] */;
        #   %4 = mnm.op.stream_wait(%3, int32(0) /* ty=int32 */) /* ty=Tensor[(10, 10), float64] */;
        #   %5 = mnm.op.abs(%4) /* ty=Tensor[(10, 10), float64] */;
        #   %6 = mnm.op.relu(%5) /* ty=Tensor[(10, 10), float64] */;
        #   %7 = mnm.op.stream_end(%6, int32(1) /* ty=int32 */) /* ty=Tensor[(10, 10), float64] */;
        #   %8 = mnm.op.relu(%2) /* ty=Tensor[(10, 10), float64] */;
        #   %9 = mnm.op.stream_wait(%7, int32(1) /* ty=int32 */) /* ty=Tensor[(10, 10), float64] */;
        #   %10 = mnm.op.add(%8, %9, nullptr /* ty=() */, nullptr /* ty=() */) /* ty=Tensor[(10, 10), float64] */;
        #   mnm.op.stream_end(%10, int32(0) /* ty=int32 */) /* ty=Tensor[(10, 10), float64] */
        # }
        x = extended_var("x", dtype="float64", shape=(10, 10))
        a1 = extended_var("a1")
        a2 = extended_var("a2")
        a3 = extended_var("a3")
        a4 = extended_var("a4")
        a5 = extended_var("a5")
        a6 = extended_var("a6")

        const = mnm.ir.const(None)

        tag0 = mnm.ir.const(0, dtype="int32")
        tag1 = mnm.ir.const(1, dtype="int32")

        relu = _relay.op.get("mnm.op.relu")
        abs = _relay.op.get("mnm.op.abs")
        add = _relay.op.get("mnm.op.add")
        begin = _relay.op.get("mnm.op.stream_start")
        end = _relay.op.get("mnm.op.stream_end")
        wait = _relay.op.get("mnm.op.stream_wait")

        master_relu = _relay.Call(begin, [x, tag0])
        master_relu = _relay.Call(relu, [master_relu])
        left_abs = _relay.Call(abs, [master_relu])
        left_relu = _relay.Call(relu, [left_abs])
        right_abs = _relay.Call(begin, [master_relu, tag1])
        right_abs = _relay.Call(wait, [right_abs, tag0])
        right_abs = _relay.Call(abs, [right_abs])
        right_relu = _relay.Call(relu, [right_abs])
        right_relu = _relay.Call(end, [right_relu, tag1])
        right_relu = _relay.Call(wait, [right_relu, tag1])
        master_add = _relay.Call(add, [left_relu, right_relu, const, const])
        master_add = _relay.Call(end, [master_add, tag0])

        return master_add

    model = Model()

    x = mnm.array(np.random.randn(10, 10), dtype="float64")
    mod = model._internal(x).mod
    mod = ToGraphNormalForm()(mod)
    mod = InferType()(mod)
    mod = AnnotateStream()(mod)
    func = mod['main']
    mod['main'] = run_infer_type(func)

    expected_func = expected()
    expected_func = run_infer_type(expected_func)

    # check the structure of the expected ir and generated ir
    assert tvm.ir.structural_equal(mod['main'], expected_func)

if __name__ == "__main__":
    pytest.main([__file__])
