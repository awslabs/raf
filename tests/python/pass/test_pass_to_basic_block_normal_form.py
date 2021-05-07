# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,no-self-use
import pytest
import mnm
from mnm.testing import run_infer_type, randn
import tvm
from tvm import relay


add_op = mnm._ffi.op.GetOp("mnm.op.add")
mul_op = mnm._ffi.op.GetOp("mnm.op.multiply")
relu_op = mnm._ffi.op.GetOp("mnm.op.relu")
equal_op = mnm._ffi.op.GetOp("mnm.op.equal")
split_op = mnm._ffi.op.GetOp("mnm.op.split")
null = mnm.ir.const(None)


def test_no_bind_tuple():
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            z = mnm.add(x, y)
            zz = mnm.split(z, 2)
            return zz[0]

    def expected():
        zero = mnm.ir.const(0)
        two = mnm.ir.const(2)
        x = relay.var("x", shape=(10, 20))
        y = relay.var("y", shape=(10, 1))
        z = relay.Call(add_op, [x, y, null, null])
        z = relay.Call(split_op, [z, two, zero])
        z = relay.TupleGetItem(z, 0)
        f = relay.Function([x, y], z)
        return f

    model = Model()
    m_x, _ = randn((10, 20))
    m_y, _ = randn((10, 1))
    mod = model._internal(m_x, m_y).mod
    mod_after = mnm._ffi.pass_.ToGraphNormalForm()(mod)
    mod_after = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod_after)
    func_after = run_infer_type(mod_after)["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)
    relay.analysis.check_basic_block_normal_form(func_after)


def test_no_bind_diamond():
    konst, _ = randn((1,))
    class Model(mnm.Model):
        def build(self):
            self.c = konst

        @mnm.model.trace
        def forward(self, x, y):
            z1 = mnm.add(x, y)
            z2 = mnm.multiply(x, self.c)
            return mnm.relu(mnm.add(z1, z2))

    def expected():
        x = relay.var("x", shape=(10, 20))
        y = relay.var("y", shape=(10, 1))
        c = relay.var("c", shape=(1,))
        z1 = relay.Call(add_op, [x, y, null, null])
        z2 = relay.Call(mul_op, [x, c])
        z = relay.Call(add_op, [z1, z2, null, null])
        z = relay.Call(relu_op, [z])
        f = relay.Function([x, y, c], z)
        return f

    model = Model()
    m_x, _ = randn((10, 20))
    m_y, _ = randn((10, 1))
    mod = model._internal(m_x, m_y).mod
    func_after = run_infer_type(mnm._ffi.pass_.ToGraphNormalForm()(mod))["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)
    relay.analysis.check_basic_block_normal_form(func_after)


def test_if():
    def if_expr(x):
        """
        free_var %x: int32
        %0 = equal(%x, 2);
        if (%0) {
          %1 = add(%x, 1);
          multiply(%1, 2)
        } else {
          multiply(%1, 1)
        }
        """
        one = mnm.ir.const(1, dtype="int32")
        two = mnm.ir.const(2, dtype="int32")
        v1 = relay.Call(add_op, [x, one, null, null])
        v2 = relay.Call(equal_op, [x, two])
        true_branch = relay.Call(mul_op, [v1, two])
        false_branch = relay.Call(mul_op, [v1, one])
        body = relay.If(v2, true_branch, false_branch)
        return relay.Function([x], body)

    def expected(x):
        """
        free_var %x: int32
        let %v1: float32 = add(%x, 1);
        %0 = equal(%x, 2);
        if (%0) {
          multiply(%v1, 2)
        } else {
          multiply(%v1, 1)
        }
        """
        one = mnm.ir.const(1, dtype="int32")
        two = mnm.ir.const(2, dtype="int32")
        v1 = mnm.ir.var("v1")
        v2 = relay.Call(equal_op, [x, two])
        true_branch = relay.Call(mul_op, [v1, two])
        false_branch = relay.Call(mul_op, [v1, one])
        body = relay.If(v2, true_branch, false_branch)
        body = relay.Let(v1, relay.Call(add_op, [x, one, null, null]), body)
        return relay.Function([x], body)

    x = mnm.ir.var("x", shape=(), dtype="int32")
    func = if_expr(x)
    mod = mnm.ir.IRModule()
    mod["main"] = func
    mod_after = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
    func_after = run_infer_type(mod_after)["main"]
    func_expected = run_infer_type(expected(x))
    assert tvm.ir.structural_equal(func_after, func_expected)
    relay.analysis.check_basic_block_normal_form(func_after)


def test_top_level_nested_if():
    cond_t = mnm.ir.const(True)
    cond_f = mnm.ir.const(False)
    one = mnm.ir.const(1., dtype="float32")
    three = mnm.ir.const(3., dtype="float32")

    def nested_if():
        """
        free_var %x: bool
        if (%x) {
          if (True) {
            free_var %z: float32
            %0 = add(%z, %z);
            free_var %y: float32
            %1 = add(%y, %y);
            add(%0, %1)
          } else {
            add(3f, %1)
          }
        } else {
          if (False) {
            %0
          } else {
            1f
          }
        }
        """
        x = mnm.ir.var("x", shape=(), dtype="bool")
        y = mnm.ir.var("y", shape=(), dtype="float32")
        z = mnm.ir.var("z", shape=(), dtype="float32")
        y2 = relay.Call(add_op, [y, y, null, null])
        z2 = relay.Call(add_op, [z, z, null, null])
        true_branch = relay.If(cond_t, relay.Call(add_op, [z2, y2, null, null]),
                               relay.Call(add_op, [three, y2, null, null]))
        false_branch = relay.If(cond_f, z2, one)
        body = relay.If(x, true_branch, false_branch)
        return relay.Function([x, y, z], body)

    def expected():
        """
        free_var %z: float32
        let %x: float32 = add(%z, %z) /* ty=float32 */;
        free_var %x1: bool
        if (%x1) {
          free_var %y: float32
          let %x2: float32 = add(%y, %y) /* ty=float32 */;
          if (True /* ty=bool */) {
            add(%x, %x2) /* ty=float32 */
          } else {
            add(3f /* ty=float32 */, %x2) /* ty=float32 */
          }
        } else {
          if (False /* ty=bool */) {
            %x
          } else {
            1f /* ty=float32 */
          }
        }
        """
        x = mnm.ir.var("x", shape=(), dtype="bool")
        y = mnm.ir.var("y", shape=(), dtype="float32")
        z = mnm.ir.var("z", shape=(), dtype="float32")
        y2 = relay.var("y2")
        z2 = relay.var("z2")
        true_branch = relay.If(cond_t, relay.Call(add_op, (z2, y2, null, null)),
                               relay.Call(add_op, (three, y2, null, null)))
        true_branch = relay.Let(y2, relay.Call(add_op, (y, y, null, null)), true_branch)
        false_branch = relay.If(cond_f, z2, one)
        body = relay.If(x, true_branch, false_branch)
        body = relay.Let(z2, relay.Call(add_op, (z, z, null, null)), body)
        return relay.Function([x, y, z], body)

    func = nested_if()
    mod = mnm.ir.IRModule()
    mod["main"] = func
    mod_after = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
    func_after = run_infer_type(mod_after)["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)
    relay.analysis.check_basic_block_normal_form(func_after)


def test_nested_if():
    cond_t = mnm.ir.const(True)
    cond_f = mnm.ir.const(False)
    one = mnm.ir.const(1., dtype="float32")
    two = mnm.ir.const(2., dtype="float32")
    three = mnm.ir.const(3., dtype="float32")

    def nested_if():
        """
        free_var %x: bool
        if (%x) {
          if (True) {
            free_var %y: float32
            %0 = add(%y, %y);
            %0
          } else {
            add(3f, %0)
          }
        } else {
          if (False) {
            2f
          } else {
            1f
          }
        }
        """
        x = mnm.ir.var("x", shape=(), dtype="bool")
        y = mnm.ir.var("y", shape=(), dtype="float32")
        y2 = relay.Call(add_op, [y, y, null, null])
        true_branch = relay.If(cond_t, y2, relay.Call(add_op, [three, y2, null, null]))
        false_branch = relay.If(cond_f, two, one)
        body = relay.If(x, true_branch, false_branch)
        return relay.Function([x, y], body)

    def expected():
        """
        free_var %x: bool
        if (%x) {
          free_var %y: float32
          let %x1: float32 = add(%y, %y) /* ty=float32 */;
          if (True /* ty=bool */) {
            %x1
          } else {
            add(3f /* ty=float32 */, %x1) /* ty=float32 */
          }
        } else {
          if (False /* ty=bool */) {
            2f /* ty=float32 */
          } else {
            1f /* ty=float32 */
          }
        }
        """
        x = mnm.ir.var("x", shape=(), dtype="bool")
        y = mnm.ir.var("y", shape=(), dtype="float32")
        y2 = relay.var("y2")
        true_branch = relay.If(cond_t, y2, relay.Call(add_op, (three, y2, null, null)))
        true_branch = relay.Let(y2, relay.Call(add_op, (y, y, null, null)), true_branch)
        false_branch = relay.If(cond_f, two, one)
        body = relay.If(x, true_branch, false_branch)
        return relay.Function([x, y], body)

    func = nested_if()
    mod = mnm.ir.IRModule()
    mod["main"] = func
    mod_after = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
    func_after = run_infer_type(mod_after)["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)
    relay.analysis.check_basic_block_normal_form(func_after)


if __name__ == "__main__":
    pytest.main([__file__])
