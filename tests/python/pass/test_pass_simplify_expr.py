import pytest
import tvm
from tvm import relay
from mnm._ffi.pass_ import FromRelay, InferType, AutoDiff, SimplifyExpr

def test_basic():
    # Get a Relay func
    def get_mod():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(10, 100), dtype="float32")
        y = relay.var("y", shape=(1, 100), dtype="float32")
        out = relay.add(x, y)
        mod['main'] = relay.Function([x, y], out)
        return mod

    tvm_mod = get_mod()
    mod = FromRelay(tvm_mod)
    mod = InferType()(mod)
    mod = AutoDiff(mod, [])
    mod = InferType()(mod)
    mod = SimplifyExpr()(mod)


    # Ensure that there is only one sum operator
    sum_ops = list()
    find_sum = lambda x: sum_ops.append(
        isinstance(x, tvm.relay.Call)
        and x.op.name == "mnm.op.sum"
    )
    tvm.relay.analysis.post_order_visit(mod['main'], find_sum)
    assert len(list(filter(lambda x: x, sum_ops))) == 1


if __name__ == "__main__":
    pytest.main([__file__])
