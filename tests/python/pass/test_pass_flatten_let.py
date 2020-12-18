import pytest
from mnm._lib import relay, tvm
from mnm._ffi.ir._make import Module
from mnm._ffi.pass_ import FlattenLet


def test_basic():
    # pylint: disable=protected-access
    # Create a symbolic model and run it
    x = relay.var('x')
    a_1 = relay.var('a1')
    a_2 = relay.var('a2')
    a_3 = relay.var('a3')
    a_4 = relay.var('a4')
    cos = relay.op.get("mnm.op.cos")
    log = relay.op.get("mnm.op.log")
    # fn (%x) {
    #     let %a1 = let %a2 = mnm.op.cos(%x);
    #     let %a3 = mnm.op.cos(%a2);
    #     %a3;
    #     let %a4 = mnm.op.log(%a1);
    #     %a4
    # }
    body = relay.Let(
        a_1,
        relay.Let(a_2, relay.Call(cos, [x]), relay.Let(a_3, relay.Call(cos, [a_2]), a_3)),
        relay.Let(a_4, relay.Call(log, [a_1]), a_4),
    )
    func = relay.Function([x], body)

    def expected():
        # fn (%x) {
        #     let %a2 = mnm.op.cos(%x);
        #     let %a3 = mnm.op.cos(%a2);
        #     let %a4 = mnm.op.log(%a3);
        #     %a4
        # }
        return relay.Let(
            a_2,
            relay.Call(cos, [x]),
            relay.Let(
                a_3, relay.Call(cos, [a_2]),
                relay.Let(a_4, relay.Call(log, [a_3]), a_4)
            )
        )

    mod = Module({})
    mod[tvm.ir.GlobalVar("main")] = func
    func_after = FlattenLet(mod)["main"]
    func_expected = expected()
    tvm.ir.structural_equal(func_after, func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
