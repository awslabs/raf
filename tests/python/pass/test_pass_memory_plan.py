# pylint: disable=protected-access, no-self-use, attribute-defined-outside-init, invalid-name
# pylint: disable=unused-variable, too-many-arguments
import pytest
import mnm
from mnm._lib import tvm
from mnm.testing import get_device_list, randn, check, get_vm_executor


def optimize(mod, device, reuse_storage=False, fusion=False):
    fuse_level = 1 if fusion else 0
    target_name = device if device != "cpu" else "llvm"
    with tvm.transform.PassContext(opt_level=3,
                                   config={"mnm.memory_plan.reuse_storage": reuse_storage,
                                           "mnm.fuse_level": fuse_level}):
        opt_mod, _ = mnm._core.vm.VMCompiler().optimize(mod, target=target_name, params={})
    return opt_mod


def verify_alloc_num(func, expected_alloc_storage, expected_alloc_tensor, expected_out_tensor,
                     expected_free_memory, expected_size):
    # A helper function to verify alloc_storage and alloc_tensor numbers and total sizes
    alloc_storage = 0
    alloc_tensor = 0
    out_tensor = 0
    free_memory = 0
    total_size = 0
    for line in mnm.ir.AsText(func).split("\n"):
        if line.find("mnm.op.vm.alloc_storage") != -1:
            alloc_storage += 1
            total_size += int(line[line.find("int64(") + 6 : line.find(")")])
        elif line.find("mnm.op.vm.alloc_tensor") != -1:
            if line.find("bool(1)") != -1:
                out_tensor += 1
            alloc_tensor += 1
        elif line.find("mnm.op.vm.free") != -1:
            free_memory += 1

    assert (
        alloc_storage == expected_alloc_storage and alloc_tensor == expected_alloc_tensor and
        out_tensor == expected_out_tensor
    ), "#storage %d, #tensor %d, #out %d" % (alloc_storage, alloc_tensor, out_tensor)
    assert free_memory == expected_free_memory, "#free %d" % free_memory
    assert total_size == expected_size, "Total size %d, but expected %d" % (
        total_size,
        expected_size,
    )


def verify_correctness(model, device, args, reuse_storage, fusion):
    # A helper function to verify the correctness
    fuse_level = 1 if fusion else 0
    executor, vm_inputs = get_vm_executor(model, device, args, fuse_level=fuse_level,
                                          reuse_storage=reuse_storage)
    outs = executor(*vm_inputs)
    outs = outs if isinstance(outs, (tuple, list)) else (outs,)

    ref_outs = model(*args)
    ref_outs = ref_outs if isinstance(ref_outs, (tuple, list)) else (ref_outs,)

    assert len(ref_outs) == len(outs)
    for ref_out, out in zip(ref_outs, outs):
        check(ref_out, out)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("fusion", [False, True])
def test_memory_plan_basic(device, fusion):
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, a, b, c, d):
            t0 = mnm.add(a, a)
            t1 = mnm.add(t0, b)
            t2 = mnm.add(t1, c)
            t3 = mnm.add(t2, t0)  # t1 and t3 share buffer
            t4 = mnm.add(t3, d)  # t4 is the final output so cannot share with t0
            return t4

    shape = (5, 5)
    model_before = Model()
    model_before.infer_mode()
    m_a, _ = randn(shape, device=device)
    m_b, _ = randn(shape, device=device)
    m_c, _ = randn(shape, device=device)
    m_d, _ = randn(shape, device=device)
    args = [m_a, m_b, m_c, m_d]

    mod = model_before._internal(*args).mod

    mod1 = optimize(mod, device, reuse_storage=True, fusion=fusion)
    if not fusion:
        verify_alloc_num(mod1["main"], 4, 5, 1, 3, 400)
    else:
        verify_alloc_num(mod1["main"], 1, 1, 1, 0, 100)

    verify_correctness(model_before, device, args, reuse_storage=False, fusion=fusion)
    verify_correctness(model_before, device, args, reuse_storage=True, fusion=fusion)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("fusion", [False, True])
def test_memory_plan_multi_outs(device, fusion):
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, m_x, m_w, m_b, m_m, m_v):
            t0 = mnm.relu(m_x)
            res = mnm.batch_norm_train(t0, m_m, m_v, m_w, m_b, 0.1, 1e-5)
            t1 = res[0]
            t2 = res[1]
            t3 = res[2]
            t4 = mnm.relu(t1)  # t0 and t4 can share buffer
            t5 = mnm.relu(t4)  # t1 is the final output so cannot share with t5
            return t5

    model_before = Model()
    model_before.infer_mode()
    shape = (16, 3, 224, 224)
    stats_shape = [shape[1]]
    m_x, _ = randn(shape, device=device)
    m_m, _ = randn(stats_shape, device=device)
    m_v, _ = randn(stats_shape, positive=True, device=device)
    m_w, _ = randn(stats_shape, device=device)
    m_b, _ = randn(stats_shape, device=device)
    args = [m_x, m_m, m_v, m_w, m_b]

    mod = model_before._internal(*args).mod
    mod = optimize(mod, device, reuse_storage=True, fusion=fusion)
    if not fusion:
        verify_alloc_num(mod["main"], 5, 6, 1, 4, 28901400)
    else:
        # The memory footprint is the same as no-fused one because FuseOps
        # just fuses two relu ops.
        verify_alloc_num(mod["main"], 5, 5, 1, 4, 28901400)

    verify_correctness(model_before, device, args, reuse_storage=False, fusion=fusion)
    verify_correctness(model_before, device, args, reuse_storage=True, fusion=fusion)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("fusion", [False, True])
def test_memory_plan_group_selection(device, fusion):
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, a, b):
            t0 = mnm.add(a, a)  # new buffer size 20
            t1 = mnm.add(b, t0)  # new buffer size 200
            t2 = mnm.add(t0, t1)  # new buffer size 200
            t3 = mnm.repeat(t2, 2, axis=0)  # can share with t0, t1. select t1, size 400
            t4 = mnm.sum(t3, axis=0)  # can share with t0, t2. select t0, size 20
            t5 = mnm.add(t3, t4)  # final output cannot share with t2. new buffer size 400
            return t5

    model_before = Model()
    model_before.infer_mode()
    m_a, _ = randn((1, 5), device=device)
    m_b, _ = randn((10, 5), device=device)
    args = [m_a, m_b]

    mod = model_before._internal(*args).mod
    mod = optimize(mod, device, reuse_storage=True, fusion=fusion)
    if not fusion:
        verify_alloc_num(mod["main"], 4, 6, 1, 3, 1020)
    else:
        verify_alloc_num(mod["main"], 3, 3, 1, 2, 820)

    verify_correctness(model_before, device, args, reuse_storage=False, fusion=fusion)
    verify_correctness(model_before, device, args, reuse_storage=True, fusion=fusion)


def test_set_shape():
    shape = [3, 4, 5]

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.relu(x)
            y = mnm.reshape(y, (12, 5))
            y = mnm.relu(y)
            return y

    model = Model()
    m_x, _ = randn(shape, device="cpu")
    args = [m_x]
    mod = model._internal(*args).mod
    mod = optimize(mod, "cpu", reuse_storage=False, fusion=False)
    verify_alloc_num(mod["main"], 2, 2, 1, 1, 480)
    verify_correctness(model, "cpu", args, reuse_storage=False, fusion=False)


if __name__ == "__main__":
    pytest.main([__file__])
