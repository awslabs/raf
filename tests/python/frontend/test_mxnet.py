import pytest
import numpy as np
import mxnet as mx
from mxnet.gluon import nn

import mnm


def check(mnm_x, mx_x, *, rtol=1e-5, atol=1e-5):
    mnm_x = mnm_x.asnumpy()
    mx_x = mx_x.asnumpy()
    np.testing.assert_allclose(mnm_x, mx_x, rtol=rtol, atol=atol)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_mlp():
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(128, activation='relu'))
        net.add(nn.Dense(64, activation='relu'))
        net.add(nn.Dense(10))
    net.initialize(mx.init.Xavier(magnitude=2.24))

    data_np = np.ones((1, 3, 224, 224), dtype="float32")
    data_mx = mx.nd.array(data_np)
    data_mnm = mnm.array(data_np, ctx='cuda')
    data_mnm.requires_grad = True
    data_mx.attach_grad()
    # test infer
    res_mx = net(data_mx)
    model = mnm.frontend.from_mxnet(net, ['data'])
    model.to(ctx='cuda')
    model.infer_mode()
    res_mnm = model(data_mnm)
    check(res_mnm, res_mx)
    # test train
    with mx.autograd.record():
        res_mx = net(data_mx)
    model.train_mode()
    res_mnm = model(data_mnm)
    check(res_mnm, res_mx)
    dy = mnm.array(np.ones((1, 10), dtype="float32"), ctx='cuda')
    res_mx.backward()
    res_mnm.backward(dy)
    check(data_mnm.grad, data_mx.grad)

    mnm_params = model.state()
    for name, param in net.collect_params().items():
        check(mnm_params[name].grad, param.grad())


if __name__ == "__main__":
    test_mlp()
