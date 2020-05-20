import pytest
import numpy as np
import mxnet as mx
from mxnet.gluon import nn

import mnm


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_mlp():
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(128, activation='relu'))
        net.add(nn.Dense(64, activation='relu'))
        net.add(nn.Dense(10))

    net.initialize(mx.init.Xavier(magnitude=2.24))

    data_np = np.ones((1, 3, 224, 224), dtype="float32")
    res_mx = net(mx.nd.array(data_np)).asnumpy()

    pyfunc = mnm.frontend.from_mxnet(net, ['data'], ctx='cuda')
    res_mnm = pyfunc(mnm.array(data_np, ctx='cuda')).asnumpy()

    np.testing.assert_allclose(res_mnm, res_mx, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_mlp()
