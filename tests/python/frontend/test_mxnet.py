# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-locals, too-many-arguments, too-many-statements
# pylint: disable=protected-access
import pytest
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

import raf


def check(raf_x, mx_x, *, rtol=1e-5, atol=1e-5):
    raf_x = raf_x.numpy()
    mx_x = mx_x.asnumpy()
    np.testing.assert_allclose(raf_x, mx_x, rtol=rtol, atol=atol)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_mlp():
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(128, activation="relu"))
        net.add(nn.Dense(64, activation="relu"))
        net.add(nn.Dense(10))
    net.initialize(mx.init.Xavier(magnitude=2.24))

    data_np = np.ones((1, 3, 224, 224), dtype="float32")
    data_mx = mx.nd.array(data_np)
    data_raf = raf.array(data_np, device="cuda")
    data_raf.requires_grad = True
    data_mx.attach_grad()
    # test infer
    res_mx = net(data_mx)
    model = raf.frontend.from_mxnet(net, ["data"])
    model.to(device="cuda")
    model.infer_mode()
    res_raf = model(data_raf)
    check(res_raf, res_mx)
    # test train
    with mx.autograd.record():
        res_mx = net(data_mx)
    model.train_mode()
    res_raf = model(data_raf)
    check(res_raf, res_mx)
    dy = raf.array(np.ones((1, 10), dtype="float32"), device="cuda")
    res_mx.backward()
    res_raf.backward(dy)
    check(data_raf.grad, data_mx.grad)

    raf_params = model.state()
    for name, param in net.collect_params().items():
        check(raf_params[name].grad, param.grad())


# @pytest.mark.parametrize("mode", ["rnn", "gru", "lstm"])
@pytest.mark.parametrize("mode", ["rnn"])
@pytest.mark.parametrize("seq_len", [1])
# @pytest.mark.parametrize("input_size", [32, 64])
@pytest.mark.parametrize("input_size", [64])
# @pytest.mark.parametrize("hidden_size", [32, 64])
@pytest.mark.parametrize("hidden_size", [64])
# @pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("num_layers", [1])
# @pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("init_states", [True])
# @pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("bidirectional", [False])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_rnn(mode, seq_len, input_size, hidden_size, num_layers, batch, init_states, bidirectional):
    if mode == "rnn":
        net = gluon.rnn.RNN(hidden_size, num_layers, bidirectional=bidirectional)
    elif mode == "gru":
        net = gluon.rnn.GRU(hidden_size, num_layers, bidirectional=bidirectional)
    else:  # mode == "lstm"
        net = gluon.rnn.LSTM(hidden_size, num_layers, bidirectional=bidirectional)
    num_states = 2 if mode == "lstm" else 1
    net.initialize()
    net.hybridize()

    device = "gpu"
    dtype = "float32"
    directions = 2 if bidirectional else 1
    np.random.seed(0)
    data_np = np.random.uniform(size=(seq_len, batch, input_size)).astype(dtype)
    data_mx = mx.nd.array(data_np)
    data_raf = raf.array(data_np, device=device)
    data_raf.requires_grad = True

    if init_states:
        shape_dict = {"data0": data_np.shape}
        inputs = {"data0": data_raf}
        state_shape = (num_layers * directions, batch, hidden_size)
        states_np = []
        states_mx = []
        for i in range(num_states):
            state = np.random.uniform(size=state_shape).astype(dtype)
            states_np.append(state)
            states_mx.append(mx.nd.array(state))
            state_raf = raf.array(state, device=device)
            state_raf.requires_grad = True
            shape_dict["data%s" % (i + 1)] = state.shape
            inputs["data%s" % (i + 1)] = state_raf
        mx_out, mx_states = net(data_mx, states_mx)
        mx_res = [mx_out] + mx_states
    else:
        shape_dict = {"data": data_np.shape}
        inputs = {"data": data_raf}
        mx_res = net(data_mx)

    mx_sym = net._cached_graph[1]
    mx_params = {}
    for name, param in net.collect_params().items():
        mx_params[name] = param._reduce()

    # TODO - What should be the input names
    model = raf.frontend.from_mxnet(mx_sym, inputs_name=["data"], arg_params=mx_params)
    model.to(device=device)
    model.infer_mode()

    op_res = model(**inputs)
    if init_states:
        assert len(op_res) == len(mx_res)
        for i, val in enumerate(op_res):
            check(val, mx_res[i], rtol=1e-3)
    else:
        check(op_res, mx_res, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
