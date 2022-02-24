# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name, no-else-return, protected-access, too-many-lines
# pylint: disable=no-member, unused-argument
# pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-branches
"""MXNet symbol frontend."""
import json

from raf._core.ndarray import Symbol, ndarray
from raf._core.ndarray import array as raf_array
from raf._core.module import IRModule as raf_module
from raf._ffi.pass_ import ExtractBinding
from raf._lib import relay
from raf._op import sym as op
from raf.frontend.model import FrameworkModel

_saved_reshape_inputs = dict()
_extra_aux_params = dict()


def _generator():
    """Generate unique names - returns a function."""

    def f():
        f.count += 1
        return "aux_param" + str(f.count)

    f.count = 0
    return f


_unique_name = _generator()


_activation_map = {"sigmoid": op.sigmoid, "tanh": op.tanh, "relu": op.relu}


def _mx_conv(inputs, attrs, is_train):
    def _mx_conv2d(inputs, attrs):
        new_attrs = {}
        new_attrs["stride"] = attrs.get_int_tuple("stride", (1, 1))
        new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0))
        new_attrs["dilation"] = attrs.get_int_tuple("dilate", (1, 1))
        new_attrs["groups"] = attrs.get_int("num_group", 1)
        use_bias = not attrs.get_bool("no_bias", False)
        res = op.conv2d(inputs[0], inputs[1], **new_attrs)
        if use_bias:
            assert len(inputs) == 3
            res = op.bias_add(res, inputs[2])
        return [res]

    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) == 2:
        return _mx_conv2d(inputs, attrs)
    else:
        raise NotImplementedError


def _mx_fully_connected(inputs, attrs, is_train):
    use_bias = not attrs.get_bool("no_bias", False)
    use_flatten = attrs.get_bool("flatten", True)
    if use_flatten:
        res = op.batch_flatten(inputs[0])
        res = op.matmul_nt(res, inputs[1])
    else:
        res = op.matmul_nt(inputs[0], inputs[1])
    if use_bias:
        assert len(inputs) == 3
        res = op.add(res, inputs[2])
    return [res]


def _mx_batch_norm(inputs, attrs, is_train):
    new_attrs = {}
    new_attrs["eps"] = attrs.get_float("eps", 0.001)
    new_attrs["momentum"] = attrs.get_float("momentum", 0.9)
    if is_train:
        res = op.batch_norm_train(
            x=inputs[0],
            w=inputs[1],
            b=inputs[2],
            running_mean=inputs[3],
            running_var=inputs[4],
            **new_attrs
        )
        return res
    else:
        res = op.batch_norm_infer(
            x=inputs[0],
            w=inputs[1],
            b=inputs[2],
            running_mean=inputs[3],
            running_var=inputs[4],
            **new_attrs
        )
        return [res]


def _mx_pooling(inputs, attrs, is_train):
    global_pool = attrs.get_bool("global_pool", False)
    pool_type = attrs.get_str("pool_type")

    def _pool2d(new_op, is_avg):
        kernel_size = attrs.get_int_tuple("kernel")
        if len(kernel_size) != 2:
            raise ValueError("Only 2D kernels are supported for operator Pool2D.")
        new_attrs = {}
        new_attrs["kernel"] = kernel_size
        new_attrs["stride"] = attrs.get_int_tuple("stride", (1, 1))
        new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0))
        new_attrs["ceil_mode"] = attrs.get_str("pooling_convention", "valid") == "full"
        if is_avg:
            new_attrs["include_pad"] = attrs.get_bool("count_include_pad", True)

        if (new_attrs["kernel"] == 1) or (new_attrs["kernel"] == (1, 1)):
            return [inputs[0]]
        return [new_op(inputs[0], **new_attrs)]

    if pool_type == "max":
        if global_pool:
            return [op.adaptive_max_pool2d(inputs[0], shape=(1, 1))]
        return _pool2d(op.max_pool2d, False)
    if pool_type == "avg":
        if global_pool:
            return [op.adaptive_avg_pool2d(inputs[0], shape=(1, 1))]
        return _pool2d(op.avg_pool2d, True)
    raise NotImplementedError


def _mx_activations(inputs, attrs, is_train):
    act_type = attrs.get_str("act_type")
    assert len(inputs) == 1
    if act_type == "relu":
        return [op.relu(inputs[0])]
    else:
        raise NotImplementedError


def _mx_add(inputs, attrs, is_train):
    return [op.add(inputs[0], inputs[1])]


def _mx_reshape(inputs, attrs, is_train):
    shape = attrs.get_int_tuple("shape")
    reverse = attrs.get_bool("reverse", False)
    if reverse:
        return op.reverse_reshape(inputs[0], newshape=shape)
    out = op.reshape(inputs[0], shape=shape)
    _saved_reshape_inputs[out] = inputs[0]
    return [out]


def _mx_expand_dims(inputs, attrs, is_train):
    axis = attrs.get_int("axis")
    out = op.expand_dims(inputs[0], axis=axis)
    return [out]


def _mx_sum(inputs, attrs, is_train):
    axis_list = attrs.get_int_tuple("axis")
    out = op.sum(inputs[0], axis=axis_list)
    return [out]


def _mx_swap_axis(inputs, attrs, is_train):
    dim1 = attrs.get_int("dim1")
    dim2 = attrs.get_int("dim2")
    out = op.swap_axis(inputs[0], axis1=dim2, axis2=dim1)
    return [out]


def _mx_multiply(inputs, attrs, is_train):
    return [op.multiply(inputs[0], inputs[1])]


def _mx_softmax(inputs, attrs, is_train):
    axis = attrs.get_int("axis")
    out = op.softmax(inputs[0], axis=axis)
    return [out]


def _mx_adaptive_avg_pooling(inputs, attrs, is_train):
    output_size = attrs.get_int_tuple("output_size")
    if output_size == (1,):
        output_size = (1, 1)

    out = op.adaptive_avg_pool2d(inputs[0], shape=output_size)
    return [out]


def _mx_flatten(inputs, attrs, is_train):
    out = op.batch_flatten(inputs[0])
    return [out]


def _mx_rnn_param_concat(inputs, attrs, _):
    # We don't need to concatenate RNN params because we will unravel the RNN op
    return [inputs]


def _mx_rnn_layer(inputs, attrs, is_train):
    def _rnn_cell(data, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias, activation):
        i2h = op.bias_add(op.dense(data, i2h_weight), i2h_bias, axis=-1)
        h2h = op.bias_add(op.dense(states[0], h2h_weight), h2h_bias, axis=-1)
        out = _activation_map[activation](op.add(i2h, h2h))
        return out, [out]

    def _gru_cell(data, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias):
        dtype = "float32"
        i2h = op.bias_add(op.dense(data, i2h_weight), i2h_bias, axis=-1)
        h2h = op.bias_add(op.dense(states[0], h2h_weight), h2h_bias, axis=-1)
        i2h_split = op.split(i2h, indices_or_sections=3, axis=1)
        i2h_r, i2h_z, i2h = i2h_split[0], i2h_split[1], i2h_split[2]
        h2h_split = op.split(h2h, indices_or_sections=3, axis=1)
        h2h_r, h2h_z, h2h = h2h_split[0], h2h_split[1], h2h_split[2]
        reset_gate = _activation_map["sigmoid"](op.add(i2h_r, h2h_r))
        update_gate = _activation_map["sigmoid"](op.add(i2h_z, h2h_z))
        next_h_tmp = _activation_map["tanh"](op.add(op.multiply(reset_gate, h2h), i2h))
        name = _unique_name()
        indices = raf_array(1, dtype=dtype)
        _extra_aux_params[name] = indices

        next_h = op.add(
            op.multiply(op.subtract(Symbol.make_var(name_hint=name), update_gate), next_h_tmp),
            op.multiply(update_gate, states[0]),
        )
        return next_h, [next_h]

    def _lstm_cell(data, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias):
        i2h = op.bias_add(op.dense(data, i2h_weight), i2h_bias, axis=-1)
        h2h = op.bias_add(op.dense(states[0], h2h_weight), h2h_bias, axis=-1)
        gates = op.add(i2h, h2h)
        slice_gates = op.split(gates, indices_or_sections=4, axis=1)
        in_gate = _activation_map["sigmoid"](slice_gates[0])
        forget_gate = _activation_map["sigmoid"](slice_gates[1])
        in_transform = _activation_map["tanh"](slice_gates[2])
        out_gate = _activation_map["sigmoid"](slice_gates[3])
        next_c = op.add(op.multiply(forget_gate, states[1]), op.multiply(in_gate, in_transform))
        next_h = op.multiply(out_gate, _activation_map["tanh"](next_c))
        return next_h, [next_h, next_c]

    num_layers = attrs.get_int("num_layers", 1)
    mode = attrs.get_str("mode")
    output_states = attrs.get_bool("state_outputs", False)
    if mode.startswith("rnn"):
        mode, activation = mode.split("_")
    assert mode in ["rnn", "gru", "lstm"]
    bidirectional = attrs.get_bool("bidirectional", False)
    direct = 2 if bidirectional else 1
    layout = attrs.get_str("layout", "TNC")
    if layout != "TNC":
        raise NotImplementedError("RNN with layout other than TNC is not supported yet")
    num_states = 2 if mode == "lstm" else 1
    assert len(inputs) == num_states + 2

    seq_data = inputs[0]
    concat_weight = inputs[1]
    init_states = inputs[2:]
    # TODO - Assuming seq_len of 1
    seq_len = 1
    assert len(concat_weight) == num_layers * 4 * direct

    ## expr = _infer_type(seq_data)
    ## data_shape = expr.checked_type.shape
    ## seq_len = int(data_shape[0])

    # TODO - This code not exercised yet
    # for idx, state in enumerate(init_states[:]):
    #     if isinstance(state, dict):
    #         node = state
    #         attrs = StrAttrsDict(node.get("attrs", {}))
    #         op_name = node["op"]
    #         # by default, RNN layer uses zeros to initialize states
    #         assert op_name == "_zeros"
    #         shape = attrs.get_int_tuple("shape")
    #         dtype = attrs.get_str("dtype", "float32")
    #         init_layout = attrs.get_str("__layout__")
    #         new_shape = list(shape)
    #         for i, dim in enumerate(shape):
    #             if dim == 0:
    #                 axis = layout.find(init_layout[i])
    #                 assert axis >= 0
    #                 new_shape[i] = int(data_shape[axis])
    #         init_states[idx] = op.zeros(new_shape, dtype)

    weights = []
    bias = []
    states = []
    back_weights = []
    back_bias = []
    back_states = []

    concat_weight_args = list()
    for c in concat_weight:
        concat_weight_args.append(_saved_reshape_inputs[c])
    for i in range(num_layers):
        weights.append([concat_weight_args[i * 2 * direct], concat_weight_args[i * 2 * direct + 1]])
        bias.append(
            [
                concat_weight_args[(num_layers + i) * 2 * direct],
                concat_weight_args[(num_layers + i) * 2 * direct + 1],
            ]
        )
        s = []
        for state in init_states:
            name = _unique_name()
            indices = raf_array(i * direct, dtype="int32")
            _extra_aux_params[name] = indices
            s.append(op.take(state, Symbol.make_var(name_hint=name), axis=0))
        states.append(s)
        if bidirectional:
            back_weights.append(
                [concat_weight_args[i * 2 * direct + 2], concat_weight_args[i * 2 * direct + 3]]
            )
            back_bias.append(
                [
                    concat_weight_args[(num_layers + i) * 2 * direct + 2],
                    concat_weight_args[(num_layers + i) * 2 * direct + 3],
                ]
            )
            s = []
            for state in init_states:
                name = _unique_name()
                indices = raf_array(i * direct + 1, dtype="int32")
                _extra_aux_params[name] = indices
                s.append(op.take(state, Symbol.make_var(name_hint=name), axis=0))
            back_states.append(s)

    xs = list()
    for t in range(seq_len):
        name = _unique_name()
        indices = raf_array(t, dtype="int32")
        _extra_aux_params[name] = indices
        xs.append(op.take(seq_data, Symbol.make_var(name_hint=name), axis=0))

    for l in range(num_layers):
        outputs = []
        back_outputs = []
        for x in xs:
            if mode == "rnn":
                out, new_states = _rnn_cell(x, states[l], *weights[l], *bias[l], activation)
            elif mode == "gru":
                out, new_states = _gru_cell(x, states[l], *weights[l], *bias[l])
            else:  # mode == "lstm"
                out, new_states = _lstm_cell(x, states[l], *weights[l], *bias[l])
            states[l] = new_states
            outputs.append(out)
        if bidirectional:
            for x in reversed(xs):
                if mode == "rnn":
                    out, new_states = _rnn_cell(
                        x, back_states[l], *back_weights[l], *back_bias[l], activation
                    )
                elif mode == "gru":
                    out, new_states = _gru_cell(x, back_states[l], *back_weights[l], *back_bias[l])
                else:  # mode == "lstm"
                    out, new_states = _lstm_cell(x, back_states[l], *back_weights[l], *back_bias[l])
                back_states[l] = new_states
                back_outputs.append(out)
            back_outputs.reverse()
            concat_outputs = []
            for t, out in enumerate(outputs):
                new_out = op.concatenate([out, back_outputs[t]], axis=-1)
                concat_outputs.append(new_out)
            outputs = concat_outputs
        xs = outputs

    ret = [op.stack(outputs, axis=0)]
    if output_states:
        for i in range(num_states):
            inputs = []
            for l, s in enumerate(states):
                inputs.append(s[i])
                if bidirectional:
                    inputs.append(back_states[l][i])
            ret.append(op.stack(inputs, axis=0))
    return ret


_convert_map = {
    "Activation": _mx_activations,
    "BatchNorm": _mx_batch_norm,
    "Convolution": _mx_conv,
    "FullyConnected": _mx_fully_connected,
    "Pooling": _mx_pooling,
    "Reshape": _mx_reshape,
    "RNN": _mx_rnn_layer,
    "elemwise_add": _mx_add,
    "_rnn_param_concat": _mx_rnn_param_concat,
    "expand_dims": _mx_expand_dims,
    "sum": _mx_sum,
    "SwapAxis": _mx_swap_axis,
    "broadcast_mul": _mx_multiply,
    "softmax": _mx_softmax,
    "_contrib_AdaptiveAvgPooling2D": _mx_adaptive_avg_pooling,
    "Flatten": _mx_flatten,
}


def _from_mxnet_impl(symbol, is_train):
    """Convert mxnet symbol to compatible relay Function.
    Migrate from TVM.
    Reconstruct a relay Function by traversing the mxnet symbol.
    Parameters
    ----------
    symbol : mxnet.sym.Symbol
        Incompatible symbol from mxnet.
        The op_name and attrs inside are not always compatible.
    is_train : bool
        Whether in train mode.
    Returns:
    -------
    func : tvm.relay.Function
        Converted relay Function
    """
    assert symbol is not None
    if isinstance(symbol, dict):
        jgraph = symbol
    else:
        jgraph = json.loads(symbol.tojson())
    jnodes = jgraph["nodes"]
    node_map = {}

    for nid, node in enumerate(jnodes):
        children = [node_map[e[0]][e[1]] for e in node["inputs"]]
        attrs = relay.frontend.common.StrAttrsDict(node.get("attrs", {}))
        node_name = node["name"]
        op_name = node["op"]
        if op_name == "null":
            node_map[nid] = [Symbol.make_var(name_hint=node_name)]
        elif op_name in _convert_map:
            if op_name in ["_cond", "_foreach", "_while_loop"]:  # pylint: disable=no-else-raise
                raise NotImplementedError
            else:
                res = _convert_map[op_name](children, attrs, is_train)
            if res is None:
                # defer conversion, used in RNN state initialization
                res = [node]
            if not isinstance(res, (list, Symbol)):
                raise RuntimeError("unexpected type %s" % type(res))
            node_map[nid] = res
        else:
            raise NotImplementedError(
                "Operator {} is not supported in frontend MXNet.".format(op_name)
            )

    outputs = [node_map[e[0]][e[1]] for e in jgraph["heads"]]
    # construct the function body
    body = [x._Symbol__handle for x in outputs]
    if len(body) == 1:
        body = body[0]
    else:
        body = Symbol.make_tuple(body)._Symbol__handle
    body = ExtractBinding(body, [])
    func = relay.Function(relay.analysis.free_vars(body), body)
    return func


def from_mxnet(
    symbol,  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    inputs_name,
    arg_params=None,
    aux_params=None,
):
    """
    Migrate from TVM.
    """
    try:
        import mxnet as mx  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError("{}. MXNet is required to parse symbols.".format(e))

    if isinstance(symbol, mx.sym.Symbol):
        params = {}
        arg_params = arg_params if arg_params else {}
        aux_params = aux_params if aux_params else {}
        for k, v in arg_params.items():
            params[k] = v.asnumpy()
        for k, v in aux_params.items():
            params[k] = v.asnumpy()
        train_func = _from_mxnet_impl(symbol, True)
        infer_func = _from_mxnet_impl(symbol, False)
    elif isinstance(symbol, mx.gluon.HybridBlock):
        if arg_params is not None or aux_params is not None:
            raise ValueError("arg_params and aux_params ae not used when importing HybridBlock")
        params = {}
        for k, v in symbol.collect_params().items():
            params[k] = v.data().asnumpy()
        inputs = []
        for name in inputs_name:
            inputs.append(mx.sym.Variable(name))
        sym = symbol(*inputs)
        if isinstance(sym, (list, tuple)):
            sym = mx.sym.Group(sym)
        train_func = _from_mxnet_impl(sym, True)
        infer_func = _from_mxnet_impl(sym, False)
    elif isinstance(symbol, mx.gluon.Block):
        raise NotImplementedError("Only Hybrid Blocks are supported now.")
    else:
        msg = "mxnet.Symbol or gluon.HybridBlock expected, got {}".format(type(symbol))
        raise ValueError(msg)
    meta_arg_params = dict()
    meta_aux_params = dict()
    for v in train_func.params:
        if v.name_hint in params:
            meta_arg_params[v.name_hint] = ndarray(params[v.name_hint])
        elif v.name_hint in _extra_aux_params:
            meta_aux_params[v.name_hint] = ndarray(_extra_aux_params[v.name_hint])
    for v in infer_func.params:
        if v.name_hint in _extra_aux_params:
            meta_aux_params[v.name_hint] = ndarray(_extra_aux_params[v.name_hint])

    train_mod = raf_module({relay.GlobalVar("main"): train_func})
    infer_mod = raf_module({relay.GlobalVar("main"): infer_func})
    front_model = FrameworkModel(train_mod, infer_mod, meta_arg_params, meta_aux_params)
    return front_model
