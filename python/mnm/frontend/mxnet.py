# pylint: disable=invalid-name, no-else-return, protected-access, too-many-lines, no-member
"""MXNet symbol frontend."""
import json

from mnm._core.ndarray import Symbol, ndarray
from mnm._ffi.model import RunModel
from mnm._ffi.pass_ import ExtractBinding
from mnm._lib import relay
from mnm._op import sym as op
from mnm.model.trace import _unwrap


def _mx_conv(inputs, attrs):
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
            res = op.add(res, inputs[2])
        return [res]
    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) == 2:
        return _mx_conv2d(inputs, attrs)
    else:
        raise NotImplementedError


def _mx_fully_connected(inputs, attrs):  # pylint: disable=unused-argument
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


def _mx_batch_norm(inputs, attrs):
    new_attrs = {}
    new_attrs["eps"] = attrs.get_float("eps", 0.001)
    new_attrs["momentum"] = attrs.get_float("momentum", 0.9)
    res = op.batch_norm_infer(x=inputs[0],
                              w=inputs[1],
                              b=inputs[2],
                              running_mean=inputs[3],
                              running_var=inputs[4],
                              **new_attrs)
    return [res]


def _mx_pooling(inputs, attrs):
    global_pool = attrs.get_bool("global_pool", False)
    pool_type = attrs.get_str("pool_type")

    def _pool2d(new_op, is_avg):
        kernel_size = attrs.get_int_tuple("kernel")
        if len(kernel_size) != 2:
            raise ValueError('Only 2D kernels are supported for operator Pool2D.')
        new_attrs = {}
        new_attrs["kernel"] = kernel_size
        new_attrs["stride"] = attrs.get_int_tuple("stride", (1, 1))
        new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0))
        new_attrs["ceil_mode"] = (attrs.get_str("pooling_convention", "valid") == "full")
        if is_avg:
            new_attrs["include_pad"] = attrs.get_bool("count_include_pad", True)
        return [new_op(inputs[0], **new_attrs)]

    if pool_type == "max":
        if global_pool:
            raise NotImplementedError
        return _pool2d(op.max_pool2d, False)
    if pool_type == "avg":
        if global_pool:
            raise NotImplementedError
            # return [op.avg_pool2d(inputs[0], 4, 4)]
        return _pool2d(op.avg_pool2d, True)
    raise NotImplementedError


def _mx_activations(inputs, attrs):
    act_type = attrs.get_str("act_type")
    assert len(inputs) == 1
    if act_type == "relu":
        return [op.relu(inputs[0])]
    else:
        raise NotImplementedError


def _mx_add(inputs, attrs):  # pylint: disable=unused-argument
    return [op.add(inputs[0], inputs[1])]

_convert_map = {
    'Activation': _mx_activations,
    'BatchNorm': _mx_batch_norm,
    'Convolution': _mx_conv,
    'FullyConnected': _mx_fully_connected,
    'Pooling': _mx_pooling,
    'elemwise_add': _mx_add
}


def _from_mxnet_impl(symbol):
    """Convert mxnet symbol to compatible relay Function.
    Migrate from TVM.

    Reconstruct a relay Function by traversing the mxnet symbol.

    Parameters
    ----------
    symbol : mxnet.sym.Symbol
        Incompatible symbol from mxnet.
        The op_name and attrs inside are not always compatible.

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
            if op_name in ['_cond', '_foreach', '_while_loop']:  # pylint: disable=no-else-raise
                raise NotImplementedError
            else:
                res = _convert_map[op_name](children, attrs)
            if res is None:
                # defer conversion, used in RNN state initialization
                res = [node]
            if not isinstance(res, (list, Symbol)):
                raise RuntimeError("unexpected type %s" % type(res))
            node_map[nid] = res
        else:
            raise NotImplementedError(
                'Operator {} is not supported in frontend MXNet.'.format(op_name))

    outputs = [node_map[e[0]][e[1]] for e in jgraph["heads"]]
    # construct the function body
    body = [x._Symbol__handle for x in outputs]
    if len(body) == 1:
        body = body[0]
    else:
        body = Symbol.make_tuple(body)._Symbol__handle
    body = ExtractBinding(body)
    func = relay.Function(relay.analysis.free_vars(body), body)
    return func


def from_mxnet(symbol,  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
               inputs_name,
               ctx=None,
               mode='infer',  # pylint: disable=unused-argument
               arg_params=None,
               aux_params=None):
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
        func = _from_mxnet_impl(symbol)
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
        func = _from_mxnet_impl(sym)
    elif isinstance(symbol, mx.gluon.Block):
        raise NotImplementedError("Only Hybrid Blocks are supported now.")
    else:
        msg = "mxnet.Symbol or gluon.HybridBlock expected, got {}".format(type(symbol))
        raise ValueError(msg)
    params_input = []
    for v in func.params:
        if v.name_hint in params:
            params_input.append(ndarray(params[v.name_hint], ctx=ctx)._ndarray__handle)

    def new_pyfunc(*args, **kwargs):  # pylint: disable=unused-argument
        assert len(args) == len(inputs_name)
        func_inputs = [arg._ndarray__handle for arg in args] + params_input
        result = _unwrap(RunModel(func, func_inputs))
        return result
    return new_pyfunc
