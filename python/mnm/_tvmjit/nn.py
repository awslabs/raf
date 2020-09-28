# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
from .._lib import register_compute
from .._lib import generic_func
from .._lib import tvm as _tvm
from .._lib import _reg
from .._lib import strategy

_topi = _tvm.topi  # pylint: disable=invalid-name,no-member

@register_compute("mnm.op.take_dx")
def take_dx_compute(attrs, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=invalid-name
    # pylint: disable=unused-variable
    x, y, dy, indices = inputs
    axis = int(attrs.axis)
    idim = len(indices.shape)
    # infer axis when negative
    dim = len(x.shape)
    if -dim < axis < 0:
        axis = dim + axis
    shape = dy.shape[:axis + idim] + [x.shape[axis],] + dy.shape[axis + idim:]
    A = _tvm.te.compute(shape, lambda *idx:
                        _tvm.tir.if_then_else(idx[axis + idim] == indices[idx[axis: axis + idim]],
                                              dy[idx[:axis + idim] + idx[axis + idim + 1:]],
                                              _tvm.tir.const(0, dy.dtype)))
    B = _topi.sum(A, axis=tuple(range(axis, axis + idim)))
    return [B]


_reg.register_injective_schedule("mnm.op.take_dx")

_reg.register_strategy("mnm.op.dense", strategy.dense_strategy)

_reg.register_strategy("mnm.op.batch_matmul", strategy.batch_matmul_strategy)

def compute_matmul_general(attr, inputs, output_type,
                           transpose_a=False, transpose_b=False):
    # pylint: disable=unused-argument
    if len(inputs) == 2:
        data, weight = inputs[0], inputs[1]
    else:
        raise ValueError("Invalid input")
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim dense"
    return [_topi.matmul(data, weight, transp_a=transpose_a, transp_b=transpose_b)]

@register_compute("mnm.op.matmul")
def compute_matmul(attr, inputs, output_type):
    return compute_matmul_general(attr, inputs, output_type, transpose_a=False, transpose_b=False)

@register_compute("mnm.op.matmul_tn")
def compute_matmul_tn(attr, inputs, output_type):
    return compute_matmul_general(attr, inputs, output_type, transpose_a=True, transpose_b=False)

@register_compute("mnm.op.matmul_nt")
def compute_matmul_nt(attr, inputs, output_type):
    return compute_matmul_general(attr, inputs, output_type, transpose_a=False, transpose_b=True)

@register_compute("mnm.op.matmul_tt")
def compute_matmul_tt(attr, inputs, output_type):
    return compute_matmul_general(attr, inputs, output_type, transpose_a=True, transpose_b=True)

_reg.register_injective_schedule("mnm.op.matmul")
_reg.register_injective_schedule("mnm.op.matmul_tn")
_reg.register_injective_schedule("mnm.op.matmul_nt")
_reg.register_injective_schedule("mnm.op.matmul_tt")

_reg.register_strategy("mnm.op.softmax", strategy.softmax_strategy)

@register_compute("mnm.op.softmax_dx")
def compute_softmax_dx(attr, inputs, output_type):
    # pylint: disable=unused-argument, unused-variable, invalid-name
    x, y, dy = inputs[0], inputs[1], inputs[2]
    axis = attr.axis
    softmax_out = _topi.nn.softmax(x, axis=axis)
    grads = _tvm.te.gradient(softmax_out, [x], head=dy)
    return grads

# TODO(@XIAO-XIA): complete the cuda schedule after the implementation of auto schedule
_reg.register_injective_schedule("mnm.op.softmax_dx")

_reg.register_schedule("mnm.op.avg_pool2d", strategy.schedule_pool)
# TODO(@XIAO-XIA): cuda schedule should be complemented after the implementation of auto schedule

@register_compute("mnm.op.avg_pool2d_dx")
def compute_avg_pool2d_dx(attr, inputs, output_type):
    # pylint: disable=too-many-locals, unbalanced-tuple-unpacking, invalid-name, unused-argument, unused-variable
    strides, padding, pool_size, layout = _topi.util.get_const_tuple(attr.strides), \
                                          _topi.util.get_const_tuple(attr.padding), \
                                          _topi.util.get_const_tuple(attr.pool_size), \
                                          attr.layout
    ceil_mode, count_include_pad = attr.ceil_mode, attr.count_include_pad
    assert layout == "NCHW"
    pt, pl = padding
    padding = (pt, pl, pt, pl)

    x, y, dy = inputs
    res = _topi.nn.pool(x, kernel=pool_size, stride=strides, padding=padding, pool_type='avg',
                        ceil_mode=ceil_mode, layout=layout, count_include_pad=count_include_pad)
    grads = _tvm.te.gradient(res, [x], head=dy)
    return grads

_reg.register_schedule("mnm.op.avg_pool2d_dx", strategy.schedule_pool_grad)

_reg.register_schedule("mnm.op.max_pool2d", strategy.schedule_pool)
# TODO(@XIAO-XIA): cuda schedule should be complemented after the implementation of auto schedule

@register_compute("mnm.op.max_pool2d_dx")
def compute_max_pool2d_dx(attr, inputs, output_type):
    # pylint: disable=too-many-locals, unbalanced-tuple-unpacking, invalid-name, unused-argument, unused-variable
    strides, padding, pool_size, layout = _topi.util.get_const_tuple(attr.strides), \
                                          _topi.util.get_const_tuple(attr.padding), \
                                          _topi.util.get_const_tuple(attr.pool_size), \
                                          attr.layout
    ceil_mode = attr.ceil_mode
    assert layout == "NCHW"
    pt, pl = padding
    padding = (pt, pl, pt, pl)

    x, y, dy = inputs
    res = _topi.nn.pool(x, kernel=pool_size, stride=strides, padding=padding,
                        pool_type='max', ceil_mode=ceil_mode, layout=layout)
    grads = _tvm.te.gradient(res, [x], head=dy)
    return grads

_reg.register_schedule("mnm.op.max_pool2d_dx", strategy.schedule_pool_grad)

@register_compute("mnm.op.log_softmax")
def compute_log_softmax(attr, inputs, output_type):
    # pylint: disable=unused-argument
    x = inputs[0]
    axis = attr.axis
    softmax_out = _topi.nn.softmax(x, axis=axis)
    log_softmax_out = _topi.log(softmax_out)
    return [log_softmax_out]

_reg.register_injective_schedule("mnm.op.log_softmax")

@register_compute("mnm.op.log_softmax_dx")
def compute_log_softmax_dx(attr, inputs, output_type):
    # pylint: disable=unused-argument, unused-variable, invalid-name
    x, y, dy = inputs[0], inputs[1], inputs[2]
    axis = attr.axis
    softmax_out = _topi.nn.softmax(x, axis=axis)
    log_softmax_out = _topi.log(softmax_out)
    grads = _tvm.te.gradient(log_softmax_out, [x], head=dy)
    return grads

# TODO(@XIAO-XIA): complete the cuda schedule after the implementation of auto schedule
_reg.register_injective_schedule("mnm.op.log_softmax_dx")

@register_compute("mnm.op.relu_dx")
def compute_relu_dx(attr, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=invalid-name
    # pylint: disable=unused-variable
    X, dy, y = inputs[0], inputs[1], inputs[2]
    R = _topi.nn.relu(X)
    grads = _tvm.te.gradient(R, [X], head=dy)
    return grads

_reg.register_injective_schedule("mnm.op.relu_dx")

@register_compute("mnm.op.layer_norm")
def compute_layer_norm(attr, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=too-many-locals
    x = inputs[0]
    axis, eps = _topi.util.get_const_int(attr.axis), _tvm.tir.const(attr.epsilon, dtype=x.dtype)
    ndim = len(x.shape)
    if axis < 0:
        axis = ndim + axis
    count = _tvm.tir.const(1, dtype=x.dtype)
    count *= x.shape[axis]
    reduce_axes = [axis]
    x_sum = _topi.sum(x, reduce_axes, keepdims=True)
    x_mean = _topi.divide(x_sum, count)
    sq_diff = _topi.power(_topi.subtract(x, x_mean), 2)
    sq_diff_sum = _topi.sum(sq_diff, reduce_axes, keepdims=True)
    x_var = _topi.divide(sq_diff_sum, count)
    denominator = _topi.sqrt(_topi.add(x_var, eps))
    out = _topi.divide(_topi.subtract(x, x_mean), denominator)
    return [out]

@generic_func
def schedule_layer_norm(attrs, outs, target):
    # pylint: disable=unused-argument
    with target:
        return _topi.generic.schedule_injective(outs)

@schedule_layer_norm.register(["cuda", "gpu"])
def schedule_layer_norm_cuda(attrs, outs, target):
    # pylint: disable=unused-argument
    # pylint: disable=invalid-name
    with target:
        out = outs[0]
        s = _topi.cuda.schedule_injective(outs)
        # fuse axes and split into bx and tx then bind
        scheduled_ops = []
        num_thread = 64
        def bind_axes(s, out):
            bind_tags = ["comm_reduce", "group_conv2d_nchw"]
            if isinstance(out.op, _tvm.te.ComputeOp) and out.op.tag in bind_tags \
                and out.op not in scheduled_ops:
                scheduled_ops.append(out.op)
                fused = s[out].fuse(*s[out].op.axis)
                bx, tx = s[out].split(fused, factor=num_thread)
                s[out].bind(bx, _tvm.te.thread_axis("blockIdx.x"))
                s[out].bind(tx, _tvm.te.thread_axis("threadIdx.x"))
            for inp in out.op.input_tensors:
                bind_axes(s, inp)
        bind_axes(s, out)
        return s

_reg.register_schedule("mnm.op.layer_norm", schedule_layer_norm)

@register_compute("mnm.op.layer_norm_dx")
def compute_layer_norm_dx(attr, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=too-many-locals
    # pylint: disable=unused-variable
    x, y, dy = inputs
    axis, eps = _topi.util.get_const_int(attr.axis), _tvm.tir.const(attr.epsilon, dtype=x.dtype)
    ndim = len(x.shape)
    if axis < 0:
        axis = ndim + axis

    count = x.shape[axis]
    reduce_axes = [axis]
    x_sum = _topi.sum(x, reduce_axes, keepdims=True)
    x_mean = _topi.divide(x_sum, count)
    sq_diff = _topi.power(_topi.subtract(x, x_mean), 2)
    sq_diff_sum = _topi.sum(sq_diff, reduce_axes, keepdims=True)
    x_var = _topi.divide(sq_diff_sum, count)
    denominator = _topi.sqrt(_topi.add(x_var, eps))
    xmu = _topi.subtract(x, x_mean)

    bar_x = _topi.divide(xmu, denominator)
    w = _topi.divide(dy, denominator)
    w_sum = _topi.sum(w, reduce_axes, keepdims=True)
    mean_w = _topi.divide(w_sum, count)
    w_times_bar_x = _topi.multiply(w, bar_x)
    w_times_bar_x_sum = _topi.sum(w_times_bar_x, reduce_axes, keepdims=True)
    mean_w_times_bar_x = _topi.divide(w_times_bar_x_sum, count)
    dx = _topi.subtract(w, mean_w)
    dx = _topi.subtract(dx, _topi.multiply(bar_x, mean_w_times_bar_x))
    return [dx]

_reg.register_schedule("mnm.op.layer_norm_dx", schedule_layer_norm)

_reg.register_strategy("mnm.op.conv2d", strategy.conv2d_strategy)

def _get_pad_tuple(padding, kernel):
    """Common code to get the pad option

    Parameters
    ----------
    padding : int or str
        Padding size, or ['VALID', 'SAME']

    kernel : tuple of int
        Conv kernel size

    Returns
    -------
    pad_top : int
        Padding size on top

    pad_left : int
        Padding size on left

    pad_down : int
        Padding size on down.

    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            pad_h = padding[0] * 2
            pad_w = padding[1] * 2
        elif len(padding) == 4:
            return padding[0], padding[1], padding[2], padding[3]
        else:
            raise ValueError("Size of padding can only be 2 or 4")
    elif isinstance(padding, int):
        pad_h = pad_w = padding * 2
    elif padding == "VALID":
        pad_h = 0
        pad_w = 0
    elif padding == "SAME":
        pad_h = kernel[0] - 1
        pad_w = kernel[1] - 1
    else:
        raise ValueError("Unknown padding option %s" % padding)
    pad_top = (pad_h + 1) // 2
    pad_left = (pad_w + 1) // 2
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left

def declaration_conv2d_transpose_impl(data, kernel, strides, padding, out_dtype, output_padding):
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # pylint: disable=invalid-name
    """Implementation of conv2d transpose"""
    data_pad, kernel_transform = \
        _topi.nn.conv2d_transpose_nchw_preprocess(data, kernel, strides, padding, out_dtype, (0, 0))
    batch, in_c, in_h, in_w = data_pad.shape
    out_c, _, filter_h, filter_w = kernel_transform.shape

    # convolution stage
    out_c = _topi.nn.simplify(out_c)

    out_h = _topi.nn.simplify(in_h - filter_h + 1 + output_padding[0])
    out_w = _topi.nn.simplify(in_w - filter_w + 1 + output_padding[1])
    dc = _tvm.te.reduce_axis((0, in_c), name='dc')
    dh = _tvm.te.reduce_axis((0, filter_h), name='dh')
    dw = _tvm.te.reduce_axis((0, filter_w), name='dw')

    Output = _tvm.te.compute(
        (batch, out_c, out_h, out_w),
        lambda b, c, h, w: _tvm.tir.sum(
            data_pad[b, dc, h+dh, w+dw].astype(out_dtype) *
            kernel_transform[c, dc, dh, dw].astype(out_dtype),
            axis=[dc, dh, dw]), tag="conv2d_transpose_nchw")

    return Output

@register_compute("mnm.op.conv2d_dx")
def compute_conv2d_dx(attr, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    # pylint: disable=too-many-locals
    # pylint: disable=invalid-name
    # pylint: disable=unbalanced-tuple-unpacking
    strides, padding, dilation, layout = _topi.util.get_const_tuple(attr.strides), \
                                         _topi.util.get_const_tuple(attr.padding), \
                                         _topi.util.get_const_tuple(attr.dilation), \
                                         attr.data_layout
    assert layout == "NCHW"
    assert dilation == (1, 1), "not support dilate now"
    assert attr.groups == 1, "only support groups == 1 for now"

    W, y, dy = inputs[0], inputs[1], inputs[2]
    X = _tvm.te.placeholder(shape=attr.kernel_size, dtype=dy.dtype)
    R = _topi.nn.conv2d(X, W, strides, padding, dilation, layout)

    # TODO: we can also leverage tvm's tensor-level autodiff
    # grads = _tvm.gradient(R, [X], head=dy)

    data_shape = _topi.util.get_const_tuple(X.shape)
    weight_shape = _topi.util.get_const_tuple(W.shape)
    _, _, grad_h, grad_w = _topi.util.get_const_tuple(R.shape)

    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = _get_pad_tuple(
        _topi.util.get_const_tuple(attr.padding), (filter_h, filter_w))
    stride_h, stride_w = _topi.util.get_const_tuple(attr.strides)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    backward_data = declaration_conv2d_transpose_impl(dy, W, strides, padding,
                                                      out_dtype=dy.dtype,
                                                      output_padding=output_padding)
    out = backward_data

    return [out]

_reg.register_schedule("mnm.op.conv2d_dx", schedule_layer_norm)

@register_compute("mnm.op.conv2d_dw")
def compute_conv2d_dw(attr, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    # pylint: disable=too-many-locals
    # pylint: disable=invalid-name
    # pylint: disable=unbalanced-tuple-unpacking
    strides, padding, dilation, layout = _topi.util.get_const_tuple(attr.strides), \
                                         _topi.util.get_const_tuple(attr.padding), \
                                         _topi.util.get_const_tuple(attr.dilation), \
                                         attr.data_layout
    assert layout == "NCHW"
    assert dilation == (1, 1), "not support dilate now"
    assert attr.groups == 1, "only support groups == 1 for now"
    X, y, dy = inputs[0], inputs[1], inputs[2]

    W = _tvm.te.placeholder(shape=attr.kernel_size, dtype=X.dtype)
    R = _topi.nn.conv2d(X, W, strides, padding, dilation, layout)
    # TODO: we can also leverage tvm's tensor-level autodiff
    # grads = _tvm.gradient(R, [W], head=dy)

    data_shape = _topi.util.get_const_tuple(X.shape)
    weight_shape = _topi.util.get_const_tuple(W.shape)
    _, _, grad_h, grad_w = _topi.util.get_const_tuple(R.shape)

    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape
    dilation_h, dilation_w = dilation

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = _get_pad_tuple(
        _topi.util.get_const_tuple(attr.padding), (filter_h, filter_w))
    stride_h, stride_w = _topi.util.get_const_tuple(attr.strides)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w

    dy = _topi.transform.tile(dy, [1, in_channel // attr.groups, 1, 1])
    dy_h, dy_w = dy.shape[2], dy.shape[3]
    # batch * oc * ic // groups, 1, oh, ow
    dy = _topi.transform.reshape(dy, [batch*out_channel*in_channel//attr.groups, 1, dy_h, dy_w])
    X = _topi.transform.reshape(X, [1, batch*in_channel, in_h, in_w])  # 1, batch * ic, ih, iw

    backward_weight = _topi.nn.group_conv2d_nchw(X, dy,
                                                 stride=dilation,
                                                 padding=padding,
                                                 dilation=strides,
                                                 groups=in_channel * batch)
    # infer shape of backward_weight
    padded_weight_grad_h = (in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom) \
                           // dilation_h + 1
    padded_weight_grad_w = (in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right) \
                           // dilation_w + 1
    backward_weight = _topi.transform.reshape(backward_weight,
                                              [batch, in_channel // attr.groups, out_channel,
                                               padded_weight_grad_h, padded_weight_grad_w])
    backward_weight = _topi.sum(backward_weight, axis=0)
    backward_weight = _topi.transform.transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = _topi.transform.strided_slice(backward_weight, begin=[0, 0, 0, 0],
                                                        end=[None, None, filter_h, filter_w])

    return [backward_weight]

_reg.register_schedule("mnm.op.conv2d_dw", schedule_layer_norm)
