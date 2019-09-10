from typing import Union, Tuple
from .._core.ndarray import ndarray
from .._core.op import get_op
from .._ffi._tvm import _make_node
from ._typing import array_like, scalar
from ._util import int2tuple
from .imports import array as _array
from ._typing import _ARG_TYPE_GUARDS, _RET_TYPE_GUARDS


def avg_pool2d(input: ndarray,
               kernel: Union[int, Tuple[int, int]],
               stride: Union[int, Tuple[int, int]],
               padding: Union[int, Tuple[int, int]]=0,
               dilation: Union[int, Tuple[int, int]]=1,
               include_pad: bool=True,
               ceil_mode: bool=False) -> ndarray:
    """ Reserved for doc string... """
    kernel = int2tuple(kernel)
    stride = kernel if stride is None else int2tuple(stride)
    padding = int2tuple(padding)
    dilation = int2tuple(dilation)
    attr_args = {
        "kernel": kernel,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "include_pad": include_pad,
        "ceil_mode": ceil_mode,
    }
    input = _ARG_TYPE_GUARDS[ndarray](input, "input")
    f = get_op("mnm.op.avg_pool2d")
    res = f(eager=True,
            args=[input],
            attrs=_make_node("mnm.attrs.AvgPoolAttrs", **attr_args))
    res = _RET_TYPE_GUARDS[ndarray](res, "return value")
    return res

def batch_flatten(x: ndarray) -> ndarray:
    """ Reserved for doc string... """
    x = _ARG_TYPE_GUARDS[ndarray](x, "x")
    f = get_op("mnm.op.batch_flatten")
    res = f(eager=True,
            args=[x],
            attrs=None)
    res = _RET_TYPE_GUARDS[ndarray](res, "return value")
    return res

def conv2d(input: ndarray,
           kernel: ndarray,
           stride: Union[int, Tuple[int, int]]=1,
           padding: Union[int, Tuple[int, int]]=0,
           dilation: Union[int, Tuple[int, int]]=1,
           groups: int=1) -> ndarray:
    """ Reserved for doc string... """
    stride = int2tuple(stride)
    padding = int2tuple(padding)
    dilation = int2tuple(dilation)
    attr_args = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
    }
    input = _ARG_TYPE_GUARDS[ndarray](input, "input")
    kernel = _ARG_TYPE_GUARDS[ndarray](kernel, "kernel")
    f = get_op("mnm.op.conv2d")
    res = f(eager=True,
            args=[input, kernel],
            attrs=_make_node("mnm.attrs.ConvAttrs", **attr_args))
    res = _RET_TYPE_GUARDS[ndarray](res, "return value")
    return res

def dropout(x: ndarray,
            dropout: float,
            seed: int) -> ndarray:
    """ Reserved for doc string... """
    x = _ARG_TYPE_GUARDS[ndarray](x, "x")
    f = get_op("mnm.op.dropout")
    res = f(eager=True,
            args=[x],
            attrs=None)
    res = _RET_TYPE_GUARDS[ndarray](res, "return value")
    return res

def max_pool2d(input: ndarray,
               kernel: Union[int, Tuple[int, int]],
               stride: Union[int, Tuple[int, int]],
               padding: Union[int, Tuple[int, int]]=0,
               dilation: Union[int, Tuple[int, int]]=1,
               ceil_mode: bool=False) -> ndarray:
    """ Reserved for doc string... """
    kernel = int2tuple(kernel)
    stride = kernel if stride is None else int2tuple(stride)
    padding = int2tuple(padding)
    dilation = int2tuple(dilation)
    attr_args = {
        "kernel": kernel,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
    }
    input = _ARG_TYPE_GUARDS[ndarray](input, "input")
    f = get_op("mnm.op.max_pool2d")
    res = f(eager=True,
            args=[input],
            attrs=_make_node("mnm.attrs.MaxPoolAttrs", **attr_args))
    res = _RET_TYPE_GUARDS[ndarray](res, "return value")
    return res

def batch_norm2d(x: ndarray,
                 mean: ndarray,
                 variance: ndarray,
                 scale: ndarray=None,
                 bias: ndarray=None,
                 eps: float=1e-05,
                 momentum: float=0.1,
                 is_training: float=False) -> ndarray:
    """ Reserved for doc string... """
    scale = _array([1] * x.shape[1], dtype=x.dtype, ctx=x.ctx) if scale is None else scale
    bias = _array([0] * x.shape[1], dtype=x.dtype, ctx=x.ctx) if bias is None else bias
    attr_args = {
        "eps": eps,
        "momentum": momentum,
        "is_training": is_training,
    }
    x = _ARG_TYPE_GUARDS[ndarray](x, "x")
    mean = _ARG_TYPE_GUARDS[ndarray](mean, "mean")
    variance = _ARG_TYPE_GUARDS[ndarray](variance, "variance")
    scale = _ARG_TYPE_GUARDS[ndarray](scale, "scale")
    bias = _ARG_TYPE_GUARDS[ndarray](bias, "bias")
    f = get_op("mnm.op.batch_norm2d")
    res = f(eager=True,
            args=[x, mean, variance, scale, bias],
            attrs=_make_node("mnm.attrs.BatchNormAttrs", **attr_args))
    res = _RET_TYPE_GUARDS[ndarray](res, "return value")
    return res

def softmax(x: ndarray,
            axis: int) -> ndarray:
    """ Reserved for doc string... """
    attr_args = {
        "axis": axis,
    }
    x = _ARG_TYPE_GUARDS[ndarray](x, "x")
    f = get_op("mnm.op.softmax")
    res = f(eager=True,
            args=[x],
            attrs=_make_node("mnm.attrs.SoftmaxAttrs", **attr_args))
    res = _RET_TYPE_GUARDS[ndarray](res, "return value")
    return res

def log_softmax(x: ndarray,
                axis: int) -> ndarray:
    """ Reserved for doc string... """
    attr_args = {
        "axis": axis,
    }
    x = _ARG_TYPE_GUARDS[ndarray](x, "x")
    f = get_op("mnm.op.log_softmax")
    res = f(eager=True,
            args=[x],
            attrs=_make_node("mnm.attrs.SoftmaxAttrs", **attr_args))
    res = _RET_TYPE_GUARDS[ndarray](res, "return value")
    return res

def relu(x: ndarray) -> ndarray:
    """ Reserved for doc string... """
    x = _ARG_TYPE_GUARDS[ndarray](x, "x")
    f = get_op("mnm.op.relu")
    res = f(eager=True,
            args=[x],
            attrs=None)
    res = _RET_TYPE_GUARDS[ndarray](res, "return value")
    return res

def sigmoid(x: ndarray) -> ndarray:
    """ Reserved for doc string... """
    x = _ARG_TYPE_GUARDS[ndarray](x, "x")
    f = get_op("mnm.op.sigmoid")
    res = f(eager=True,
            args=[x],
            attrs=None)
    res = _RET_TYPE_GUARDS[ndarray](res, "return value")
    return res

def tanh(x: ndarray) -> ndarray:
    """ Reserved for doc string... """
    x = _ARG_TYPE_GUARDS[ndarray](x, "x")
    f = get_op("mnm.op.tanh")
    res = f(eager=True,
            args=[x],
            attrs=None)
    res = _RET_TYPE_GUARDS[ndarray](res, "return value")
    return res

