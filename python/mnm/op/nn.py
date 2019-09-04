from typing import Union, Tuple

from .._core.ndarray import ndarray
from .._core.op import create_op
from ._typing import array_like, scalar, type_check
from .._ffi._tvm import _make_node

_conv2d = create_op("mnm.op.conv2d")

def conv2d(img: ndarray, ker: ndarray,
        padding: Union[int, Tuple[int]]=0,
        dilation: Union[int, Tuple[int]]=1,
        stride: Union[int, Tuple[int]]=1,
        groups: int=1) -> ndarray:
    """
    Apply a convolution kernel on the given image tensor.

    Parameters
    ----------
    img: ndarray
        The input image.
    ker: ndarray
        The kernel tensor.
    padding: int or a 2-d tuple
        The implicit padding on width and height of the input.
    dilation: int or a 2-d tuple
        The spacing between each kernel points.
    stride: int or a 2-d tuple
        The stride of the convolution kernel.
    groups: int
        Split input into groups. The input channels should be divisible by this number.
    """

    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    if isinstance(stride, int):
        stride = (stride, stride)

    @type_check
    def impl(img: ndarray, ker: ndarray, **kwargs) -> ndarray:
        return _conv2d(args=(img, ker), attrs=_make_node('mnm.attrs.ConvAttrs', **kwargs))

    kwargs = {
            'padding': padding,
            'dilation': dilation,
            'stride': stride,
            'groups': groups
    }

    return impl(img, ker, **kwargs)
