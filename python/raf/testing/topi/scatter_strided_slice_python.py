# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-arguments

"""scatter_strided_slice in python"""


def scatter_strided_slice_python(data, src, begin, end, strides, slice_mode="end", axes=None):
    """Python version of scatter strided slice operator.

    Parameters
    ----------
    data : numpy.ndarray
        Input data

    begin : list
        Beginning of the slices.

    end : list
        End of the slices.

    strides : list
        The stride of each slice.

    slice_mode : str, optional
        The slice mode [end, size].
        end: The default slice mode, ending indices for the slice.
        size: The input strides will be ignored, input end in this mode indicates
              the sizeof a slice starting at the location specified by begin. If end[i] is -1,
              all remaining elements in that dimension are included in the slice.

    axes : list, optional
        Axes along which slicing is applied

    Returns
    -------
    result : numpy.ndarray
        The sliced result.
    """
    strides = [] if strides is None else strides
    if axes is not None:
        rank = len(data.shape)
        new_begin = [0] * rank
        new_end = [data.shape[i] for i in range(rank)]
        new_strides = [1] * rank

        for i, axis in enumerate(axes):
            new_begin[axis] = begin[i]
            new_end[axis] = end[i]
            if len(strides) > i:
                new_strides[axis] = strides[i]

        begin = new_begin
        end = new_end
        strides = new_strides

    slices = []
    for i in range(len(data.shape)):
        new_stride = None
        if slice_mode == "end" and i < len(strides):
            new_stride = strides[i]

        new_begin = begin[i] if i < len(begin) else None
        if i >= len(end):
            new_end = None
        elif slice_mode == "size":
            if end[i] < 0:
                new_end = None
            else:
                new_end = new_begin + end[i]
        else:
            new_end = end[i]

        slices.append(slice(new_begin, new_end, new_stride))

    data[tuple(slices)] = src
    return data
