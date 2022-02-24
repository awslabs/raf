# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access
import numpy as np
import pytest

import raf
from raf.testing import check_type, run_infer_type, randn
from tvm.relay import TensorType, FuncType, TupleType


# pylint: disable=too-many-locals, attribute-defined-outside-init
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("inputs", [((1, 2500, 6), 0, 0, 1), ((16, 500, 5), 0.95, -1, 0)])
def test_get_valid_counts(inputs, dtype):
    class GetValidCounts(raf.Model):
        def build(self, id_index, score_index):
            self._id_index = id_index
            self._score_index = score_index

        @raf.model.trace
        def forward(self, x, y):
            return raf.get_valid_counts(x, y, self._id_index, self._score_index)

    m_x, _ = randn(inputs[0], dtype=dtype)
    batch_size, num_anchor, _ = inputs[0]
    score_threshold, id_index, score_index = inputs[1], inputs[2], inputs[3]
    np_s = np.array(score_threshold).astype("float32")
    m_s = raf.array(np_s)
    model = GetValidCounts(id_index, score_index)
    # forward
    m_func = model._internal(m_x, m_s).mod["main"]
    m_func = run_infer_type(m_func)
    x_ty = TensorType(inputs[0], dtype=dtype)
    s_ty = TensorType(np_s.shape, dtype="float32")
    valid_count_ty = TensorType((batch_size,), dtype="int32")
    out_tensor_ty = TensorType(inputs[0], dtype=dtype)
    out_indices_ty = TensorType((batch_size, num_anchor), dtype="int32")
    expected_type = FuncType(
        [x_ty, s_ty], TupleType([valid_count_ty, out_tensor_ty, out_indices_ty])
    )
    check_type(m_func, expected_type)


# pylint: disable=import-outside-toplevel, too-many-instance-attributes, too-many-arguments
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("return_indices", [True, False])
def test_non_max_suppression(return_indices, dtype):
    class NonMaxSuppression(raf.Model):
        def build(
            self,
            force_suppress=False,
            top_k=-1,
            coord_start=2,
            score_index=1,
            id_index=0,
            return_indices=True,
            invalid_to_bottom=False,
        ):
            self._force_suppress = force_suppress
            self._top_k = top_k
            self._coord_start = coord_start
            self._score_index = score_index
            self._id_index = id_index
            self._return_indices = return_indices
            self._invalid_to_bottom = invalid_to_bottom

        @raf.model.trace
        def forward(self, data, valid_count, indices, max_output_size, iou_threshold):
            return raf.non_max_suppression(
                data,
                valid_count,
                indices,
                max_output_size,
                iou_threshold,
                self._force_suppress,
                self._top_k,
                self._coord_start,
                self._score_index,
                self._id_index,
                self._return_indices,
                self._invalid_to_bottom,
            )

    model = NonMaxSuppression(return_indices=return_indices)
    # forward
    np_data = np.array(
        [
            [
                [0, 0.8, 1, 20, 25, 45],
                [1, 0.7, 30, 60, 50, 80],
                [0, 0.4, 4, 21, 19, 40],
                [2, 0.9, 35, 61, 52, 79],
                [1, 0.5, 100, 60, 70, 110],
            ]
        ]
    ).astype(dtype)
    np_valid_count = np.array([4]).astype("int32")
    np_indices = np.array([[0, 1, 3, 4, -1]]).astype("int32")
    np_max_output_size = np.array(-1).astype("int32")
    np_iou_threshold = np.array(0.5).astype("float32")
    m_data = raf.array(np_data, dtype=dtype)
    m_valid_count = raf.array(np_valid_count, dtype="int32")
    m_indices = raf.array(np_indices, dtype="int32")
    m_max_output_size = raf.array(np_max_output_size, dtype="int32")
    m_iou_threshold = raf.array(np_iou_threshold, dtype="float32")

    data_ty = TensorType(np_data.shape, dtype=dtype)
    valid_count_ty = TensorType(np_valid_count.shape, dtype="int32")
    indices_ty = TensorType(np_indices.shape, dtype="int32")
    max_output_size_ty = TensorType(np_max_output_size.shape, dtype="int32")
    iou_threshold_ty = TensorType(np_iou_threshold.shape, dtype="float32")
    m_func = model._internal(
        m_data, m_valid_count, m_indices, m_max_output_size, m_iou_threshold
    ).mod["main"]
    m_func = run_infer_type(m_func)

    if return_indices:
        return_data_ty = TensorType(
            np_data.shape[:2], dtype="int32"  # pylint: disable=unsubscriptable-object
        )
        return_indices_ty = TensorType(
            (np_data.shape[0], 1), dtype="int32"  # pylint: disable=unsubscriptable-object
        )
        expected_type = FuncType(
            [data_ty, valid_count_ty, indices_ty, max_output_size_ty, iou_threshold_ty],
            TupleType([return_data_ty, return_indices_ty]),
        )
    else:
        return_data_ty = data_ty
        expected_type = FuncType(
            [data_ty, valid_count_ty, indices_ty, max_output_size_ty, iou_threshold_ty],
            return_data_ty,
        )
    check_type(m_func, expected_type)


@pytest.mark.parametrize(
    "config",
    [((1, 4, 16, 16), (32, 5), (7, 7), 1.0, -1), ((4, 4, 16, 16), (32, 5), (7, 7), 0.5, 2)],
)
@pytest.mark.parametrize("mode", ["avg", "max"])
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_roi_align(config, mode, layout):
    class RoiAlign(raf.Model):
        def build(self, pooled_size, spatial_scale, sample_ratio, layout, mode):
            self.pooled_size = pooled_size
            self.spatial_scale = spatial_scale
            self.sample_ratio = sample_ratio
            self.layout = layout
            self.mode = mode

        @raf.model.trace
        def forward(self, data, rois):
            return raf.roi_align(
                data,
                rois,
                self.pooled_size,
                self.spatial_scale,
                self.sample_ratio,
                self.layout,
                self.mode,
            )

    data_shape, rois_shape, pooled_size, spatial_scale, sample_ratio = config
    np_data = np.random.uniform(size=data_shape).astype("float32")
    np_rois = np.random.uniform(size=rois_shape).astype("float32")
    m_data = raf.array(np_data)
    m_rois = raf.array(np_rois)
    model = RoiAlign(pooled_size, spatial_scale, sample_ratio, layout, mode)

    m_func = model._internal(m_data, m_rois).mod["main"]
    m_func = run_infer_type(m_func)
    data_ty = TensorType(data_shape, dtype="float32")
    rois_ty = TensorType(rois_shape, dtype="float32")
    if layout == "NCHW":
        out_tensor_ty = TensorType((rois_shape[0], data_shape[1], *pooled_size), "float32")
    else:
        out_tensor_ty = TensorType((rois_shape[0], *pooled_size, data_shape[3]), "float32")
    expected_type = FuncType([data_ty, rois_ty], out_tensor_ty)
    check_type(m_func, expected_type)


if __name__ == "__main__":
    pytest.main([__file__])
