# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,no-self-use
import numpy as np
import torch
import torchvision
import pytest
import raf
from raf.testing import get_testable_devices, randn, check, run_vm_model

import tvm.topi.testing


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize(
    "inputs", [((1, 2500, 6), 0, 0, 1), ((16, 500, 5), 0.95, -1, 0)]
)  # pylint: disable=too-many-locals
def test_get_valid_counts(inputs, device):
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            return raf.get_valid_counts(x, y, id_index, score_index)

    dtype = "float32"
    m_x, np_data = randn(inputs[0], device=device)
    batch_size, num_anchor, elem_length = inputs[0]
    score_threshold, id_index, score_index = inputs[1], inputs[2], inputs[3]
    np_s = np.array(score_threshold).astype("float32")
    m_s = raf.array(np_s, device=device)
    np_out1 = np.zeros(shape=(batch_size,))
    np_out2 = np.zeros(shape=inputs[0]).astype(dtype)
    np_out3 = np.zeros(shape=(batch_size, num_anchor))
    for i in range(batch_size):
        np_out1[i] = 0
        inter_idx = 0
        for j in range(num_anchor):
            score = np_data[i, j, score_index]
            if score > score_threshold and (id_index < 0 or np_data[i, j, id_index] >= 0):
                for k in range(elem_length):
                    np_out2[i, inter_idx, k] = np_data[i, j, k]
                np_out1[i] += 1
                np_out3[i, inter_idx] = j
                inter_idx += 1
            if j >= np_out1[i]:
                for k in range(elem_length):
                    np_out2[i, j, k] = -1.0
                np_out3[i, j] = -1
    model = TestModel()
    m_out = model(m_x, m_s)
    v_out = run_vm_model(model, device, [m_x, m_s])
    check(m_out[0], np_out1, rtol=1e-3, atol=1e-04)
    check(v_out[0], np_out1, rtol=1e-3, atol=1e-04)
    if device == "cpu":
        # tvm get_valid_count for cuda doesn't do data rearrangement
        check(m_out[1], np_out2, rtol=1e-3, atol=1e-04)
        check(m_out[2], np_out3, rtol=1e-3, atol=1e-04)
        check(v_out[1], np_out2, rtol=1e-3, atol=1e-04)
        check(v_out[2], np_out3, rtol=1e-3, atol=1e-04)


@pytest.mark.parametrize("device", get_testable_devices())
def test_nms(device):
    # pylint: disable=too-many-locals
    class TestModel(raf.Model):
        def build(self, force_suppress, top_k, return_indices):
            self._force_suppress = force_suppress
            self._top_k = top_k
            self._return_indices = return_indices

        @raf.model.trace
        def forward(self, data, valid_count, indices, max_output_size, iou_threshold):
            # pylint: disable=too-many-arguments
            return raf.non_max_suppression(
                data,
                valid_count,
                indices,
                max_output_size,
                iou_threshold,
                force_suppress=self._force_suppress,
                top_k=self._top_k,
                return_indices=self._return_indices,
            )

    def verify_nms(
        m_data,
        m_valid_count,
        m_indices,
        m_max_output_size,
        m_iou_threshold,
        np_result,
        dshape,
        np_indices_result,
        force_suppress=False,
        top_k=2,
    ):
        # pylint: disable=too-many-arguments
        model = TestModel(force_suppress, top_k, False)
        m_out = model(m_data, m_valid_count, m_indices, m_max_output_size, m_iou_threshold)
        v_out = run_vm_model(
            model, device, [m_data, m_valid_count, m_indices, m_max_output_size, m_iou_threshold]
        )
        check(m_out, np_result, rtol=1e-5, atol=1e-07)
        check(v_out, np_result, rtol=1e-5, atol=1e-07)
        if device == "cpu":
            model = TestModel(force_suppress, top_k, True)
            m_out = model(m_data, m_valid_count, m_indices, m_max_output_size, m_iou_threshold)
            v_out = run_vm_model(
                model,
                device,
                [m_data, m_valid_count, m_indices, m_max_output_size, m_iou_threshold],
            )
            assert m_out[0].shape == (dshape[0], dshape[1]) and m_out[0].dtype == "int32"
            assert m_out[1].shape == (dshape[0], 1) and m_out[1].dtype == "int32"
            assert v_out[0].shape == (dshape[0], dshape[1]) and v_out[0].dtype == "int32"
            assert v_out[1].shape == (dshape[0], 1) and v_out[1].dtype == "int32"
            check(m_out[0], np_indices_result, rtol=1e-5, atol=1e-07)
            check(v_out[0], np_indices_result, rtol=1e-5, atol=1e-07)

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
    ).astype("float32")
    np_valid_count = np.array([4]).astype("int32")
    np_indices = np.array([[0, 1, 3, 4, -1]]).astype("int32")
    np_max_output_size = np.array(-1).astype("int32")
    np_iou_threshold = np.array(0.5).astype("float32")

    np_result = np.array(
        [
            [
                [2, 0.9, 35, 61, 52, 79],
                [0, 0.8, 1, 20, 25, 45],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ]
        ]
    )
    np_indices_result = np.array([[4, 0, -1, -1, -1]])
    num_anchors = 5
    dshape = (1, num_anchors, 6)

    m_data = raf.array(np_data, device=device)
    m_valid_count = raf.array(np_valid_count, device=device)
    m_indices = raf.array(np_indices, device=device)
    m_max_output_size = raf.array(np_max_output_size, device=device)
    m_iou_threshold = raf.array(np_iou_threshold, device=device)

    verify_nms(
        m_data,
        m_valid_count,
        m_indices,
        m_max_output_size,
        m_iou_threshold,
        np_result,
        dshape,
        np_indices_result,
        force_suppress=True,
    )
    verify_nms(
        m_data,
        m_valid_count,
        m_indices,
        m_max_output_size,
        m_iou_threshold,
        np_result,
        dshape,
        np_indices_result,
    )


@pytest.mark.xfail(reason="See apache/tvm#12126")
@pytest.mark.parametrize(
    "config",
    [((1, 4, 16, 16), (32, 5), (7, 7), 1.0, -1), ((4, 4, 16, 16), (32, 5), (7, 7), 0.5, 2)],
)
@pytest.mark.parametrize("mode", ["avg", "max"])
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
@pytest.mark.parametrize("device", get_testable_devices())
def test_roi_align(config, mode, layout, device):
    # pylint: disable=too-many-locals, not-callable
    class TestModel(raf.Model):
        def build(
            self, pooled_size, spatial_scale, sample_ratio, layout, mode
        ):  # pylint: disable=too-many-arguments
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

    def verify_roi_align(
        data_shape, rois_shape, pooled_size, spatial_scale, sample_ratio, mode, layout
    ):  # pylint: disable=too-many-arguments
        if layout == "NCHW":
            _, _, in_size, _ = data_shape
            ref_func = tvm.topi.testing.roi_align_nchw_python
        else:
            _, in_size, _, _ = data_shape
            ref_func = tvm.topi.testing.roi_align_nhwc_python
        np_data = np.random.uniform(size=data_shape).astype("float32")
        np_rois = np.random.uniform(size=rois_shape).astype("float32") * in_size
        np_rois[:, 0] = np.random.randint(low=0, high=data_shape[0], size=rois_shape[0])
        np_res = ref_func(
            np_data,
            np_rois,
            pooled_size=pooled_size,
            spatial_scale=spatial_scale,
            sample_ratio=sample_ratio,
            mode=mode,
        )

        # forward
        m_data = raf.array(np_data, device=device)
        m_rois = raf.array(np_rois, device=device)
        m_data.requires_grad = True

        model = TestModel(pooled_size, spatial_scale, sample_ratio, layout, mode)
        m_out = model(m_data, m_rois)
        v_out = run_vm_model(model, device, [m_data, m_rois])
        check(m_out, np_res, rtol=1e-5, atol=1e-05)
        check(v_out, np_res, rtol=1e-5, atol=1e-05)

        # backward
        np_dy = np.ones(m_out.shape, dtype="float32")
        m_out.backward(raf.array(np_dy, device=device))
        if layout == "NCHW" and mode == "avg":
            t_data = torch.tensor(np_data)
            t_rois = torch.tensor(np_rois)
            t_data.requires_grad = True
            t_y = torchvision.ops.roi_align(
                t_data, t_rois, pooled_size, spatial_scale, sample_ratio
            )
            t_y.backward(torch.tensor(np_dy))
            check(m_data.grad, t_data.grad)

    verify_roi_align(*config, mode, layout)


if __name__ == "__main__":
    pytest.main([__file__])
