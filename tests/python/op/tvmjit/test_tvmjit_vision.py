# pylint: disable=attribute-defined-outside-init,no-self-use
import numpy as np
import pytest
import mnm
from mnm.testing import get_ctx_list, randn, check, run_vm_model


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("inputs", [
    ((1, 2500, 6), 0, 0, 1),
    ((1, 2500, 5), -1, -1, 0),
    ((3, 1000, 6), 0.55, 1, 0),
    ((16, 500, 5), 0.95, -1, 0)
]) # pylint: disable=too-many-locals
def test_get_valid_counts(inputs, ctx):
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return mnm.get_valid_counts(x, score_threshold, id_index, score_index)

    dtype = "float32"
    m_x, np_data = randn(inputs[0], ctx=ctx)
    batch_size, num_anchor, elem_length = inputs[0]
    score_threshold, id_index, score_index = inputs[1], inputs[2], inputs[3]
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
    m_out = model(m_x)
    v_out = run_vm_model(model, ctx, [m_x])
    check(m_out[0], np_out1, rtol=1e-3, atol=1e-04)
    check(v_out[0], np_out1, rtol=1e-3, atol=1e-04)
    if ctx == "cpu":
        # tvm get_valid_count for cuda doesn't do data rearrangement
        check(m_out[1], np_out2, rtol=1e-3, atol=1e-04)
        check(m_out[2], np_out3, rtol=1e-3, atol=1e-04)
        check(v_out[1], np_out2, rtol=1e-3, atol=1e-04)
        check(v_out[2], np_out3, rtol=1e-3, atol=1e-04)


@pytest.mark.parametrize("ctx", get_ctx_list())
def test_nms(ctx):
    class TestModel(mnm.Model):
        def build(self, force_suppress, top_k, return_indices):
            self._force_suppress = force_suppress
            self._top_k = top_k
            self._return_indices = return_indices

        @mnm.model.trace
        def forward(self, data, valid_count, indices, max_output_size):
            return mnm.non_max_suppression(data, valid_count, indices, max_output_size,
                                           force_suppress=self._force_suppress, top_k=self._top_k,
                                           return_indices=self._return_indices)

    def verify_nms(m_data, m_valid_count, m_indices, m_max_output_size, np_result, dshape, # pylint: disable=too-many-arguments
                   np_indices_result, force_suppress=False, top_k=2):
        model = TestModel(force_suppress, top_k, False)
        m_out = model(m_data, m_valid_count, m_indices, m_max_output_size)
        v_out = run_vm_model(model, ctx, [m_data, m_valid_count, m_indices, m_max_output_size])
        check(m_out, np_result, rtol=1e-5, atol=1e-07)
        check(v_out, np_result, rtol=1e-5, atol=1e-07)
        if ctx == 'cpu':
            model = TestModel(force_suppress, top_k, True)
            m_out = model(m_data, m_valid_count, m_indices, m_max_output_size)
            v_out = run_vm_model(model, ctx, [m_data, m_valid_count, m_indices, m_max_output_size])
            assert m_out[0].shape == (dshape[0], dshape[1]) and \
                   m_out[0].dtype == "int32"
            assert m_out[1].shape == (dshape[0], 1) and m_out[1].dtype == "int32"
            assert v_out[0].shape == (dshape[0], dshape[1]) and \
                   v_out[0].dtype == "int32"
            assert v_out[1].shape == (dshape[0], 1) and v_out[1].dtype == "int32"
            check(m_out[0], np_indices_result, rtol=1e-5, atol=1e-07)
            check(v_out[0], np_indices_result, rtol=1e-5, atol=1e-07)


    np_data = np.array([[[0, 0.8, 1, 20, 25, 45], [1, 0.7, 30, 60, 50, 80],
                         [0, 0.4, 4, 21, 19, 40], [2, 0.9, 35, 61, 52, 79],
                         [1, 0.5, 100, 60, 70, 110]]]).astype("float32")
    np_valid_count = np.array([4]).astype("int32")
    np_indices = np.array([[0, 1, 3, 4, -1]]).astype("int32")
    np_max_output_size = np.array(-1).astype("int32")

    np_result = np.array([[[2, 0.9, 35, 61, 52, 79], [0, 0.8, 1, 20, 25, 45],
                           [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1, -1]]])
    np_indices_result = np.array([[4, 0, -1, -1, -1]])
    num_anchors = 5
    dshape = (1, num_anchors, 6)

    m_data = mnm.array(np_data, ctx=ctx)
    m_valid_count = mnm.array(np_valid_count, ctx=ctx)
    m_indices = mnm.array(np_indices, ctx=ctx)
    m_max_output_size = mnm.array(np_max_output_size, ctx=ctx)

    verify_nms(m_data, m_valid_count, m_indices, m_max_output_size, np_result, dshape,
               np_indices_result, force_suppress=True)
    verify_nms(m_data, m_valid_count, m_indices, m_max_output_size, np_result, dshape,
               np_indices_result)


if __name__ == "__main__":
    pytest.main([__file__])
