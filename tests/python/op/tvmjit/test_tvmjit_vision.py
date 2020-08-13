import numpy as np
import pytest
import mnm


def get_ctx_list():
    ret = ["cpu"]
    if mnm.build.with_cuda():
        ret.append("cuda")
    return ret


def randn(shape, *, ctx="cpu", dtype="float32"):
    x = np.random.randn(*shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    return m_x, n_x


def check(m_x, n_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    np.testing.assert_allclose(m_x, n_x, rtol=rtol, atol=atol)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("inputs", [
    ((1, 2500, 6), 0, 0, 1),
    ((1, 2500, 5), -1, -1, 0),
    ((3, 1000, 6), 0.55, 1, 0),
    ((16, 500, 5), 0.95, -1, 0)
]) # pylint: disable=too-many-locals
def test_get_valid_counts(inputs, ctx):
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
    mx_out = mnm.get_valid_counts(m_x, score_threshold, id_index, score_index)
    check(mx_out[0], np_out1, rtol=1e-3, atol=1e-04)
    if ctx == "cpu":
        # tvm get_valid_count for cuda doesn't do data rearrangement
        check(mx_out[1], np_out2, rtol=1e-3, atol=1e-04)
        check(mx_out[2], np_out3, rtol=1e-3, atol=1e-04)


@pytest.mark.parametrize("ctx", get_ctx_list())
def test_nms(ctx):
    def verify_nms(mx_data, mx_valid_count, mx_indices, mx_max_output_size, np_result, dshape, # pylint: disable=too-many-arguments
                   np_indices_result, force_suppress=False, top_k=2):
        res = mnm.non_max_suppression(mx_data, mx_valid_count, mx_indices, mx_max_output_size,
                                      force_suppress=force_suppress, top_k=top_k,
                                      return_indices=False)
        check(res, np_result, rtol=1e-5, atol=1e-07)
        if ctx == 'cpu':
            res_indices = mnm.non_max_suppression(mx_data, mx_valid_count, mx_indices,
                                                  mx_max_output_size, force_suppress=force_suppress,
                                                  top_k=top_k, return_indices=True)
            assert res_indices[0].shape == (dshape[0], dshape[1]) and \
                   res_indices[0].dtype == "int32"
            assert res_indices[1].shape == (dshape[0], 1) and res_indices[1].dtype == "int32"
            check(res_indices[0], np_indices_result, rtol=1e-5, atol=1e-07)


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

    mx_data = mnm.array(np_data, ctx=ctx)
    mx_valid_count = mnm.array(np_valid_count, ctx=ctx)
    mx_indices = mnm.array(np_indices, ctx=ctx)
    mx_max_output_size = mnm.array(np_max_output_size, ctx=ctx)

    verify_nms(mx_data, mx_valid_count, mx_indices, mx_max_output_size, np_result, dshape,
               np_indices_result, force_suppress=True)
    verify_nms(mx_data, mx_valid_count, mx_indices, mx_max_output_size, np_result, dshape,
               np_indices_result)


if __name__ == "__main__":
    pytest.main([__file__])
