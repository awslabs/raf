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


if __name__ == "__main__":
    pytest.main([__file__])
