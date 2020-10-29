import pytest
import mnm
from mnm.testing import check_type, run_infer_type, randn
from tvm.relay import TensorType, FuncType, TupleType


# pylint: disable=too-many-locals, attribute-defined-outside-init
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("inputs", [
    ((1, 2500, 6), 0, 0, 1),
    ((1, 2500, 5), -1, -1, 0),
    ((3, 1000, 6), 0.55, 1, 0),
    ((16, 500, 5), 0.95, -1, 0)
])
def test_get_valid_counts(inputs, dtype):

    class GetValidCounts(mnm.Model):
        def build(self, score_threshold, id_index, score_index):
            self._score_threshold = score_threshold
            self._id_index = id_index
            self._score_index = score_index

        @mnm.model.trace
        def forward(self, x):
            return mnm.get_valid_counts(x, self._score_threshold,
                                        self._id_index, self._score_index)

    m_x, _ = randn(inputs[0], dtype=dtype)
    batch_size, num_anchor, _ = inputs[0]
    score_threshold, id_index, score_index = inputs[1], inputs[2], inputs[3]
    model = GetValidCounts(score_threshold, id_index, score_index)
    # forward
    m_func = model.get_relay_func(m_x)
    m_func = run_infer_type(m_func)
    x_ty = TensorType(inputs[0], dtype=dtype)
    valid_count_ty = TensorType((batch_size,), dtype="int32")
    out_tensor_ty = TensorType(inputs[0], dtype=dtype)
    out_indices_ty = TensorType((batch_size, num_anchor), dtype="int32")
    expected_type = FuncType([x_ty], TupleType([valid_count_ty, out_tensor_ty, out_indices_ty]))
    check_type(m_func, expected_type)


# pylint: disable=import-outside-toplevel, too-many-instance-attributes, too-many-arguments
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("return_indices", [True, False])
def test_non_max_suppression(return_indices, dtype):
    import numpy as np

    class NonMaxSuppression(mnm.Model):
        def build(self, iou_threshold=0.5, force_suppress=False, top_k=-1, coord_start=2,
                  score_index=1, id_index=0, return_indices=True, invalid_to_bottom=False):
            self._iou_threshold = iou_threshold
            self._force_suppress = force_suppress
            self._top_k = top_k
            self._coord_start = coord_start
            self._score_index = score_index
            self._id_index = id_index
            self._return_indices = return_indices
            self._invalid_to_bottom = invalid_to_bottom

        @mnm.model.trace
        def forward(self, data, valid_count, indices, max_output_size):
            return mnm.non_max_suppression(data, valid_count, indices, max_output_size,
                                           self._iou_threshold, self._force_suppress, self._top_k,
                                           self._coord_start, self._score_index, self._id_index,
                                           self._return_indices, self._invalid_to_bottom)

    model = NonMaxSuppression(return_indices=return_indices)
    # forward
    np_data = np.array([[[0, 0.8, 1, 20, 25, 45], [1, 0.7, 30, 60, 50, 80],
                         [0, 0.4, 4, 21, 19, 40], [2, 0.9, 35, 61, 52, 79],
                         [1, 0.5, 100, 60, 70, 110]]]).astype(dtype)
    np_valid_count = np.array([4]).astype("int32")
    np_indices = np.array([[0, 1, 3, 4, -1]]).astype("int32")
    np_max_output_size = np.array(-1).astype("int32")
    m_data = mnm.array(np_data, dtype=dtype)
    m_valid_count = mnm.array(np_valid_count, dtype="int32")
    m_indices = mnm.array(np_indices, dtype="int32")
    m_max_output_size = mnm.array(np_max_output_size, dtype="int32")

    data_ty = TensorType(np_data.shape, dtype=dtype)
    valid_count_ty = TensorType(np_valid_count.shape, dtype="int32")
    indices_ty = TensorType(np_indices.shape, dtype="int32")
    max_output_size_ty = TensorType(np_max_output_size.shape, dtype="int32")
    m_func = model.get_relay_func(m_data, m_valid_count, m_indices, m_max_output_size)
    m_func = run_infer_type(m_func)

    if return_indices:
        return_data_ty = TensorType(np_data.shape[:2], dtype="int32") # pylint: disable=unsubscriptable-object
        return_indices_ty = TensorType((np_data.shape[0], 1), dtype="int32") # pylint: disable=unsubscriptable-object
        expected_type = FuncType([data_ty, valid_count_ty, indices_ty, max_output_size_ty],
                                 TupleType([return_data_ty, return_indices_ty]))
    else:
        return_data_ty = data_ty
        expected_type = FuncType([data_ty, valid_count_ty, indices_ty, max_output_size_ty],
                                 return_data_ty)
    check_type(m_func, expected_type)

if __name__ == "__main__":
    pytest.main([__file__])
