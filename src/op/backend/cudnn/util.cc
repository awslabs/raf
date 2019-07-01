#include <algorithm>

#include <mnm/value.h>

#include "./util.h"

namespace mnm {
namespace op {
namespace backend {
namespace cudnn {

using rly::Array;
using rly::Attrs;
using rly::Integer;
using value::Value;

int MakeStride(int n, int* dims, int* stride) {
  int carry = 1;
  for (int i = n - 1; i >= 0; --i) {
    if (stride) stride[i] = carry;
    carry *= dims[i];
  }
  return carry;
}

cudnnDataType_t DType2CudnnType(DType dt) {
  switch (dt.code) {
    case kDLInt: {
      if (dt.bits == 8)
        return CUDNN_DATA_INT8;
      else if (dt.bits == 32)
        return CUDNN_DATA_INT32;
      LOG(FATAL) << "NotImplementedError: " << dt.c_str();
    }
    case kDLUInt:
      if (dt.bits == 8) return CUDNN_DATA_UINT8;
      LOG(FATAL) << "NotImplementedError: " << dt.c_str();
    case kDLFloat:
      if (dt.bits == 16) return CUDNN_DATA_HALF;
      if (dt.bits == 32) return CUDNN_DATA_FLOAT;
      if (dt.bits == 64) return CUDNN_DATA_DOUBLE;
      LOG(FATAL) << "NotImplementedError: " << dt.c_str();
  }
  LOG(FATAL) << "NotImplementedError: " << dt.c_str();
  throw;
}

cudnnMoreThan4DTensor::cudnnMoreThan4DTensor(const DLTensor* dl)
    : dt(DType2CudnnType(dl->dtype)), n(std::max(4, dl->ndim)) {
  for (int i = 0; i < n - dl->ndim; ++i) dims[i] = 1;
  for (int i = 0; i < dl->ndim; ++i) dims[i + n - dl->ndim] = dl->shape[i];
  MakeStride(n, dims, strides);
}

void ArrayIntegerToIntPtr(Array<Integer> src, int* dest) {
  for (size_t i = 0; i < src.size(); ++i) {
    dest[i] = src[i];
  }
}

}  // namespace cudnn
}  // namespace backend
}  // namespace op
}  // namespace mnm
