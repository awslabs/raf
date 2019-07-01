#include <algorithm>

#include "util.h"

namespace mnm {

using rly::Array;
using rly::Attrs;
using rly::Integer;
using value::Value;

namespace op {
namespace backend {
namespace cudnn {

int MakeStride(int n, int *dims, int *stride) {
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
      if (dt.bits == 8)
        return CUDNN_DATA_UINT8;
      LOG(FATAL) << "NotImplementedError: " << dt.c_str();
    case kDLFloat:
      if (dt.bits == 16)
        return CUDNN_DATA_HALF;
      if (dt.bits == 32)
        return CUDNN_DATA_FLOAT;
      if (dt.bits == 64)
        return CUDNN_DATA_DOUBLE;
      LOG(FATAL) << "NotImplementedError: " << dt.c_str();
  }
  LOG(FATAL) << "NotImplementedError: " << dt.c_str();
}

cudnnPoolingMode_t ToCUDNNPoolingMethod(attrs::PoolingMethod pm) {
  switch (pm) {
  case mnm::op::attrs::PoolingMethod::AvgIncludePadding:
    return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  case mnm::op::attrs::PoolingMethod::AvgExcludePadding:
    return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  case mnm::op::attrs::PoolingMethod::Max:
    return CUDNN_POOLING_MAX;
  }
}

cudnnActivationMode_t ToCUDNNActivationMethod(attrs::ActivationMethod am) {
  switch (am) {
  case attrs::ActivationMethod::Sigmoid:
    return CUDNN_ACTIVATION_SIGMOID;
  case attrs::ActivationMethod::Relu:
    return CUDNN_ACTIVATION_RELU;
  case attrs::ActivationMethod::TanH:
    return CUDNN_ACTIVATION_TANH;
  case attrs::ActivationMethod::ClipRelu:
    return CUDNN_ACTIVATION_CLIPPED_RELU;
  case attrs::ActivationMethod::Elu:
    return CUDNN_ACTIVATION_ELU;
#if CUDNN_VERSION >= 710
  case attrs::ActivationMethod::Identity:
    return CUDNN_ACTIVATION_IDENTITY;
#endif
  }
}

cudnnMoreThan4DTensor::cudnnMoreThan4DTensor(const DLTensor *dl) : dt(DType2CudnnType(dl->dtype)), n(std::max(4, dl->ndim)) {
  for (int i = 0; i < n - dl->ndim; ++i)
    dims[i] = 1;
  for (int i = 0; i < dl->ndim; ++i)
    dims[i + n - dl->ndim] = dl->shape[i];
  MakeStride(n, dims, strides);
}

void ArrayIntegerToIntPtr(Array<Integer> src, int *dest) {
  for (size_t i = 0; i < src.size(); ++i)
    dest[i] = src[i];
}

} // namespace cudnn
} // namespace op
} // namespace backend
} // namespace mnm
