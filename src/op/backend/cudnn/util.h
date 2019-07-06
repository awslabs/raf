#pragma once

#include <cudnn.h>

#include <mnm/base.h>
#include <mnm/rly.h>

#include "../../../common/cuda.h"

#define CUDNN_CALL(func)                                                      \
  {                                                                           \
    cudnnStatus_t e = (func);                                                 \
    CHECK_EQ(e, CUDNN_STATUS_SUCCESS) << "cuDNN: " << cudnnGetErrorString(e); \
  }

namespace mnm {
namespace op {
namespace backend {
namespace cudnn {

class CUDNNThreadEntry {
 public:
  CUDNNThreadEntry();
  static CUDNNThreadEntry* ThreadLocal();

 public:
  cudnnHandle_t handle{nullptr};
};

/*
 * Make stride and calculate the size of the given array.
 * The stride can be null. In this case, compute the array size only.
 */
int MakeStride(int n, int* dims, int* stride);

cudnnDataType_t DType2CudnnType(DType dt);

struct cudnnMoreThan4DTensor {
  cudnnMoreThan4DTensor() {
  }
  cudnnMoreThan4DTensor(const DLTensor* dl);

  cudnnDataType_t dt;
  int n;
  int dims[8];
  int strides[8];
};

void ArrayIntegerToIntPtr(mnm::rly::Array<mnm::rly::Integer> src, int* dest);

template <typename T, int value>
const void* const_addr() {
  static const T a = static_cast<T>(value);
  return static_cast<const void*>(&a);
}

template <int value>
const void* getAlphaBeta(cudnnDataType_t dt) {
  switch (dt) {
    case CUDNN_DATA_FLOAT:
      return const_addr<float, value>();
    case CUDNN_DATA_HALF:
      return const_addr<float, value>();
    case CUDNN_DATA_DOUBLE:
      return const_addr<double, value>();
    case CUDNN_DATA_INT8:
      return const_addr<char, value>();
    case CUDNN_DATA_INT8x32:
#if CUDNN_VERSION >= 710
    case CUDNN_DATA_UINT8:
      LOG(FATAL) << "NotImplementedError: " << dt << " no default alpha beta!";
    case CUDNN_DATA_UINT8x4:
      LOG(FATAL) << "NotImplementedError: " << dt << " no default alpha beta!";
#endif
    case CUDNN_DATA_INT32:
      LOG(FATAL) << "NotImplementedError: " << dt << " no default alpha beta!";
    default:
      LOG(FATAL) << "NotImplementedError: " << dt << " no default alpha beta!";
  }
  throw;
}

}  // namespace cudnn
}  // namespace backend
}  // namespace op
}  // namespace mnm
