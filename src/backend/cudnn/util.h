#pragma once

#include <cudnn.h>

#include <dlpack/dlpack.h>

#include <mnm/rly.h>
#include <mnm/value.h>
#include <mnm/opattrs/activation.h>
#include <mnm/opattrs/pooling.h>


#define CHECK_CALL(func_call, SUCCESS)                        \
  do {                                                        \
    cudnnStatus_t status = func_call;                         \
    CHECK(status == SUCCESS) << "CUDNN: " << status << "\n";  \
  } while (false)


#define CUDNN_CALL(func_call) CHECK_CALL(func_call, CUDNN_STATUS_SUCCESS)
#define CUDA_CALL(func_call) CHECK_CALL(func_call, cudaSuccess)


namespace mnm {
namespace op {
namespace backend {
namespace cudnn {

/*
 * Make stride and calculate the size of the given array.
 * The stride can be null. In this case, compute the array size only.
 */
int MakeStride(int n, int *dims, int *stride);

cudnnDataType_t DType2CudnnType(DType dt);
cudnnPoolingMode_t ToCUDNNPoolingMethod(attrs::PoolingMethod pm);
cudnnActivationMode_t ToCUDNNActivationMethod(attrs::ActivationMethod am);

struct cudnnMoreThan4DTensor {
  cudnnMoreThan4DTensor() {}
  cudnnMoreThan4DTensor(const DLTensor *dl);

  cudnnDataType_t dt;
  int n;
  int dims[8];
  int strides[8];
};

void ArrayIntegerToIntPtr(mnm::rly::Array<mnm::rly::Integer> src, int *dest);

template<typename T, int value> const void *const_addr() {
  static const T a = static_cast<T>(value);
  return static_cast<const void*>(&a);
}

template <int value>
const void *getAlphaBeta(cudnnDataType_t dt) {

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
}

} // namespace cudnn
} // namespace backend
} // namespace op
} // namespace mnm
