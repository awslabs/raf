#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CALL(func)                                           \
  do {                                                            \
    cudaError_t e = (func);                                       \
    CHECK(e == cudaSuccess) << "CUDA: " << cudaGetErrorString(e); \
  } while (false)

namespace mnm {
namespace common {
namespace cuda {

template <typename T, int value>
inline const void* const_addr() {
  static const T a = static_cast<T>(value);
  return static_cast<const void*>(&a);
}

}  // namespace cuda
}  // namespace common
}  // namespace mnm
