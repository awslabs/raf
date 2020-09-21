/*!
 * Copyright (c) 2019 by Contributors
 * \file src/common/cuda_utils.h
 * \brief Utilities for cuda
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CALL(func)                                           \
  do {                                                            \
    cudaError_t e = (func);                                       \
    CHECK(e == cudaSuccess) << "CUDA: " << cudaGetErrorString(e); \
  } while (false)

template <typename T, int value,
          typename std::enable_if<std::is_same<T, __half>::value, int>::type = 0>
inline const void* const_typed_addr() {
  float tmp = static_cast<float>(value);
  static const T a = static_cast<T>(tmp);
  return static_cast<const void*>(&a);
}

template <typename T, int value,
          typename std::enable_if<!std::is_same<T, __half>::value, int>::type = 0>
inline const void* const_typed_addr() {
  static const T a = static_cast<T>(value);
  return static_cast<const void*>(&a);
}
