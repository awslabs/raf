/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/common/cuda_utils.h
 * \brief Utilities for cuda
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "raf/device.h"

// Wrap CUDA runtime API calls with error checking.
#define CUDA_CALL(func)                                                             \
  do {                                                                              \
    cudaError_t e = (func);                                                         \
    CHECK(e == cudaSuccess) << "CUDA error " << e << ": " << cudaGetErrorString(e); \
  } while (false)

// Wrap CUDA runtime API calls with error checking, but ignore the CUDA driver shutdown error.
// Since we maintain CUDADeviceAPI in a static instance, the order of calling its deconstructor
// is uncertain. Thus, it is possible that the CUDA driver shutdown error is thrown when calling
// CUDA runtime APIs (i.e., cudaFree) in other object's deconstruction (i.e., Memory).
// We use this macro in such case to avoid the error.
#define CUDA_CALL_IF_DRIVER_IS_LOADED(func)                     \
  do {                                                          \
    cudaError_t e = (func);                                     \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)    \
        << "CUDA error " << e << ": " << cudaGetErrorString(e); \
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

template <int value>
inline const void* const_addr(cudaDataType_t dt) {
  switch (dt) {
    case CUDA_R_8I:
      return const_typed_addr<int8_t, value>();
    case CUDA_R_8U:
      return const_typed_addr<uint8_t, value>();
    case CUDA_R_16F:
      return const_typed_addr<__half, value>();
    case CUDA_R_32F:
      return const_typed_addr<float, value>();
    case CUDA_R_64F:
      return const_typed_addr<double, value>();
    default:
      LOG(FATAL) << "Not supported data type!";
      throw;
  }
  LOG(FATAL) << "ValueError: Unknown error!\n";
  throw;
}

template <typename T>
inline std::shared_ptr<void> shared_typed_addr(float value) {
  return std::make_shared<T>(value);
}

inline std::shared_ptr<void> shared_addr(cudaDataType_t dt, float value) {
  switch (dt) {
    case CUDA_R_8I:
      return shared_typed_addr<int8_t>(value);
    case CUDA_R_8U:
      return shared_typed_addr<uint8_t>(value);
    case CUDA_R_16F:
      return shared_typed_addr<__half>(value);
    case CUDA_R_32F:
      return shared_typed_addr<float>(value);
    case CUDA_R_64F:
      return shared_typed_addr<double>(value);
    default:
      LOG(FATAL) << "Not supported data type!";
      throw;
  }
  LOG(FATAL) << "ValueError: Unknown error!\n";
  throw;
}

namespace raf {

template <>
inline DType::operator cudaDataType_t() const {
  switch (code) {
    case kDLInt:
      if (bits == 8) return CUDA_R_8I;
      break;
    case kDLUInt:
      if (bits == 8) return CUDA_R_8U;
      break;
    case kDLFloat:
      if (bits == 16) return CUDA_R_16F;
      if (bits == 32) return CUDA_R_32F;
      if (bits == 64) return CUDA_R_64F;
    default:
      LOG(FATAL) << "NotImplementedError: " << c_str();
  }
  throw;
}

}  // namespace raf
