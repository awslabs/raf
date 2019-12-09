/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/backend/cublas/cublas_utils.h
 * \brief Helper functions for cuBLAS
 */
#pragma once
#include <cublas_v2.h>
#include "mnm/base.h"
#include "mnm/enum_base.h"
#include "mnm/ir.h"
#include "../../../common/cuda_utils.h"

#define CUBLAS_CALL(func)                                                        \
  do {                                                                           \
    cublasStatus_t e = (func);                                                   \
    CHECK_EQ(e, CUBLAS_STATUS_SUCCESS) << "cublas: " << cublasGetErrorString(e); \
  } while (false)

namespace mnm {

template <>
inline DType::operator cudaDataType_t() const {
  switch (code) {
    case kDLInt: {
      if (bits == 8) return CUDA_R_8I;
      LOG(FATAL) << "NotImplementedError: " << c_str();
    }
    case kDLUInt:
      if (bits == 8) return CUDA_R_8U;
      LOG(FATAL) << "NotImplementedError: " << c_str();
    case kDLFloat:
      if (bits == 16) return CUDA_R_16F;
      if (bits == 32) return CUDA_R_32F;
      if (bits == 64) return CUDA_R_64F;
      LOG(FATAL) << "NotImplementedError: " << c_str();
  }
  LOG(FATAL) << "NotImplementedError: " << c_str();
  throw;
}

namespace op {
namespace backend {
namespace cublas {

inline const char* cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    default:
      LOG(FATAL) << "ValueError: Unknown error!\n";
      throw;
  }
  LOG(FATAL) << "ValueError: Unknown error!\n";
  throw;
}

class CUBlasThreadEntry {
 public:
  CUBlasThreadEntry();
  static CUBlasThreadEntry* ThreadLocal();

 public:
  cublasHandle_t handle{nullptr};
};

template <typename T, int value>
inline const void* const_typed_addr() {
  static const T a = static_cast<T>(value);
  return static_cast<const void*>(&a);
}

template<int value>
const void *const_typed_addr(cudaDataType_t dt) {
  switch (dt) {
    case CUDA_R_8I: return const_typed_addr<int8_t, value>();
    case CUDA_R_8U: return const_typed_addr<uint8_t, value>();
    case CUDA_R_16F: return const_typed_addr<float, value>();
    case CUDA_R_32F: return const_typed_addr<float, value>();
    case CUDA_R_64F: return const_typed_addr<double, value>();
    default:
      LOG(FATAL) << "Not supported data type!";
      throw;
  }
  LOG(FATAL) << "ValueError: Unknown error!\n";
  throw;
}

}  // namespace cublas
}  // namespace backend
}  // namespace op
}  // namespace mnm
