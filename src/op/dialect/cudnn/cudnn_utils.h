/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cudnn/cudnn_utils.h
 * \brief Helper functions for cuDNN
 */
#pragma once
#include <cudnn.h>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include <numeric>
#include "raf/device.h"
#include "raf/op.h"
#include "raf/enum_base.h"
#include "raf/ir.h"
#include "raf/value.h"
#include "../../../common/cuda_utils.h"
#include "../../../common/shape_utils.h"
#include "../tvm/tvm_utils.h"

#define CUDNN_CALL(func)                                                      \
  do {                                                                        \
    cudnnStatus_t e = (func);                                                 \
    CHECK_EQ(e, CUDNN_STATUS_SUCCESS) << "cuDNN: " << cudnnGetErrorString(e); \
  } while (false)

namespace raf {

template <>
inline DType::operator cudnnDataType_t() const {
  switch (code) {
    case kDLInt: {
      if (bits == 8)
        return CUDNN_DATA_INT8;
      else if (bits == 32)
        return CUDNN_DATA_INT32;
      LOG(FATAL) << "NotImplementedError: " << c_str();
    }
    case kDLUInt:
      if (bits == 8) return CUDNN_DATA_UINT8;
      LOG(FATAL) << "NotImplementedError: " << c_str();
    case kDLFloat:
      if (bits == 16) return CUDNN_DATA_HALF;
      if (bits == 32) return CUDNN_DATA_FLOAT;
      if (bits == 64) return CUDNN_DATA_DOUBLE;
      LOG(FATAL) << "NotImplementedError: " << c_str();
  }
  LOG(FATAL) << "NotImplementedError: " << c_str();
  throw;
}

namespace op {
namespace cudnn {

inline std::string cudnnDataTypeToString(cudnnDataType_t dtype) {
  switch (dtype) {
    case CUDNN_DATA_FLOAT:
      return "CUDNN_DATA_FLOAT";
    case CUDNN_DATA_DOUBLE:
      return "CUDNN_DATA_DOUBLE";
    case CUDNN_DATA_HALF:
      return "CUDNN_DATA_HALF";
    case CUDNN_DATA_INT8:
      return "CUDNN_DATA_INT8";
    case CUDNN_DATA_INT32:
      return "CUDNN_DATA_INT32";
    case CUDNN_DATA_INT8x4:
      return "CUDNN_DATA_INT8x4";
#if CUDNN_VERSION >= 7100
    case CUDNN_DATA_UINT8:
      return "CUDNN_DATA_UINT8";
    case CUDNN_DATA_UINT8x4:
      return "CUDNN_DATA_UINT8x4";
#endif
    default:
      std::ostringstream oss;
      oss << "Unknown data type " << static_cast<int>(dtype);
      return oss.str();
  }
}

inline std::string cudnnMathTypeToString(cudnnMathType_t type) {
  switch (type) {
    case CUDNN_DEFAULT_MATH:
      return "CUDNN_DEFAULT_MATH";
    case CUDNN_TENSOR_OP_MATH:
      return "CUDNN_TENSOR_OP_MATH";
    case CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION:
      return "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION";
#if CUDNN_VERSION >= 8000
    case CUDNN_FMA_MATH:
      return "CUDNN_FMA_MATH";
#endif
    default:
      std::ostringstream oss;
      oss << "Unknown math type " << static_cast<int>(type);
      return oss.str();
  }
}

inline std::string conv2dFwdAlgoToString(const cudnnConvolutionFwdAlgo_t algo) {
  switch (algo) {
    case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
      return "CUDNN_CONVOLUTION_FWD_ALGO_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
      return "CUDNN_CONVOLUTION_FWD_ALGO_FFT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
      return "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING";
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
      return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
      return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
      return "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
      return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
      return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED";
    default:
      std::ostringstream oss;
      oss << "Unknown cudnn convolution forward algorithm " << static_cast<int>(algo);
      return oss.str();
  }
}

inline std::string conv2dBwdDataAlgoToString(const cudnnConvolutionBwdDataAlgo_t algo) {
  switch (algo) {
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
      return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
      return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
      return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
      return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
      return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
      return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED";
    default:
      std::ostringstream oss;
      oss << "Unknown cudnn convolution backward data algorithm " << static_cast<int>(algo);
      return oss.str();
  }
}

inline std::string conv2dBwdFilterAlgoToString(const cudnnConvolutionBwdFilterAlgo_t algo) {
  switch (algo) {
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
      return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
      return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
      return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
      return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
      return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
      return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING";
    default:
      std::ostringstream oss;
      oss << "Unknown cudnn convolution backward filter algorithm " << static_cast<int>(algo);
      return oss.str();
  }
}

class CUDNNThreadEntry {
 public:
  CUDNNThreadEntry();
  static CUDNNThreadEntry* ThreadLocal();

 public:
  /*! \brief cudnn handle. */
  cudnnHandle_t handle = nullptr;
  /*! \brief Whether to benchmark the performance when choosing CUDNN algorithms. */
  bool benchmark = true;
};

#if CUDNN_VERSION >= 7100

class CUDNNDType final : public EnumBase<CUDNNDType, 9, int32_t, cudnnDataType_t> {
 public:
  ENUM_DEF_HEADER(CUDNNDType, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 0, Float, CUDNN_DATA_FLOAT, "float32");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 1, Double, CUDNN_DATA_DOUBLE, "float64");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 2, Half, CUDNN_DATA_HALF, "float16");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 3, Char, CUDNN_DATA_INT8, "int8");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 4, Int, CUDNN_DATA_INT32, "int32");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 5, Charx4, CUDNN_DATA_INT8x4, "int8x4");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 6, UChar, CUDNN_DATA_UINT8, "uint8");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 7, UCharx4, CUDNN_DATA_INT8x4, "uint8x4");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 8, UCharx32, CUDNN_DATA_INT8x32, "uint8x32");

  explicit CUDNNDType(DType dt) : EnumBase(cudnnDataType_t(dt)) {
  }

 public:
  template <int value>
  inline const void* const_addr() {
    cudnnDataType_t dt(*this);
    switch (dt) {
      case CUDNN_DATA_FLOAT:
        return const_typed_addr<float, value>();
      case CUDNN_DATA_HALF:
        return const_typed_addr<float, value>();
      case CUDNN_DATA_DOUBLE:
        return const_typed_addr<double, value>();
      case CUDNN_DATA_INT8:
        return const_typed_addr<char, value>();
      case CUDNN_DATA_INT8x32:
        LOG(FATAL) << "NotImplementedError: " << dt << " no default alpha beta!";
      case CUDNN_DATA_UINT8:
        LOG(FATAL) << "NotImplementedError: " << dt << " no default alpha beta!";
      case CUDNN_DATA_UINT8x4:
        LOG(FATAL) << "NotImplementedError: " << dt << " no default alpha beta!";
      case CUDNN_DATA_INT32:
        LOG(FATAL) << "NotImplementedError: " << dt << " no default alpha beta!";
      default:
        LOG(FATAL) << "NotImplementedError: " << dt << " no default alpha beta!";
    }
    throw;
  }
};

#else

class CUDNNDType final : public EnumBase<CUDNNDType, 7, int32_t, cudnnDataType_t> {
 public:
  ENUM_DEF_HEADER(CUDNNDType, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 0, Float, CUDNN_DATA_FLOAT, "float32");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 1, Double, CUDNN_DATA_DOUBLE, "float64");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 2, Half, CUDNN_DATA_HALF, "float16");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 3, Char, CUDNN_DATA_INT8, "int8");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 4, Int, CUDNN_DATA_INT32, "int32");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 5, Charx4, CUDNN_DATA_INT8x4, "int8x4");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 6, UChar8x4, CUDNN_DATA_INT8x32, "uint8x32");

  explicit CUDNNDType(DType dt) : EnumBase(cudnnDataType_t(dt)) {
  }

 public:
  template <int value>
  inline const void* const_addr() {
    cudnnDataType_t dt(*this);
    switch (dt) {
      case CUDNN_DATA_FLOAT:
        return common::cuda::const_addr<float, value>();
      case CUDNN_DATA_HALF:
        return common::cuda::const_addr<float, value>();
      case CUDNN_DATA_DOUBLE:
        return common::cuda::const_addr<double, value>();
      case CUDNN_DATA_INT8:
        return common::cuda::const_addr<char, value>();
      case CUDNN_DATA_INT8x32:
        LOG(FATAL) << "NotImplementedError: " << dt << " no default alpha beta!";
      case CUDNN_DATA_INT32:
        LOG(FATAL) << "NotImplementedError: " << dt << " no default alpha beta!";
      default:
        LOG(FATAL) << "NotImplementedError: " << dt << " no default alpha beta!";
    }
    throw;
  }
};

#endif

inline int64_t GetTensorTypeDim(ir::TensorType tt, int i) {
  int n = tt->shape.size();
  CHECK(0 <= i && i < n) << "Get dim out of bound!";
  auto res = tvm::Downcast<ir::Integer>(tt->shape[i]);
  return res;
}

inline ir::TensorType SquashTensorShape(const DLTensor* tensor, const std::vector<int>& slices) {
  ir::Array<tvm::relay::IndexExpr> shape;
  if (slices.empty()) {
    for (int i = 0; i < tensor->ndim; ++i) {
      ir::Integer dim = tensor->shape[i];
      shape.push_back(dim);
    }
  } else {
    for (int i = 1, n = slices.size(); i < n; ++i) {
      ir::Integer prod = 1;
      if (0 <= slices[i - 1] && slices[i - 1] <= slices[i] && slices[i] <= tensor->ndim) {
        prod = std::accumulate(tensor->shape + slices[i - 1], tensor->shape + slices[i], 1ll,
                               std::multiplies<int64_t>());
      }
      shape.push_back(prod);
    }
  }
  auto res = ir::TensorType(shape, ir::DataType(tensor->dtype));
  return res;
}

inline cudnnTensorDescriptor_t NormalizeTensorType(ir::TensorType tt) {
  DLDataType dtype{(uint8_t)tt->dtype.code(), (uint8_t)tt->dtype.bits(),
                   (uint16_t)tt->dtype.lanes()};

  std::vector<int64_t> shape(tt->shape.size());
  for (int i = 0, n = shape.size(); i < n; ++i) {
    shape[i] = tvm::Downcast<ir::Integer>(tt->shape[i]);
  }

  std::vector<int> padded_shape = common::shape_utils::PadDims<int, int64_t>(shape, 4);
  std::vector<int> stride = common::shape_utils::Shape2Strides<int>(padded_shape);
  cudnnTensorDescriptor_t res;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&res));
  CUDNN_CALL(cudnnSetTensorNdDescriptor(res, CUDNNDType(dtype),
                                        static_cast<int>(padded_shape.size()),
                                        dmlc::BeginPtr(padded_shape), dmlc::BeginPtr(stride)));
  return res;
}

inline cudnnFilterDescriptor_t NormalizeFilter(const DLTensor* tv,
                                               cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW) {
  std::vector<int64_t> shape(tv->shape, tv->shape + tv->ndim);
  std::vector<int> padded_shape = common::shape_utils::PadDims<int, int64_t>(shape, 4);
  cudnnFilterDescriptor_t res;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&res));
  CUDNN_CALL(cudnnSetFilterNdDescriptor(res, CUDNNDType(tv->dtype), format,
                                        static_cast<int>(padded_shape.size()),
                                        dmlc::BeginPtr(padded_shape)));
  return res;
}

inline std::vector<int64_t> MakeAlgoKey(const std::vector<std::vector<int64_t>>& vs) {
  std::vector<int64_t> res;
  for (auto& v : vs) {
    res.push_back(v.size());
    res.insert(res.end(), v.begin(), v.end());
  }
  return res;
}

template <typename TDst, typename TSrc>
inline std::vector<TDst> CastVector(const std::vector<TSrc>& v) {
  std::vector<TDst> res(v.size());
  for (int i = 0, e = res.size(); i < e; ++i) {
    res[i] = v[i];
  }
  return res;
}

template <int numel>
inline std::vector<int64_t> NormalizeScalarToTuple(const std::vector<int64_t>& v) {
  int n = v.size();
  CHECK(n == 1 || n == numel) << "ValueError: we only accept a single integer or a tuple of "
                              << numel << " integers";
  return n == 1 ? std::vector<int64_t>(numel, v[0]) : v;
}

template <int numel>
inline std::vector<int64_t> NormalizeScalarToTuple(
    const ir::Optional<ir::Array<value::IntValue>> v) {
  CHECK(v.defined());
  auto value = v.value();
  int n = value.size();
  CHECK(n == 1 || n == numel) << "ValueError: we only accept a single integer or a tuple of "
                              << numel << " integers";
  if (n == 1) {
    return std::vector<int64_t>(numel, value[0]->value);
  } else {
    std::vector<int64_t> re;
    for (auto i : value) {
      re.push_back(i->value);
    }
    return re;
  }
}

template <typename T>
inline ir::Array<ir::Integer> ToArrayOfInteger(const std::vector<T>& v) {
  ir::Array<ir::Integer> res;
  for (auto elem : v) {
    res.push_back(static_cast<int64_t>(elem));
  }
  return res;
}

inline void SetStream(cudaStream_t stream) {
  CUDNN_CALL(cudnnSetStream(CUDNNThreadEntry::ThreadLocal()->handle, stream));
}

inline size_t ComputeStorageInBytes(const ir::TensorType& type) {
  size_t size = 1;
  for (auto dim : type->shape) {
    const auto* dim_imm = dim.as<ir::IntImmNode>();
    CHECK(dim_imm);
    size *= dim_imm->value;
  }
  size *= (type->dtype.bits() * type->dtype.lanes() + 7) / 8;
  return size;
}

}  // namespace cudnn
}  // namespace op
}  // namespace raf
