/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/dispatch/cudnn/cudnn_utils.h
 * \brief Helper functions for cuDNN
 */
#pragma once
#include <cudnn.h>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include <numeric>
#include "mnm/base.h"
#include "mnm/op.h"
#include "mnm/enum_base.h"
#include "mnm/ir.h"
#include "mnm/value.h"
#include "../../../common/cuda_utils.h"
#include "../../../common/shape_utils.h"

#define CUDNN_CALL(func)                                                      \
  do {                                                                        \
    cudnnStatus_t e = (func);                                                 \
    CHECK_EQ(e, CUDNN_STATUS_SUCCESS) << "cuDNN: " << cudnnGetErrorString(e); \
  } while (false)

namespace mnm {

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

class CUDNNThreadEntry {
 public:
  CUDNNThreadEntry();
  static CUDNNThreadEntry* ThreadLocal();

 public:
  cudnnHandle_t handle{nullptr};
};

template <typename T, int value>
inline const void* const_typed_addr() {
  static const T a = static_cast<T>(value);
  return static_cast<const void*>(&a);
}

#if CUDNN_VERSION >= 7100

class CUDNNDType final : public EnumBase<CUDNNDType, 9, int32_t, cudnnDataType_t> {
 public:
  ENUM_DEF_HEADER(CUDNNDType, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 0, Float, CUDNN_DATA_FLOAT, "float32");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 1, Double, CUDNN_DATA_DOUBLE, "float64");
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 2, Half, CUDNN_DATA_FLOAT, "float16");
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
  ENUM_DEF_ENTRY_WITH_NAME(CUDNNDType, 2, Half, CUDNN_DATA_FLOAT, "float16");
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

template <typename T>
class AlgorithmCache {
  std::map<std::vector<int64_t>, T> cached_results;

  static std::string Key2String(const std::vector<int64_t>& key) {
    std::ostringstream oss;
    for (size_t i = 0; i < key.size(); ++i) {
      oss << key[i];
      if (i != key.size() - 1) {
        oss << " ";
      }
    }
    return oss.str();
  }

 public:
  // TODO(@were): serialize and dump the cached results when exiting.
  ~AlgorithmCache() {
  }

  bool has(const std::vector<int64_t>& key) {
    return cached_results.count(key);
  }

  T get(const std::vector<int64_t>& key) {
    if (!has(key)) {
      LOG(FATAL) << "KeyError: The cached results have no key: " << AlgorithmCache::Key2String(key)
                 << "\n";
      throw;
    }
    return cached_results[key];
  }

  void set(const std::vector<int64_t>& key, T val) {
    if (has(key)) {
      LOG(FATAL) << "KeyError: The result is already cached: " << AlgorithmCache::Key2String(key)
                 << "\n";
      throw;
    }
    cached_results[key] = val;
  }
};

inline cudnnTensorDescriptor_t FlattenAndNormalizeTensor(const DLTensor* tv,
                                                         int flatten_from_axis) {
  CHECK(0 <= flatten_from_axis && flatten_from_axis <= tv->ndim);
  std::vector<int64_t> shape(tv->shape, tv->shape + flatten_from_axis);
  if (flatten_from_axis != tv->ndim) {
    shape.push_back(
        std::accumulate(tv->shape + flatten_from_axis,  // from tv->shape[flatten_from_axis]
                        tv->shape + tv->ndim,           // to   tv->shape[tv->ndim - 1]
                        1LL,                            // do   product
                        std::multiplies<int64_t>()));
  }

  std::vector<int> padded_shape = common::shape_utils::PadDims<int, int64_t>(shape, 4);
  std::vector<int> stride = common::shape_utils::Shape2Strides<int>(padded_shape);
  cudnnTensorDescriptor_t res;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&res));
  CUDNN_CALL(cudnnSetTensorNdDescriptor(res, CUDNNDType(tv->dtype),
                                        static_cast<int>(padded_shape.size()),
                                        dmlc::BeginPtr(padded_shape), dmlc::BeginPtr(stride)));
  return res;
}

inline cudnnFilterDescriptor_t NormalizeFilter(const DLTensor* tv,
                                               cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW) {
  int ndim = tv->ndim;
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

}  // namespace cudnn
}  // namespace op
}  // namespace mnm
