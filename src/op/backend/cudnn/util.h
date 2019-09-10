#pragma once

#include <cudnn.h>

#include <mnm/base.h>
#include <mnm/enum_base.h>
#include <mnm/ir.h>
#include <mnm/value.h>

#include "../../../common/cuda.h"
#include "../../../common/shape_utils.h"

#define CUDNN_CALL(func)                                                      \
  do {                                                                        \
    cudnnStatus_t e = (func);                                                 \
    CHECK_EQ(e, CUDNN_STATUS_SUCCESS) << "cuDNN: " << cudnnGetErrorString(e); \
  } while (false)

#define FORM_SHAPE(def, dl_tensor)                                   \
  std::vector<int> def = common::shape_utils::PadDims<int, int64_t>( \
      std::vector<int64_t>((dl_tensor)->shape, (dl_tensor)->shape + (dl_tensor)->ndim), 4)

#define FORM_STRIDE(def, shape) \
  std::vector<int> def = common::shape_utils::Shape2Strides<int>(shape)

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
namespace backend {
namespace cudnn {

class CUDNNThreadEntry {
 public:
  CUDNNThreadEntry();
  static CUDNNThreadEntry* ThreadLocal();

 public:
  cudnnHandle_t handle{nullptr};
};

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

  CUDNNDType(DType dt) : EnumBase(cudnnDataType_t(dt)) {
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

  CUDNNDType(DType dt) : EnumBase(cudnnDataType_t(dt)) {
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

inline void VecAppend(std::vector<int64_t>& res, int64_t v) {
  res.push_back(1);
  res.push_back(v);
}

inline void VecAppend(std::vector<int64_t>& res, const std::vector<int>& v) {
  res.push_back(v.size());
  for (auto elem : v) {
    res.push_back(elem);
  }
}

inline void VecAppend(std::vector<int64_t>& res, ir::Array<ir::Integer> a) {
  return VecAppend(res, common::shape_utils::MakeShape<int>(a));
}

class BufferNode : public value::ValueNode {
 public:
  mutable void* data{nullptr};
  mutable int64_t size_in_bytes{0};
  static constexpr const char* _type_key = "mnm.value.BufferNode";
  MNM_DEF_NODE_TYPE_INFO(BufferNode, ValueNode);
};

class BufferValue : public value::Value {
 public:
  MNM_DEF_NODE_REF_METHODS(BufferValue, Value, BufferNode);
};

}  // namespace cudnn
}  // namespace backend
}  // namespace op
}  // namespace mnm
