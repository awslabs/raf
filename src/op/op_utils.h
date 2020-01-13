/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/op_utils.h
 * \brief Useful classes of storing op metadata.
 */
#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include "mnm/op.h"

namespace mnm {
namespace op {
namespace utils {

class HashKey {
  template<typename T>
  HashKey& PODShrImpl(const T& v, uint8_t typecode) {
    byte_vector.push_back(typecode);
    byte_vector.resize(byte_vector.size() + sizeof(v));
    *reinterpret_cast<T*>(dmlc::BeginPtr(byte_vector) + byte_vector.size() - sizeof(v)) = v;
    return *this;
  }

 public:
  inline HashKey& operator<<(const bool &v) {
    return PODShrImpl(v, 0);
  }
  inline HashKey& operator<<(const int8_t &v) {
    return PODShrImpl(v, 1);
  }
  inline HashKey& operator<<(const int16_t &v) {
    return PODShrImpl(v, 2);
  }
  inline HashKey& operator<<(const int32_t &v) {
    return PODShrImpl(v, 3);
  }
  inline HashKey& operator<<(const int64_t &v) {
    return PODShrImpl(v, 4);
  }
  inline HashKey& operator<<(const uint8_t &v) {
    return PODShrImpl(v, 5);
  }
  inline HashKey& operator<<(const uint16_t &v) {
    return PODShrImpl(v, 6);
  }
  inline HashKey& operator<<(const uint32_t &v) {
    return PODShrImpl(v, 7);
  }
  inline HashKey& operator<<(const uint64_t &v) {
    return PODShrImpl(v, 8);
  }
  inline HashKey& operator<<(const float &v) {
    return PODShrImpl(v, 9);
  }
  inline HashKey& operator<<(const double &v) {
    return PODShrImpl(v, 10);
  }
  inline HashKey& operator<<(const std::vector<int64_t> &v) {
    byte_vector.push_back(11);
    (*this) << static_cast<int64_t>(v.size());
    for (int i = 0, n = v.size(); i < n; ++i) {
      (*this) << v[i];
    }
    return *this;
  }
  inline HashKey& operator<<(const tvm::DataType &v) {
    byte_vector.push_back(12);
    (*this) << v.code() << v.bits() << v.lanes();
    return *this;
  }
  inline HashKey& operator<<(const ir::TensorType &v) {
    byte_vector.push_back(13);
    (*this) << v->dtype;
    (*this) << v->shape.size();
    for (int i = 0, n = v->shape.size(); i < n; ++i) {
      if (v->shape.as<tvm::ir::Any>()) {
        (*this) << static_cast<uint64_t>(~0ull);
      } else {
        int64_t dim = tvm::Downcast<ir::Integer>(v->shape[i]);
        (*this) << dim;
      }
    }
    return *this;
  }
  inline HashKey& operator<<(const Context &ctx) {
    byte_vector.push_back(14);
    (*this) << ctx.device_type;
    (*this) << ctx.device_id;
    return *this;
  }
  std::vector<uint8_t> byte_vector;
};

template <typename T>
class MetaCache {
  std::unordered_map<std::string, T> cached_results;
  std::mutex mutex_;

 public:
  // TODO(@were): serialize and dump the cached results when exiting.
  ~MetaCache() {
  }


  bool has(const std::vector<uint8_t>& key) {
    const std::string s(key.begin(), key.end());
    {
      std::lock_guard<std::mutex> lock(mutex_);
      return cached_results.count(s);
    }
  }

  const T *get(const std::vector<uint8_t>& key) {
    const std::string s(key.begin(), key.end());
    // TODO(@were): We need to write a optional utility to systematically do this.
    static thread_local T res;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto iter = cached_results.find(s);
      if (iter == cached_results.end()) {
        return nullptr;
      }
      res = iter->second;
      return &res;
    }
  }

  void set(const std::vector<uint8_t>& key, T val) {
    const std::string s(key.begin(), key.end());
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto iter = cached_results.find(s);
      if (iter != cached_results.end()) {
        LOG(FATAL) << "KeyError: The key is already cached!";
        throw;
      }
      cached_results.emplace(s, val);
    }
  }
};

}  // namespace utils
}  // namespace op
}  // namespace mnm
