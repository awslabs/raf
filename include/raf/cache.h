/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file cache.h
 * \brief The RAF cache.
 */
#pragma once

#include <chrono>
#include <dmlc/memory_io.h>
#include <sys/stat.h>
#include "./file.h"
#include "./op.h"
#include "./value.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;

using PackedMetricMap = Map<String, Integer>;

#define RAF_APPEND_BYTES(type, nbytes, value)                               \
  {                                                                         \
    constexpr int NUM_BYTES = nbytes;                                       \
    union UNION {                                                           \
      type v;                                                               \
      struct {                                                              \
        uint8_t bytes[NUM_BYTES];                                           \
      };                                                                    \
    } u;                                                                    \
    u.v = value;                                                            \
    static_assert(sizeof(UNION) == sizeof(uint8_t) * NUM_BYTES, "invalid"); \
    static_assert(sizeof(UNION) == sizeof(type), "invalid");                \
    for (int i = 0; i < NUM_BYTES; ++i) {                                   \
      byte_vector.push_back(u.bytes[i]);                                    \
    }                                                                       \
  }

#define RAF_DEF_PRIMITIVE(type_code, type, nbytes)                              \
  inline HashKey& operator<<(const type& v) {                                   \
    static_assert(0 <= type_code, "invalid");                                   \
    static_assert(type_code <= std::numeric_limits<uint8_t>::max(), "invalid"); \
    RAF_APPEND_BYTES(type, nbytes, v);                                          \
    return *this;                                                               \
  }

class HashKey {
 public:
  RAF_DEF_PRIMITIVE(0, bool, 1);
  RAF_DEF_PRIMITIVE(1, int8_t, 1);
  RAF_DEF_PRIMITIVE(2, int16_t, 2);
  RAF_DEF_PRIMITIVE(3, int32_t, 4);
  RAF_DEF_PRIMITIVE(4, int64_t, 8);
  RAF_DEF_PRIMITIVE(5, uint8_t, 1);
  RAF_DEF_PRIMITIVE(6, uint16_t, 2);
  RAF_DEF_PRIMITIVE(7, uint32_t, 4);
  RAF_DEF_PRIMITIVE(8, uint64_t, 8);
  RAF_DEF_PRIMITIVE(9, float, 4);
  RAF_DEF_PRIMITIVE(10, double, 8);
  RAF_DEF_PRIMITIVE(11, DLDataType, 4);
  RAF_DEF_PRIMITIVE(12, DLDevice, 8);

  inline HashKey& operator<<(const char* v) {
    // In the case of `key << "string"`, gcc treats "string" as a char array and use "char *" to
    // overload the operator<<, which results in operator<<(bool) being called.
    // To avoid this, we explicitly handle char* and dispatch to operator<<(const std::string&).
    // Ref: https://stackoverflow.com/questions/14770252
    return operator<<(std::string(v));
  }

  inline HashKey& operator<<(const std::vector<int64_t>& v) {
    byte_vector.push_back(13);
    for (int i = 0, n = v.size(); i < n; ++i) {
      RAF_APPEND_BYTES(int64_t, 8, v[i]);
    }
    RAF_APPEND_BYTES(int64_t, 8, 0);
    return *this;
  }

  inline HashKey& operator<<(const tvm::runtime::Optional<ir::Array<value::IntValue>> v) {
    CHECK(v.defined());
    byte_vector.push_back(13);
    tvm::runtime::Array<IntValue> value = v.value();
    for (int i = 0, n = value.size(); i < n; ++i) {
      RAF_APPEND_BYTES(int64_t, 8, value[i]->value);
    }
    RAF_APPEND_BYTES(int64_t, 8, 0);
    return *this;
  }

  inline HashKey& operator<<(const ir::TensorType& v) {
    byte_vector.push_back(14);
    RAF_APPEND_BYTES(DLDataType, 4, v->dtype);
    for (int i = 0, n = v->shape.size(); i < n; ++i) {
      int64_t dim_i;
      if (v->shape.as<ir::AnyNode>()) {
        dim_i = -1;
      } else {
        dim_i = ir::Downcast<ir::Integer>(v->shape[i]).IntValue();
      }
      RAF_APPEND_BYTES(int64_t, 8, dim_i);
    }
    RAF_APPEND_BYTES(int64_t, 8, 0);
    return *this;
  }

  inline HashKey& operator<<(const DLTensor& v) {
    // N.B.: stride and ctx are not taken into consideration
    byte_vector.push_back(15);
    RAF_APPEND_BYTES(DLDataType, 4, v.dtype);
    for (int i = 0, n = v.ndim; i < n; ++i) {
      RAF_APPEND_BYTES(int64_t, 8, v.shape[i]);
    }
    RAF_APPEND_BYTES(int64_t, 8, 0);
    return *this;
  }

  inline HashKey& operator<<(const value::TensorValue& v) {
    DLTensor* t = v;
    return operator<<(*t);
  }

  inline HashKey& operator<<(const std::string& v) {
    byte_vector.push_back(16);
    for (int i = 0, n = v.size(); i < n; ++i) {
      RAF_APPEND_BYTES(int8_t, 1, v[i]);
    }
    RAF_APPEND_BYTES(int64_t, 8, 0);
    return *this;
  }

  inline HashKey& operator<<(const HashKey& other) {
    byte_vector.insert(byte_vector.end(), other.byte_vector.begin(), other.byte_vector.end());
    return *this;
  }

  HashKey() {
    byte_vector.reserve(1024);
  }

  std::vector<uint8_t> byte_vector;
};

#undef RAF_DEF_PRIMITIVE
#undef RAF_APPEND_BYTES

template <typename T>
class MetaCache {
 public:
  ~MetaCache() = default;

  bool Has(const std::vector<uint8_t>& key) {
    const std::string key_str(key.begin(), key.end());
    return Has(key_str);
  }

  bool Has(const std::string& key) {
    std::lock_guard<std::mutex> lock(mu_);
    return cached_.count(key);
  }

  const T* Get(const std::vector<uint8_t>& key) {
    const std::string key_str(key.begin(), key.end());
    return Get(key_str);
  }

  const T* Get(const std::string& key) {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = cached_.find(key);
    if (iter == cached_.end()) {
      return nullptr;
    }
    return &iter->second;
  }

  void Set(const std::vector<uint8_t>& key, T val) {
    const std::string key_str(key.begin(), key.end());
    Set(key_str, val);
  }

  void Set(const std::string& key, T val) {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = cached_.find(key);
    if (iter != cached_.end()) {
      LOG(FATAL) << "KeyError: The key is already cached!";
      throw;
    }
    cached_.emplace(key, val);
  }

 private:
  /*! \brief The cache mapping from string key to value. */
  std::unordered_map<std::string, T> cached_;
  /*! \brief The thread-safe lock. */
  std::mutex mu_;
};

class MetaCacheMetric {
 public:
  virtual std::unordered_map<std::string, size_t> GetMetric() = 0;
};

template <typename T>
class MetaPersistCache : public MetaCache<T>, public MetaCacheMetric {
 public:
  MetaPersistCache(const std::string persist_name) : persist_name_(persist_name) {
    // Enable persistent by users.
    const char* enable_persist = getenv("RAF_PERSIST_CACHE");
    if (enable_persist != nullptr && strcmp(enable_persist, "1") == 0) {
      DLOG(INFO) << "Persistent for cache " << persist_name_ << " is enabled";
      persist_ = true;
    } else {
      return;
    }

    // Determine and create the directory for the cache root.
    const char* temp = getenv("RAF_PERSIST_CACHE_PATH");
    std::string cache_path;
    if (temp == nullptr) {
      const char* home = getenv("HOME");
      CHECK(home != nullptr) << "HOME environment variable is not set";
      cache_path = std::string(home) + "/.raf_cache";
    } else {
      cache_path = std::string(temp);
    }
    CreateDir(cache_path);
    path_ = cache_path + "/" + persist_name_;

    // Create the directory for this cache.
    CreateDir(path_);
  }

  const T* Get(const std::vector<uint8_t>& key) {
    const std::string key_str(key.begin(), key.end());
    return Get(key_str);
  }

  const T* Get(const std::string& key) {
    AddMetric("CacheGet", 1);

    // Cache hit.
    if (auto val = MetaCache<T>::Get(key)) {
      AddMetric("CacheHit", 1);
      return val;
    }
    AddMetric("CacheMiss", 1);
    if (!persist_) {
      return nullptr;
    }

    // Cache miss, try to load from the persistent cache.
    std::lock_guard<std::mutex> lock(mu_);

    auto persist_path = GetPersistPath(key);

    // Persistent cache miss.
    if (!DirExists(persist_path)) {
      AddMetric("PersistCacheMiss", 1);
      return nullptr;
    }
    AddMetric("PersistCacheHit", 1);

    try {
      MetaCache<T>::Set(key, T::Load(persist_path));
      return MetaCache<T>::Get(key);
    } catch (dmlc::Error& e) {
      AddMetric("PersistCacheLoadFailure", 1);
      LOG(WARNING) << "Failed to load persist entry " << path_ << ": " << e.what();
      return nullptr;
    }
    return nullptr;
  }

  void Set(const std::vector<uint8_t>& key, T val) {
    const std::string key_str(key.begin(), key.end());
    Set(key_str, val);
  }

  void Set(const std::string& key, T val) {
    AddMetric("CacheSet", 1);
    MetaCache<T>::Set(key, val);
    if (!persist_) {
      return;
    }

    std::lock_guard<std::mutex> lock(mu_);

    auto persist_path = GetPersistPath(key);
    CreateDir(persist_path);

    // Persist the cache value.
    try {
      if (!val.Save(persist_path)) {
        throw;
      }
    } catch (dmlc::Error& e) {
      AddMetric("PersistCacheSaveFailure", 1);
      LOG(WARNING) << "Failed to persist cache entry to " << path_ << ": " << e.what();
      return;
    }

    std::ofstream metadata_file(persist_path + "/timestamp");
    metadata_file << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())
                  << std::endl;
    metadata_file.close();
  }

  std::unordered_map<std::string, size_t> GetMetric() override {
    return metrics_;
  }

 private:
  inline std::string GetPersistPath(const std::string& key) {
    const size_t hashed_key = std::hash<std::string>{}(key);
    return path_ + "/" + std::to_string(hashed_key);
  }

  inline void AddMetric(const std::string name, size_t val) {
    metrics_[name] += val;
  }

  /*! \brief The cache metrics for analysis. */
  std::unordered_map<std::string, size_t> metrics_;
  /*! \brief Persist directory name. */
  std::string persist_name_;
  /*! \brief Persist directory path. */
  std::string path_;
  /*! \brief Whether to presist values. */
  bool persist_ = false;
  /*! \brief The thread-safe lock. */
  std::mutex mu_;
};

PackedMetricMap DumpMetric(const std::string& cache_name);

}  // namespace op
}  // namespace raf
