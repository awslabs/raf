/*!
 * Copyright (c) 2019 by Contributors
 * \file registry.h
 * \brief Utilities for registering items in separate translation units
 */
#pragma once

#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "tvm/runtime/registry.h"
#include "./base.h"

#define MNM_REGISTER_GLOBAL(name) TVM_REGISTER_GLOBAL(name)

namespace mnm {
namespace registry {

using tvm::runtime::PackedFunc;
using tvm::runtime::Registry;
using tvm::runtime::TVMArgs;
using tvm::runtime::TVMArgValue;
using tvm::runtime::TVMRetValue;
using tvm::runtime::TypedPackedFunc;

const PackedFunc& GetPackedFunc(const std::string& name);

}  // namespace registry
}  // namespace mnm

namespace mnm {
namespace registry {

template <class EntryType, bool create_default = true>
class PerDevTypeStore {
 public:
  using EntryPtr = std::shared_ptr<EntryType>;

  PerDevTypeStore() = default;

  ~PerDevTypeStore() DMLC_THROW_EXCEPTION {
    for (EntryPtr& entry : entries_) {
      entry = nullptr;
    }
  }

  EntryPtr& Get(DevType dev_type) {
    int dev_type_int = dev_type.operator int();
    EnsureCapacity(dev_type_int);
    EntryPtr& ret = entries_[dev_type_int];
    if (create_default) {
      CreateMissing(&ret);
    }
    return ret;
  }

 protected:
  template <bool b = create_default>
  void CreateMissing(EntryPtr *p, typename std::enable_if_t<b, int> = 0) {
    if (*p == nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (*p == nullptr) {
        *p = std::make_shared<EntryType>();
      }
    }
  }

  template <bool b = create_default>
  void CreateMissing(EntryPtr *p, typename std::enable_if_t<!b, int> = 0) {
  }

  void EnsureCapacity(int i) {
    if (i >= static_cast<int>(entries_.size())) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (i >= static_cast<int>(entries_.size())) {
        entries_.resize(i + 1);
      }
    }
  }

 public:
  std::vector<EntryPtr> entries_;
  std::mutex mutex_;
};

}  // namespace registry
}  // namespace mnm

namespace mnm {
namespace registry {

template <class EntryType, bool create_default = true>
class PerContextStore {
 public:
  using EntryPtr = std::shared_ptr<EntryType>;

  PerContextStore() = default;

  ~PerContextStore() DMLC_THROW_EXCEPTION {
    for (auto& outer : entries_) {
      for (EntryPtr& entry : outer) {
        entry = nullptr;
      }
    }
  }

  EntryPtr& Get(Context ctx) {
    int dev_type_int = ctx.device_type;
    EnsureCapacity(dev_type_int, ctx.device_id);
    EntryPtr& ret = entries_[dev_type_int][ctx.device_id];
    if (create_default) {
      CreateMissing(&ret);
    }
    return ret;
  }

 protected:
  template <bool b = create_default>
  void CreateMissing(EntryPtr* p, typename std::enable_if_t<b, int> = 0) {
    if (p == nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (p == nullptr) {
        p = std::make_shared<EntryType>();
      }
    }
  }

  template <bool b = create_default>
  void CreateMissing(EntryPtr* p, typename std::enable_if_t<!b, int> = 0) {
  }

  void EnsureCapacity(int i, int j) {
    if (i >= static_cast<int>(entries_.size()) || j >= static_cast<int>(entries_[i].size())) {
      if (i >= static_cast<int>(entries_.size())) {
        entries_.resize(i + 1);
      }
      if (j >= static_cast<int>(entries_[i].size())) {
        entries_[i].resize(j + 1);
      }
    }
  }

 public:
  std::vector<std::vector<EntryPtr> > entries_;
  std::mutex mutex_;
};

}  // namespace registry
}  // namespace mnm
