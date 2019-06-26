#pragma once

#include <array>
#include <functional>
#include <memory>
#include <mutex>

#include <mnm/base.h>
#include <tvm/runtime/registry.h>

#define MNM_REGISTER_GLOBAL TVM_REGISTER_GLOBAL

namespace mnm {
namespace registry {
using Registry = tvm::runtime::Registry;
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
    EnsureCapacity(dev_type.operator int());
    EntryPtr& ret = entries_[int(dev_type)];
    if (create_default) {
      CreateMissing(ret);
    }
    return ret;
  }

  std::unique_lock<std::mutex>&& GrabLock() {
    std::unique_lock<std::mutex> lock(mutex_);
    return std::move(lock);
  }

 protected:
  void CreateMissing(EntryPtr& ret) {
    if (ret == nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (ret == nullptr) {
        ret = std::make_shared<EntryType>();
      }
    }
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
    EnsureCapacity(int(ctx.device_type), ctx.device_id);
    EntryPtr& ret = entries_[int(ctx.device_type)][ctx.device_id];
    if (create_default) {
      CreateMissing(ret);
    }
    return ret;
  }

  std::unique_lock<std::mutex>&& GrabLock() {
    std::unique_lock<std::mutex> lock(mutex_);
    return std::move(lock);
  }

 protected:
  void CreateMissing(EntryPtr& ret) {
    if (ret == nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (ret == nullptr) {
        ret = std::make_shared<EntryType>();
      }
    }
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
  std::vector<std::vector<EntryPtr>> entries_;
  std::mutex mutex_;
};

}  // namespace registry
}  // namespace mnm
