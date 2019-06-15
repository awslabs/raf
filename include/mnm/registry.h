#pragma once

#include <array>
#include <functional>
#include <memory>
#include <mutex>

#include <tvm/runtime/registry.h>

#define MNM_REGISTER_GLOBAL TVM_REGISTER_GLOBAL

namespace mnm {
namespace registry {
using Registry = tvm::runtime::Registry;
}  // namespace registry
}  // namespace mnm

namespace mnm {
namespace registry {

constexpr int kMaxDeviceTypes = 32;
constexpr int kMaxDevicesPerType = 32;

template <class EntryType>
class PerContextStorage {
 public:
  ~PerContextStorage() DMLC_THROW_EXCEPTION {
    for (auto& outer : entries_) {
      for (std::unique_ptr<EntryType>& entry : outer) {
        entry.reset(nullptr);
      }
    }
  }

  void Read(int device_type,                               //
            int device_id,                                 //
            std::function<void(EntryType*)> f,             //
            std::function<EntryType*()> init_f = nullptr)  //
  {
    std::unique_ptr<EntryType>& ptr = entries_[device_type][device_id];
    if (ptr == nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (ptr == nullptr) {
        if (init_f != nullptr) {
          ptr.reset(init_f());
        } else {
          ptr.reset(new EntryType());
        }
      }
    }
    if (f) {
      f(ptr.get());
    }
  }

  void Write(int device_type,                               //
             int device_id,                                 //
             std::function<void(EntryType*)> f,             //
             std::function<EntryType*()> init_f = nullptr)  //
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unique_ptr<EntryType>& ptr = entries_[device_type][device_id];
    if (ptr == nullptr) {
      if (init_f != nullptr) {
        ptr.reset(init_f());
      } else {
        ptr.reset(new EntryType());
      }
    }
    if (f) {
      f(ptr.get());
    }
  }

 private:
  std::array<std::array<std::unique_ptr<EntryType>, kMaxDevicesPerType>, kMaxDeviceTypes> entries_;
  std::mutex mutex_;
};

template <class EntryType>
class PerDeviceTypeStorage {
 public:
  ~PerDeviceTypeStorage() DMLC_THROW_EXCEPTION {
    for (std::unique_ptr<EntryType>& entry : entries_) {
      entry.reset(nullptr);
    }
  }

  void Read(int device_type,                               //
            std::function<void(EntryType*)> f,             //
            std::function<EntryType*()> init_f = nullptr)  //
  {
    std::unique_ptr<EntryType>& ptr = entries_[device_type];
    if (ptr == nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (ptr == nullptr) {
        if (init_f != nullptr) {
          ptr.reset(init_f());
        } else {
          ptr.reset(new EntryType());
        }
      }
    }
    if (f) {
      f(ptr.get());
    }
  }

  void Write(int device_type,                               //
             std::function<void(EntryType*)> f,             //
             std::function<EntryType*()> init_f = nullptr)  //
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unique_ptr<EntryType>& ptr = entries_[device_type];
    if (ptr == nullptr) {
      if (init_f != nullptr) {
        ptr.reset(init_f());
      } else {
        ptr.reset(new EntryType());
      }
    }
    if (f) {
      f(ptr.get());
    }
  }

 private:
  std::array<std::unique_ptr<EntryType>, kMaxDeviceTypes> entries_;
  std::mutex mutex_;
};

}  // namespace registry
}  // namespace mnm
