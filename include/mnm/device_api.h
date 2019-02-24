#pragma once

#include <mnm/types.h>
#include <mnm/registry.h>
#include <mutex>
#include <tvm/runtime/device_api.h>

namespace mnm {
namespace device_api {

class DeviceAPIManager;

class DeviceAPI {
  /*
   * A device provides the following interfaces
   * 0) Initialize / Deinitialize the device
   * 1) Memory allocation and deallocation
   * 2) A stream where kernels launch
   * 3) Events that flags whether certain computation is done on a stream
   * 4) Data movement on the same or different devices
   *
   * For a certain device, developers are required to wrap their own low-level C API into the
   * interface of DeviceAPI, so that the resources could be better managed via upstream manager like
   * memory pool.
   *
   * Note for developers:
   * 1) All DeviceAPIs will be used as singletons inside the DeviceAPIManager singleton.
   * 2) For each type of device, there will be at most one singleton existing.
   * 3) We suggest not to keep any state in this class.
   *
   * TODO(@junrushao1994): many interfaces are not implemented
   */
 public:
  virtual ~DeviceAPI() {}
  virtual void SetDevice(mnm::types::Context ctx) = 0;
  virtual void *AllocDataSpace(mnm::types::Context ctx, size_t nbytes, size_t alignment,
                               mnm::types::DataType type_hint) = 0;
  virtual void FreeDataSpace(mnm::types::Context ctx, void *ptr) = 0;
};

class DeviceAPIManager {
  /*
   * The manager is a global singleton, which has ownership of all DeviceAPI singletons.
   * It shares the ownership with upstream managers like memory pool.
   */
 public:
  static const int kMaxDeviceAPI = 32;

  DeviceAPIManager() {
    std::fill(api_.begin(), api_.end(), nullptr);
  }

  static DeviceAPIManager* Global() {
    static DeviceAPIManager inst;
    return &inst;
  }

  std::shared_ptr<DeviceAPI> GetAPI(mnm::types::DeviceType device_type, bool allow_missing = false) {
    if (api_[device_type] == nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (api_[device_type] == nullptr) {
        api_[device_type].reset(Create(device_type, allow_missing));
      }
    }
    return api_[device_type];
  }

 private:
  std::array<std::shared_ptr<DeviceAPI>, kMaxDeviceAPI> api_;
  std::mutex mutex_;

  inline DeviceAPI* Create(mnm::types::DeviceType device_type, bool allow_missing = false) {
    std::string creator_name = std::string("mnm.device_api.") + mnm::types::DeviceName(device_type);
    auto creator = mnm::registry::Registry::Get(creator_name);
    if (creator == nullptr) {
      CHECK(allow_missing) << "ValueError: DeviceAPI " << creator_name << " is not enabled.";
      return nullptr;
    }
    void *ret = (*creator)();
    return static_cast<DeviceAPI*>(ret);
  }
};

}  // namespace device_api
}  // namespace mnm
