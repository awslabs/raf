#pragma once

#include <array>
#include <memory>
#include <mutex>

#include <mnm/types.h>

namespace mnm {
namespace device_api {

/*
 * A device provides the following interfaces
 * 0) Initialize / Deinitialize the device
 * 1) Memory allocation and deallocation
 * 2) A stream where kernels launch
 * 3) Events that flags whether certain computation is done on a stream
 * 4) Data movement on the same or different devices
 *
 * For a certain device, developers are required to wrap their own low-level C API into the
 * interface of DeviceAPI, so that the resources could be better managed via upstream manager
 * like memory pool.
 *
 * Note for developers:
 * 1) All DeviceAPIs will be used as singletons inside the DeviceAPIManager singleton.
 * 2) For each type of device, there will be at most one singleton existing.
 * 3) We suggest not to keep any state in this class.
 *
 * TODO(@junrushao1994): many interfaces are not implemented
 * TODO(@junrushao1994): don't use size_t
 */

class DeviceAPI;
class DeviceAPIManager final {
  /*
   * The manager is a global singleton, which has ownership of all DeviceAPI singletons.
   * It shares the ownership with upstream managers like memory pool.
   */
  using APIPtr = std::unique_ptr<DeviceAPI>;

 public:
  static const int kMaxDeviceAPI = 32;

  DeviceAPIManager() = default;
  ~DeviceAPIManager() = default;

  DeviceAPI* GetAPI(mnm::types::DeviceType device_type, bool allow_missing);

 public:
  static std::shared_ptr<DeviceAPIManager> Global() {
    static std::shared_ptr<DeviceAPIManager> inst = std::make_shared<DeviceAPIManager>();
    return inst;
  }

 private:
  std::array<APIPtr, kMaxDeviceAPI> api_;
  std::mutex mutex_;
};

class DeviceAPI {
  friend DeviceAPIManager;

 public:
  virtual ~DeviceAPI() = default;
  virtual int GetNDevices() = 0;
  virtual void* AllocMemory(int device_id, size_t nbytes, size_t alignment,
                            mnm::types::DataType type_hint) = 0;
  virtual void DeallocMemory(int device_id, void* ptr) = 0;

 private:
  static DeviceAPI* Create(mnm::types::DeviceType device_type, bool allow_missing);
};

}  // namespace device_api
}  // namespace mnm
