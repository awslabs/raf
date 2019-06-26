#pragma once

#include <memory>

#include <mnm/base.h>

namespace mnm {
namespace device_api {

class DeviceAPI {
 public:
  virtual ~DeviceAPI() = default;

  virtual void* AllocMemory(int device_id, int64_t nbytes, int64_t alignment, DType type_hint) = 0;

  virtual void DeallocMemory(int device_id, void* ptr) = 0;

 public:
  static std::shared_ptr<DeviceAPI> Get(DevType device_type);
};

}  // namespace device_api
}  // namespace mnm
