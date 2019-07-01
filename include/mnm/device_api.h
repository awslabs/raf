#pragma once

#include <memory>

#include <mnm/base.h>

namespace mnm {
namespace device_api {

// TODO(@junrushao1994): To pass flags to stream/event/..., do we add thread_local flags?
class DeviceAPI {
 public:
  virtual ~DeviceAPI() = default;

  virtual void SetDevice(int device_id) = 0;

  virtual int GetDevice() = 0;

  // Memory
  virtual void* AllocMemory(int64_t nbytes, int64_t alignment) = 0;

  virtual void FreeMemory(void* ptr) = 0;

  // Stream
  virtual void* CreateStream() = 0;

  virtual void FreeStream(void* stream) = 0;

  // will call the device api of `next_ctx` to wait for `prev`
  // therefore, we should the assumption that `after.ctx == device_api.ctx`
  virtual void SyncStream(const Context& prev_ctx, void* prev, void* next) = 0;

  // Granularity of synchronization
  virtual void WaitDevice() = 0;

  virtual void WaitStream(void* stream) = 0;

 public:
  static std::shared_ptr<DeviceAPI> Get(DevType device_type);
};

}  // namespace device_api
}  // namespace mnm
