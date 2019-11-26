/*!
 * Copyright (c) 2019 by Contributors
 * \file device_api.h
 * \brief Unified low-level API for heterogeneous devices
 */
#pragma once
#include <memory>
#include "./base.h"

namespace mnm {
namespace device_api {

// TODO(@junrushao1994): To pass flags to stream/event/..., do we add thread_local flags?
class DeviceAPI {
 public:
  virtual ~DeviceAPI() = default;

  // Memory
  virtual void* AllocMemory(int64_t nbytes, int64_t alignment = kDefaultMemoryAlignment) = 0;

  virtual void FreeMemory(void* ptr) = 0;

  // Stream
  virtual void* CreateStream(const Context& ctx) = 0;

  virtual void FreeStream(const Context& ctx, void* stream) = 0;

  // will call the device api of `next_ctx` to wait for `prev`
  // therefore, we should the assumption that `after.ctx == device_api.ctx`
  virtual void SyncStream(const Context& prev_ctx, void* prev, void* next) = 0;

  // Granularity of synchronization
  virtual void WaitDevice(const Context& ctx) = 0;

  virtual void WaitStream(const Context& ctx, void* stream) = 0;

 public:
  static std::shared_ptr<DeviceAPI> Get(DevType device_type);
};

}  // namespace device_api
}  // namespace mnm
