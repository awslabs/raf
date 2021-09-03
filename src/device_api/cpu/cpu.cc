/*!
 * Copyright (c) 2019 by Contributors
 * \file src/device_api/cpu/cpu.cc
 * \brief CPU device API
 */
#include "mnm/device_api.h"
#include "mnm/registry.h"

namespace mnm {
namespace device_api {
namespace cpu {

class CPUDeviceAPI final : public DeviceAPI {
 public:
  CPUDeviceAPI() = default;
  ~CPUDeviceAPI() = default;

  void* AllocMemory(int64_t nbytes, int64_t alignment) override {
    void* ptr = nullptr;
    // TODO(@junrushao1994): do not throw like this
    // TODO(@junrushao1994): recover the SGX and Android part
#if _MSC_VER
    ptr = _aligned_malloc(nbytes, alignment);
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
#else
    int ret = posix_memalign(&ptr, alignment, nbytes);
    if (ret != 0) {
      throw std::bad_alloc();
    }
#endif
    return ptr;
  }

  void* AllocMemoryAsync(int64_t nbytes, void* stream,
                         int64_t alignment = kDefaultMemoryAlignment) {
    throw;
  }

  void FreeMemory(void* ptr) override {
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

  void FreeMemoryAsync(void* ptr, void* stream) {
    throw;
  }

  void* CreateStream(const Device&) override {
    throw;
  }

  void FreeStream(const Device&, void* stream) override {
    throw;
  }

  void* CreateEvent(const Device& dev, uint32_t flags) override {
    throw;
  }

  void FreeEvent(const Device& dev, void* event) {
    throw;
  }

  void EventRecordOnStream(const Device& dev, void* event, void* stream) {
    throw;
  }

  void StreamWaitEvent(const Device& dev, void* stream, void* event) {
    throw;
  }

  void SyncStream(const Device& prev_dev, void* prev, void* next) override {
    throw;
  }

  void WaitDevice(const Device&) override {
    // Do nothing
  }

  void WaitStream(const Device&, void* stream) override {
    throw;
  }

  void SetDevice(const int device_id) override {
    throw;
  }

  static void* make() {
    return new CPUDeviceAPI();
  }
};

MNM_REGISTER_GLOBAL("mnm.device_api._make.cpu").set_body_typed(CPUDeviceAPI::make);

}  // namespace cpu
}  // namespace device_api
}  // namespace mnm
