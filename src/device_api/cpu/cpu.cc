/*!
 * Copyright (c) 2019 by Contributors
 * \file src/device_api/cpu/cpu.cc
 * \brief CPU device API
 */
#include <mnm/device_api.h>
#include <mnm/registry.h>

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

  void FreeMemory(void* ptr) override {
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

  void* CreateStream(const Context&) override {
    throw;
  }

  void FreeStream(const Context&, void* stream) override {
    throw;
  }

  void SyncStream(const Context& prev_ctx, void* prev, void* next) override {
    throw;
  }

  void WaitDevice(const Context&) override {
    throw;
  }

  void WaitStream(const Context&, void* stream) override {
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
