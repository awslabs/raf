#include <mnm/device_api.h>
#include <mnm/registry.h>

namespace mnm {
namespace device_api {
namespace cpu_device_api {

class CPUDeviceAPI final : public DeviceAPI {
 public:
  CPUDeviceAPI() = default;
  ~CPUDeviceAPI() = default;

  void* AllocMemory(int device_id, int64_t nbytes, int64_t alignment, DType type_hint) override {
    CHECK_EQ(device_id, 0) << "InternalError: CPU expect device_id = 0, but got" << device_id;
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

  void DeallocMemory(int device_id, void* ptr) override {
    CHECK_EQ(device_id, 0) << "InternalError: CPU expect device_id = 0, but got" << device_id;
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

  static void* make() {
    return new CPUDeviceAPI();
  }
};  // namespace device_api

MNM_REGISTER_GLOBAL("mnm.device_api._make.cpu").set_body_typed(CPUDeviceAPI::make);

}  // namespace cpu_device_api
}  // namespace device_api
}  // namespace mnm
