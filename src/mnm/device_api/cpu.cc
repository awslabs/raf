#include <dmlc/logging.h>
#include <mnm/device_api.h>
#include <mnm/registry.h>
#include <mnm/types.h>

namespace mnm {
namespace device_api {

using mnm::types::DataType;

class CPUDeviceAPI final : public DeviceAPI {
 public:
  CPUDeviceAPI() = default;
  ~CPUDeviceAPI() override = default;
  int GetNDevices() override {
    return 1;
  }
  void* AllocMemory(int device_id, size_t nbytes, size_t alignment, DataType type_hint) override {
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
};

using mnm::types::Args;
using mnm::types::RetValue;

MNM_REGISTER_GLOBAL("mnm.device_api.cpu").set_body([](Args args, RetValue* rv) {
  DeviceAPI* ptr = new CPUDeviceAPI();
  *rv = static_cast<void*>(ptr);
});

}  // namespace device_api
}  // namespace mnm
