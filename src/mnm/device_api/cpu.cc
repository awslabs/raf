#include <dmlc/logging.h>
#include <mnm/device_api.h>
#include <mnm/registry.h>

namespace mnm {
namespace device_api {

class CPUDeviceAPI final : public mnm::device_api::DeviceAPI {
 public:
  CPUDeviceAPI() {
  }
  ~CPUDeviceAPI() override {
  }
  void SetDevice(mnm::types::Context ctx) override {
    CheckContext(ctx);
  }
  void* AllocDataSpace(mnm::types::Context ctx, size_t nbytes, size_t alignment,
                       mnm::types::DataType type_hint) override {
    CheckContext(ctx);
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
  void FreeDataSpace(mnm::types::Context ctx, void* ptr) override {
    CheckContext(ctx);
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

 private:
  static void CheckContext(mnm::types::Context ctx) {
    // TODO(@junrushao1994): too lazy to convert ctx.device_type to std::string;
    CHECK_EQ(ctx.device_type, kDLCPU)
        << "InternalError: CPU device API expect context is kCPU, but got" << ctx.device_type;
    CHECK_EQ(ctx.device_id, 0) << "InternalError: CPU device API expect device_id = 0, but got"
                               << ctx.device_id;
  }
};

MNM_REGISTER_GLOBAL("mnm.device_api.cpu")
    .set_body([](mnm::types::Args args, mnm::types::RetValue* rv) {
      // While it is relatively unsafe to directly "new" an object
      // we expect this object to be correctly managed by a shared_ptr in DeviceAPIManager
      DeviceAPI* ptr = new CPUDeviceAPI();
      *rv = static_cast<void*>(ptr);
    });

}  // namespace device_api
}  // namespace mnm
