#include <mnm/device_api.h>
#include <mnm/memory_pool.h>
#include <mnm/registry.h>

namespace mnm {
namespace memory_pool {
namespace no_pool {

using device_api::DeviceAPI;

class NoPool final : public MemoryPool {
 public:
  NoPool(Context ctx) {
    this->ctx_ = ctx;
    this->api_ = DeviceAPI::Get(ctx.device_type);
  }

  virtual ~NoPool() = default;

  void* Alloc(int64_t nbytes, int64_t alignment) override {
    api_->SetDevice(ctx_.device_id);
    return api_->AllocMemory(nbytes, alignment);
  }

  void Dealloc(void* mem) override {
    api_->SetDevice(ctx_.device_id);
    api_->FreeMemory(mem);
  }

  static void* make(DLContext ctx) {
    return new NoPool(ctx);
  }

 public:
  Context ctx_;
  std::shared_ptr<DeviceAPI> api_;
};

MNM_REGISTER_GLOBAL("mnm.memory_pool.no_pool").set_body_typed(NoPool::make);

}  // namespace no_pool
}  // namespace memory_pool
}  // namespace mnm
