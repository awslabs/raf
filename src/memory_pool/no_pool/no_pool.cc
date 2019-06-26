#include <mnm/device_api.h>
#include <mnm/memory_pool.h>
#include <mnm/registry.h>

namespace mnm {
namespace memory_pool {

using device_api::DeviceAPI;

class NoPool final : public MemoryPool {
 public:
  NoPool(Context ctx) {
  }

  virtual ~NoPool() = default;

  void* Alloc(int64_t nbytes, int64_t alignment, DType type_hint) override {
    return api_->AllocMemory(ctx_.device_id, nbytes, alignment, type_hint);
  }

  void Dealloc(void* mem) override {
    api_->DeallocMemory(ctx_.device_id, mem);
  }

  static void* make(DLContext ctx) {
    return new NoPool(ctx);
  }

 public:
  Context ctx_;
  std::shared_ptr<DeviceAPI> api_;
};

MNM_REGISTER_GLOBAL("mnm.memory_pool.no_pool").set_body_typed(NoPool::make);

}  // namespace memory_pool
}  // namespace mnm
