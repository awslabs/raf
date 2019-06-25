#include <dmlc/logging.h>
#include <mnm/memory_pool.h>
#include <mnm/registry.h>

namespace mnm {
namespace memory_pool {

class NoPool final : public MemoryPool {
 public:
  NoPool() = default;
  // FIXME(@junrushao1994): this is not actually correct.
  virtual ~NoPool() = default;

  MemoryChunk* Alloc(size_t nbytes, size_t alignment, DType type_hint) override {
    CHECK(f_alloc_ != nullptr) << "f_alloc_ is not defined";
    MemoryChunk* mem = new MemoryChunk();
    mem->data = f_alloc_(nbytes, alignment, type_hint);
    ++n_chunks_;
    return mem;
  }

  void Dealloc(MemoryChunk* mem) override {
    CHECK(f_dealloc_ != nullptr) << "f_dealloc_ is not defined";
    f_dealloc_(mem->data);
    --n_chunks_;
    delete mem;
  }

  void DeallocAll() override {
    int n_chunks = n_chunks_;
    CHECK_EQ(n_chunks, 0)
        << "Cannot release the memory pool, because some memory chunks are not released";
  }

  static void* make() {
    return new NoPool();
  }

 private:
  std::atomic<int> n_chunks_{0};
};

MNM_REGISTER_GLOBAL("mnm.memory_pool.no_pool").set_body_typed(NoPool::make);

}  // namespace memory_pool
}  // namespace mnm
