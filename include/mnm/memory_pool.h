#pragma once

#include <memory>

#include <mnm/base.h>

namespace mnm {
namespace memory_pool {

class MemoryPool {
 public:
  virtual ~MemoryPool() = default;

  virtual void* Alloc(int64_t nbytes, int64_t alignment, DType type_hint) = 0;

  virtual void Dealloc(void* mem) = 0;

 public:
  static std::shared_ptr<MemoryPool> Get(Context ctx);
};

}  // namespace memory_pool
}  // namespace mnm
