#pragma once

#include <memory>

#include <mnm/base.h>

namespace mnm {
namespace memory_pool {

class MemoryPool;

// A chunk of memory, which may have shared reference to the pool so that they are freed correctly.
// Interaction between memory pool manager also happens here
class Memory {
 public:
  virtual ~Memory() = default;

 public:
  static std::shared_ptr<Memory> Alloc(const Context& ctx, int64_t nbytes,
                                       int64_t alignment = kDefaultMemoryAlignment);
  static std::vector<std::shared_ptr<Memory> > AllocMany(
      const Context& ctx, const std::vector<int64_t>& nbytes,
      int64_t alignment = kDefaultMemoryAlignment);

  // means "no longer considered as allocator when asking for new memory."
  static void RemovePool(const Context& ctx);

  static MemoryPool* GetPool(const Context& ctx);

  static MemoryPool* InitPool(const Context& ctx, const std::string& name);

 public:
  void* data = nullptr;
  Context ctx{};
};

// Only interface for implementing new allocation strategy, no static interface is included.
class MemoryPool {
 public:
  virtual ~MemoryPool() = default;

  virtual std::shared_ptr<Memory> Alloc(int64_t nbytes,
                                        int64_t alignment = kDefaultMemoryAlignment) = 0;

  virtual std::vector<std::shared_ptr<Memory> > AllocMany(
      const std::vector<int64_t>& nbytes, int64_t alignment = kDefaultMemoryAlignment) = 0;
};

}  // namespace memory_pool
}  // namespace mnm
