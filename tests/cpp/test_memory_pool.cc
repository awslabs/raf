#include <gtest/gtest.h>

#include <mnm/base.h>
#include <mnm/memory_pool.h>

using mnm::Context;
using mnm::DevType;
using mnm::kDefaultMemoryAlignment;
using mnm::memory_pool::Memory;
using mnm::memory_pool::MemoryPool;

TEST(NoPool, CPU) {
  Context ctx{DevType::kCPU(), 0};
  Memory::InitPool(ctx, "no_pool");
  {
    std::shared_ptr<Memory> result = Memory::Alloc(ctx, 0);
    ASSERT_EQ(result.use_count(), 1);
    ASSERT_EQ(result->data, nullptr);
  }
  for (int memory : {11, 19, 2019, 1024124}) {
    for (int align : {16, (int)kDefaultMemoryAlignment, 512, 1024, 4096}) {
      std::shared_ptr<Memory> result = Memory::Alloc(ctx, memory, align);
      ASSERT_EQ(result.use_count(), 1);
      int64_t address = (int64_t)result->data;
      ASSERT_EQ(address % align, 0);
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
