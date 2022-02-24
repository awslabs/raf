/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <raf/device.h>
#include <raf/memory_pool.h>

using raf::Device;
using raf::DevType;
using raf::kDefaultMemoryAlignment;
using raf::memory_pool::Memory;
using raf::memory_pool::MemoryPool;

TEST(NoPool, CPU) {
  Device dev{DevType::kCPU(), 0};
  Memory::InitPool(dev, "no_pool");
  {
    std::shared_ptr<Memory> result = Memory::Alloc(dev, 0);
    ASSERT_EQ(result.use_count(), 1);
    ASSERT_EQ(result->data, nullptr);
  }
  for (int memory : {11, 19, 2019, 1024124}) {
    for (int align : {16, (int)kDefaultMemoryAlignment, 512, 1024, 4096}) {
      std::shared_ptr<Memory> result = Memory::Alloc(dev, memory, align);
      ASSERT_EQ(result.use_count(), 1);
      int64_t address = (int64_t)result->data;
      ASSERT_EQ(address % align, 0);
    }
  }
  Memory::RemovePool(dev);
}

TEST(PageUnitPool, CPU) {
  Device dev{DevType::kCPU(), 0};
  Memory::InitPool(dev, "page_unit_pool");
  {
    std::shared_ptr<Memory> result = Memory::Alloc(dev, 0);
    ASSERT_EQ(result.use_count(), 1);
    ASSERT_EQ(result->data, nullptr);
  }
  for (int memory : {11, 19, 2019, 1024124}) {
    for (int align : {16, (int)kDefaultMemoryAlignment, 512, 1024, 4096}) {
      std::shared_ptr<Memory> result = Memory::Alloc(dev, memory, align);
      ASSERT_EQ(result.use_count(), 2);
      int64_t address = (int64_t)result->data;
      ASSERT_EQ(address % align, 0);
    }
  }
  auto pool_size = Memory::GetPoolSize(dev);
  ASSERT_EQ(pool_size.first, 0);  // No chunk is used.

  std::shared_ptr<Memory> result = Memory::Alloc(dev, 4096, 64);
  pool_size = Memory::GetPoolSize(dev);
  auto used_size = pool_size.first * 1048576.0;
  auto abs_diff = (used_size > 4096) ? used_size - 4096 : 4096 - used_size;
  ASSERT_LE(abs_diff, 1);
  result.reset();
  pool_size = Memory::GetPoolSize(dev);
  ASSERT_EQ(pool_size.first, 0);
  Memory::RemovePool(dev);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
