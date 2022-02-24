/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file memory_pool.h
 * \brief Memory pool API
 */
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "./device.h"

namespace raf {
namespace memory_pool {

class MemoryPool;

/*!
 * \brief A wrapper for a chunk of memory, which may have shared reference to the pool so that they
 * are freed correctly. Interaction between memory pool manager also happens here.
 *
 * This wrapper is the base wrapper for memory.
 *
 * \sa Memory
 */
class Memory {
 public:
  virtual ~Memory() = default;

 public:
  static int64_t GetAllocBytes(const Device& dev, int64_t nbytes);

  static std::shared_ptr<Memory> Alloc(const Device& dev, int64_t nbytes,
                                       int64_t alignment = kDefaultMemoryAlignment);

  static std::shared_ptr<Memory> AllocAsync(const Device& dev, int64_t nbytes, void* stream,
                                            int64_t alignment = kDefaultMemoryAlignment);

  static std::vector<std::shared_ptr<Memory> > AllocBatch(
      const Device& dev, const std::vector<int64_t>& nbytes,
      int64_t alignment = kDefaultMemoryAlignment);

  static std::pair<float, float> GetPoolSize(const Device& dev);

  // means "no longer considered as allocator when asking for new memory."
  static void RemovePool(const Device& dev);

  static MemoryPool* ResetPool(const Device& dev);

  static MemoryPool* GetPool(const Device& dev);

  static MemoryPool* InitPool(const Device& dev, const std::string& name);

 public:
  /*! \brief The pointer to the allocated chunk of memory. */
  void* data = nullptr;
  /*! \brief The context of the allocated chunk of memory. */
  Device device{};
};

/*!
 * \brief A base class for memory pool.
 * Only interface for implementing new allocation strategy, no static interface is included.
 */
class MemoryPool {
 public:
  virtual ~MemoryPool() = default;

  /*!
   * \brief Get the name of this memory pool.
   * \return The memory pool name.
   */
  virtual std::string GetName() = 0;

  /*!
   * \brief Calculate the actual bytes to be allocated. This may be different as the requested
   * size due to alignment or page unit.
   * \param nbytes The requested bytes to be allocated.
   */
  virtual int64_t GetAllocBytes(int64_t nbytes) = 0;

  /*!
   * \brief A helper function to change bytes to mega-bytes.
   *
   * \param nbytes The number in bytes.
   *
   * \return The number in MBs.
   */
  inline float BytesToMegaBytes(float nbytes) {
    return nbytes / 1048576.0;
  }

  /*!
   * \brief Allocate a chunk of memory with given size and alignment.
   *
   * \param nbytes The size of the memory chunk to allocate.
   * \param alignment The alignment of the memory chunk to allocate.
   *
   * \return A shared pointer to Memory object which holds the memory chunk.
   */
  virtual std::shared_ptr<Memory> Alloc(int64_t nbytes,
                                        int64_t alignment = kDefaultMemoryAlignment) = 0;

  virtual std::shared_ptr<Memory> AllocAsync(int64_t nbytes, void* stream,
                                             int64_t alignment = kDefaultMemoryAlignment) = 0;

  /*!
   * \brief Allocate a bacth of memory chunks with given sizes and alignments.
   *
   * \param nbytes The sizes of the memory chunks to allocate.
   * \param alignment The alignments of the memory chunks to allocate.
   *
   * \return The shared pointers to Memory object which hold the memory chunks.
   */
  virtual std::vector<std::shared_ptr<Memory> > AllocBatch(
      const std::vector<int64_t>& nbytes, int64_t alignment = kDefaultMemoryAlignment) = 0;

  /*!
   * \brief Get the current pool size in MBs.
   *
   * \return A pair of the total size of (used chunks, pool).
   */
  virtual std::pair<float, float> GetPoolSize() = 0;
};

}  // namespace memory_pool
}  // namespace raf
