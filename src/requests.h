/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/requests.h
 * \brief Resource request underlying each operator
 */
#pragma once
#include <memory>
#include <vector>
#include "raf/device.h"
#include "raf/memory_pool.h"
#include "raf/stream_pool.h"

namespace raf {
namespace requests {

class Requests {
 public:
  struct MemoryRequest {
    void** dest;
    Device device;
    int64_t nbytes;
    std::shared_ptr<memory_pool::Memory> memory;
  };

  struct WorkspaceRequest {
    void** dest;
    Device device;
    int64_t nbytes;
    std::shared_ptr<memory_pool::Memory> memory;
  };

  struct StreamRequest {
    void** dest;
    Device device;
    int tag_idx;
    int stream_idx;
    std::shared_ptr<stream_pool::Stream> stream;
  };

  struct DistributedRequest {
    void** dest;
  };

 public:
  std::vector<MemoryRequest> memory;
  std::vector<WorkspaceRequest> workspace;
  std::vector<StreamRequest> stream;
  std::vector<DistributedRequest> distributed;
};

}  // namespace requests
}  // namespace raf
