/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/requests.h
 * \brief Resource request underlying each operator
 */
#pragma once
#include <memory>
#include <vector>
#include "mnm/device.h"
#include "mnm/memory_pool.h"
#include "mnm/stream_pool.h"

namespace mnm {
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
}  // namespace mnm
