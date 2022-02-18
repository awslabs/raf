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
 * \file src/device_api/cuda/cuda_host.cc
 * \brief CUDA host (CPU pinned memory) device API.
 */
#include <tvm/runtime/device_api.h>
#include "mnm/device_api.h"
#include "mnm/registry.h"
#include "../../common/cuda_utils.h"

namespace mnm {
namespace device_api {
namespace cuda_host {

class CUDAHostDeviceAPI final : public DeviceAPI {
 public:
  CUDAHostDeviceAPI() = default;

  ~CUDAHostDeviceAPI() = default;

  int GetDeviceCount() override {
    int count = 0;
    CUDA_CALL(cudaGetDeviceCount(&count));
    return count;
  }

  void* AllocMemory(int64_t nbytes, int64_t alignment) override {
    void* ptr = nullptr;
    CHECK_EQ(512 % alignment, 0);
    CUDA_CALL(cudaMallocHost(&ptr, nbytes));
    return ptr;
  }

  void FreeMemory(void* ptr) override {
    CUDA_CALL(cudaFreeHost(ptr));
  }

  void SetDevice(const int dev_id) override {
    throw;
  }

  void* AllocMemoryAsync(int64_t nbytes, void* stream,
                         int64_t alignment = kDefaultMemoryAlignment) {
    throw;
  }

  void FreeMemoryAsync(void* ptr, void* stream) {
    throw;
  }

  void CopyDataFromTo(DLTensor* from, DLTensor* to, void* stream) final {
    throw;
  }

  void* CreateStream(const Device& dev) override {
    throw;
  }

  void FreeStream(const Device& dev, void* stream) override {
    throw;
  }

  void SetStream(const Device& dev, void* stream) override {
    throw;
  }

  void* GetStream() override {
    return nullptr;
  }

  void* CreateEvent(const Device& dev, uint32_t flags) override {
    throw;
  }

  void FreeEvent(const Device& dev, void* event) override {
    throw;
  }

  float EventElapsedTimeInMilliSeconds(void* start_event, void* end_event) override {
    throw;
  }

  void EventRecordOnStream(void* event, void* stream) override {
    throw;
  }

  void StreamWaitEvent(void* stream, void* event) override {
    throw;
  }

  void WaitDevice(const Device& dev) override {
    // Do nothing
  }

  void WaitStream(void* stream) override {
    throw;
  }

  virtual void WaitEvent(void* event) override {
    throw;
  }

  static void* make() {
    return new CUDAHostDeviceAPI();
  }
};

MNM_REGISTER_GLOBAL("mnm.device_api._make.cuda_host").set_body_typed(CUDAHostDeviceAPI::make);

}  // namespace cuda_host
}  // namespace device_api
}  // namespace mnm
