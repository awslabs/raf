/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/device_api/cuda/cuda_host.cc
 * \brief CUDA host (CPU pinned memory) device API.
 */
#include <tvm/runtime/device_api.h>
#include "raf/device_api.h"
#include "raf/registry.h"
#include "../../common/cuda_utils.h"

namespace raf {
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
    CUDA_CALL_IF_DRIVER_IS_LOADED(cudaFreeHost(ptr));
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

RAF_REGISTER_GLOBAL("raf.device_api._make.cuda_host").set_body_typed(CUDAHostDeviceAPI::make);

}  // namespace cuda_host
}  // namespace device_api
}  // namespace raf
