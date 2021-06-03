/*!
 * Copyright (c) 2019 by Contributors
 * \file src/device_api/cuda/cuda.cc
 * \brief CUDA device API
 */
#include "mnm/device_api.h"
#include "mnm/registry.h"
#include "../../common/cuda_utils.h"

namespace mnm {
namespace device_api {
namespace cuda {

class CUDADeviceAPI final : public DeviceAPI {
 public:
  CUDADeviceAPI() = default;

  ~CUDADeviceAPI() = default;

  int GetDeviceCount() {
    int count = 0;
    CUDA_CALL(cudaGetDeviceCount(&count));
    return count;
  }

  void* AllocMemory(int64_t nbytes, int64_t alignment) override {
    void* ptr = nullptr;
    // TODO(@junrushao1994): make sure it is correct
    CHECK_EQ(512 % alignment, 0);
#if CUDA_VERSION >= 11020
    // TODO(@comaniac): Specify stream ID when multi-stream is enabled.
    CUDA_CALL(cudaMallocAsync(&ptr, nbytes, 0));
#else
    CUDA_CALL(cudaMalloc(&ptr, nbytes));
#endif
    return ptr;
  }

  void FreeMemory(void* ptr) override {
#if CUDA_VERSION >= 11020
    // TODO(@comaniac): Specify stream ID when multi-stream is enabled.
    CUDA_CALL(cudaFreeAsync(ptr, 0));
#else
    CUDA_CALL(cudaFree(ptr));
#endif
  }

  void* CreateStream(const Device& dev) override {
    CHECK_EQ(dev.device_type, DevType::kCUDA());
    CUDA_CALL(cudaSetDevice(dev.device_id));
    cudaStream_t ret = nullptr;
    CUDA_CALL(cudaStreamCreate(&ret));
    return ret;
  }

  void FreeStream(const Device& dev, void* stream) override {
    CHECK_EQ(dev.device_type, DevType::kCUDA());
    CUDA_CALL(cudaSetDevice(dev.device_id));
    CUDA_CALL(cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
  }

  void SyncStream(const Device& prev_dev, void* prev, void* next) override {
    throw;
  }

  void WaitDevice(const Device& dev) override {
    CHECK_EQ(dev.device_type, DevType::kCUDA());
    CUDA_CALL(cudaSetDevice(dev.device_id));
    CUDA_CALL(cudaDeviceSynchronize());
  }

  void WaitStream(const Device& dev, void* stream) override {
    CHECK_EQ(dev.device_type, DevType::kCUDA());
    CHECK(stream != nullptr) << "Cannot sync a null stream";
    CUDA_CALL(cudaSetDevice(dev.device_id));
    CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

  static void* make() {
    return new CUDADeviceAPI();
  }
};

MNM_REGISTER_GLOBAL("mnm.device_api._make.cuda").set_body_typed(CUDADeviceAPI::make);

}  // namespace cuda
}  // namespace device_api
}  // namespace mnm
