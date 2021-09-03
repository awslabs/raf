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
    CUDA_CALL(cudaMalloc(&ptr, nbytes));
    return ptr;
  }

  void FreeMemory(void* ptr) override {
    CUDA_CALL(cudaFree(ptr));
  }

#if CUDA_VERSION >= 11030
  void SetDevice(const int dev_id) override {
    device_id_ = dev_id;
  }

  static cudaMemPool_t GetCUDAMemoryPool(int dev_id) {
    cudaMemPool_t mem_pool;
    CUDA_CALL(cudaDeviceGetDefaultMemPool(&mem_pool, dev_id));

    cuuint64_t setVal = UINT64_MAX;
    CUDA_CALL(cudaMemPoolSetAttribute(mem_pool, cudaMemPoolAttrReleaseThreshold, &setVal));
    return mem_pool;
  }

  std::pair<int64_t, int64_t> GetPoolSize() override {
    cudaMemPool_t mem_pool;
    CUDA_CALL(cudaDeviceGetDefaultMemPool(&mem_pool, device_id_));
    cuuint64_t allocated;
    cuuint64_t used;

    CUDA_CALL(cudaMemPoolGetAttribute(mem_pool, cudaMemPoolAttrReservedMemCurrent, &allocated));
    CUDA_CALL(cudaMemPoolGetAttribute(mem_pool, cudaMemPoolAttrUsedMemCurrent, &used));
    return {used, allocated};
  }

  void* AllocMemoryAsync(int64_t nbytes, void* stream,
                         int64_t alignment = kDefaultMemoryAlignment) {
    static auto cuda_pool = GetCUDAMemoryPool(device_id_);
    void* ptr = nullptr;

    // TODO(@junrushao1994): make sure it is correct
    CHECK_EQ(512 % alignment, 0);

    try {
      CUDA_CALL(
          cudaMallocFromPoolAsync(&ptr, nbytes, cuda_pool, static_cast<cudaStream_t>(stream)));
    } catch (const dmlc::Error& e) {
      CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
      CUDA_CALL(
          cudaMallocFromPoolAsync(&ptr, nbytes, cuda_pool, static_cast<cudaStream_t>(stream)));
    }
    return ptr;
  }

  void FreeMemoryAsync(void* ptr, void* stream) {
    CUDA_CALL(cudaFreeAsync(ptr, static_cast<cudaStream_t>(stream)));
  }
#else
  void SetDevice(const int dev_id) override {
    device_id_ = dev_id;
    CUDA_CALL(cudaSetDevice(dev_id));
  }

  void* AllocMemoryAsync(int64_t nbytes, void* stream,
                         int64_t alignment = kDefaultMemoryAlignment) {
    LOG(FATAL) << "AllocMemroyAsync requires CUDA Version >= 11.3";
  }

  void FreeMemoryAsync(void* ptr, void* stream) {
    LOG(FATAL) << " FreeMemoryAsync requires CUDA Version >= 11.3";
  }
#endif

  void* CreateStream(const Device& dev) override {
    CHECK_EQ(dev.device_type(), DevType::kCUDA());
    CUDA_CALL(cudaSetDevice(dev.device_id()));
    cudaStream_t ret = nullptr;
    CUDA_CALL(cudaStreamCreate(&ret));
    return ret;
  }

  void FreeStream(const Device& dev, void* stream) override {
    CHECK_EQ(dev.device_type(), DevType::kCUDA());
    CUDA_CALL(cudaSetDevice(dev.device_id()));
    CUDA_CALL(cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
  }

  void* CreateEvent(const Device& dev, uint32_t flags) override {
    CHECK_EQ(dev.device_type(), DevType::kCUDA());
    cudaEvent_t event;
    CUDA_CALL(cudaSetDevice(dev.device_id()));
    CUDA_CALL(cudaEventCreate(&event, flags));
    return event;
  }

  void FreeEvent(const Device& dev, void* event) override {
    CHECK_EQ(dev.device_type(), DevType::kCUDA());
    CUDA_CALL(cudaSetDevice(dev.device_id()));
    CUDA_CALL(cudaEventDestroy(static_cast<cudaEvent_t>(event)));
  }

  // Between event and stream
  void EventRecordOnStream(const Device& dev, void* event, void* stream) override {
    CHECK_EQ(dev.device_type(), DevType::kCUDA());
    CUDA_CALL(cudaSetDevice(dev.device_id()));
    CUDA_CALL(cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream)));
  }

  void StreamWaitEvent(const Device& dev, void* stream, void* event) override {
    CHECK_EQ(dev.device_type(), DevType::kCUDA());
    CUDA_CALL(cudaSetDevice(dev.device_id()));
    CUDA_CALL(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream),
                                  static_cast<cudaEvent_t>(event), 0 /*cudaEventWaitDefault*/));
  }

  void SyncStream(const Device& prev_dev, void* prev, void* next) override {
    throw;
  }

  void WaitDevice(const Device& dev) override {
    CHECK_EQ(dev.device_type(), DevType::kCUDA());
    CUDA_CALL(cudaSetDevice(dev.device_id()));
    CUDA_CALL(cudaDeviceSynchronize());
  }

  void WaitStream(const Device& dev, void* stream) override {
    CHECK_EQ(dev.device_type(), DevType::kCUDA());
    CHECK(stream != nullptr) << "Cannot sync a null stream";
    CUDA_CALL(cudaSetDevice(dev.device_id()));
    CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

  static void* make() {
    return new CUDADeviceAPI();
  }

 private:
  int device_id_;
};

MNM_REGISTER_GLOBAL("mnm.device_api._make.cuda").set_body_typed(CUDADeviceAPI::make);

}  // namespace cuda
}  // namespace device_api
}  // namespace mnm
