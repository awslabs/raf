/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/device_api/cuda/cuda.cc
 * \brief CUDA device API
 */
#include <tvm/runtime/device_api.h>
#include "raf/op.h"
#include "raf/device_api.h"
#include "raf/registry.h"
#include "raf/profiler.h"
#include "../../common/cuda_utils.h"

#include "../../op/dialect/cudnn/cudnn_utils.h"
#include "../../op/dialect/cublas/cublas_utils.h"
#include "../../op/dialect/cutlass/cutlass_utils.h"

namespace raf {
namespace device_api {
namespace cuda {

class CUDADeviceAPI final : public DeviceAPI {
 public:
  CUDADeviceAPI() = default;

  ~CUDADeviceAPI() = default;

  int GetDeviceCount() override {
    int count = 0;
    CUDA_CALL(cudaGetDeviceCount(&count));
    return count;
  }

  void* AllocMemory(int64_t nbytes, int64_t alignment) override {
    CUDA_CALL(cudaSetDevice(device_id_));
    void* ptr = nullptr;
    CUDA_CALL(cudaMalloc(&ptr, nbytes));
    return ptr;
  }

  void FreeMemory(void* ptr) override {
    CUDA_CALL_IF_DRIVER_IS_LOADED(cudaFree(ptr));
  }

#if CUDA_VERSION >= 11030
  void SetDevice(const int dev_id) override {
    device_id_ = dev_id;
    CUDA_CALL(cudaSetDevice(dev_id));
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
    CUDA_CALL_IF_DRIVER_IS_LOADED(cudaFreeAsync(ptr, static_cast<cudaStream_t>(stream)));
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

  void CopyDataFromTo(DLTensor* from, DLTensor* to, void* stream) final {
    size_t nbytes = tvm::runtime::GetDataSize(*from);
    ICHECK_EQ(nbytes, tvm::runtime::GetDataSize(*to));
    ICHECK(tvm::runtime::IsContiguous(*from) && tvm::runtime::IsContiguous(*to))
        << "CopyDataFromTo only support contiguous array for now";

    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    auto from_data_ptr = static_cast<const char*>(from->data) + from->byte_offset;
    auto to_data_ptr = static_cast<char*>(to->data) + to->byte_offset;

    auto from_dev_type =
        (from->device.device_type == kDLCUDAHost) ? kDLCPU : from->device.device_type;
    auto to_dev_type = (to->device.device_type == kDLCUDAHost) ? kDLCPU : to->device.device_type;

    // In case there is a copy from host memory to host memory.
    if (to_dev_type == kDLCPU && from_dev_type == kDLCPU) {
      memcpy(to_data_ptr, from_data_ptr, nbytes);
      return;
    }

    auto curr_device_id = device_id_;
    if (from_dev_type == kDLCUDA && to_dev_type == kDLCUDA) {
      // GPU to another GPU.
      SetDevice(from->device.device_id);
      if (from->device.device_id == to->device.device_id) {
        HandleCopy(from_data_ptr, to_data_ptr, nbytes, cudaMemcpyDeviceToDevice, cu_stream);
      } else {
        cudaMemcpyPeerAsync(to_data_ptr, to->device.device_id, from_data_ptr,
                            from->device.device_id, nbytes, cu_stream);
      }
    } else if (from_dev_type == kDLCUDA && to_dev_type == kDLCPU) {
      // GPU to CPU.
      SetDevice(from->device.device_id);
      HandleCopy(from_data_ptr, to_data_ptr, nbytes, cudaMemcpyDeviceToHost, cu_stream);
    } else if (from_dev_type == kDLCPU && to_dev_type == kDLCUDA) {
      // CPU to GPU.
      SetDevice(to->device.device_id);
      HandleCopy(from_data_ptr, to_data_ptr, nbytes, cudaMemcpyHostToDevice, cu_stream);
    } else {
      LOG(FATAL) << "expect copy from/to GPU or between GPU";
    }

    SetDevice(curr_device_id);
  }

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

  void SetStream(const Device& dev, void* stream) override {
    stream_ = stream;
    tvm::runtime::DeviceAPI::Get(dev)->SetStream(dev, stream);
    raf::op::cudnn::SetStream(static_cast<cudaStream_t>(stream));
    raf::op::cublas::SetStream(static_cast<cudaStream_t>(stream));
  }

  void* GetStream() override {
    return stream_;
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

  float EventElapsedTimeInMilliSeconds(void* start_event, void* end_event) override {
    float elapsed_time;
    int dev_id;
    CUDA_CALL(cudaGetDevice(&dev_id));
    CUDA_CALL(cudaEventElapsedTime(&elapsed_time, static_cast<cudaEvent_t>(start_event),
                                   static_cast<cudaEvent_t>(end_event)));
    return elapsed_time;
  }

  // Between event and stream
  void EventRecordOnStream(void* event, void* stream) override {
    CUDA_CALL(cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream)));
  }

  void StreamWaitEvent(void* stream, void* event) override {
    CUDA_CALL(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream),
                                  static_cast<cudaEvent_t>(event), 0 /*cudaEventWaitDefault*/));
  }

  void WaitDevice(const Device& dev) override {
    CHECK_EQ(dev.device_type(), DevType::kCUDA());
    CUDA_CALL(cudaSetDevice(dev.device_id()));
    CUDA_CALL(cudaDeviceSynchronize());
  }

  void WaitStream(void* stream) override {
    CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

  virtual void WaitEvent(void* event) override {
    CHECK(event != nullptr) << "Cannot sync a null event";
    CUDA_CALL(cudaEventSynchronize(static_cast<cudaEvent_t>(event)));
  }

  static void* make() {
    return new CUDADeviceAPI();
  }

 private:
  static void HandleCopy(const void* from, void* to, size_t size, cudaMemcpyKind kind,
                         cudaStream_t cu_stream) {
    if (cu_stream != nullptr) {
      // Note that async happens only for the CUDA-CUDA and CUDA-CUDAHost (pinned CPU memory).
      // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution
      CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, cu_stream));
    } else {
      CUDA_CALL(cudaMemcpy(to, from, size, kind));
    }
  }

  int device_id_;
  // using cuda default stream if stream is not set explicitly
  void* stream_ = nullptr;
};

RAF_REGISTER_GLOBAL("raf.device_api._make.cuda").set_body_typed(CUDADeviceAPI::make);

}  // namespace cuda
}  // namespace device_api
}  // namespace raf
