#include <mnm/device_api.h>
#include <mnm/registry.h>

#include "../../common/cuda.h"

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

  void SetDevice(int device_id) override {
    CUDA_CALL(cudaSetDevice(device_id));
  }

  int GetDevice() override {
    int device_id = -1;
    CUDA_CALL(cudaGetDevice(&device_id));
    return device_id;
  }

  void* AllocMemory(int64_t nbytes, int64_t alignment) override {
    void* ptr = nullptr;
    // TODO(@junrushao1994): make sure it is correct
    CHECK(512 % alignment == 0);
    CUDA_CALL(cudaMalloc(&ptr, nbytes));
    return ptr;
  }

  void FreeMemory(void* ptr) override {
    CUDA_CALL(cudaFree(ptr));
  }

  void* CreateStream() override {
    cudaStream_t ret = nullptr;
    CUDA_CALL(cudaStreamCreate(&ret));
    return ret;
  }

  void FreeStream(void* stream) override {
    CUDA_CALL(cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
  }

  void SyncStream(const Context& prev_ctx, void* prev, void* next) override {
    throw;
  }

  void WaitDevice() override {
    CUDA_CALL(cudaDeviceSynchronize());
  }

  void WaitStream(void* stream) override {
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
