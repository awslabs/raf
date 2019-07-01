#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CALL(func)                                           \
  {                                                               \
    cudaError_t e = (func);                                       \
    CHECK(e == cudaSuccess) << "CUDA: " << cudaGetErrorString(e); \
  }
