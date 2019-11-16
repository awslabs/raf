/*!
 * Copyright (c) 2019 by Contributors
 * \file src/common/cuda_utils.h
 * \brief Utilities for cuda
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CALL(func)                                           \
  do {                                                            \
    cudaError_t e = (func);                                       \
    CHECK(e == cudaSuccess) << "CUDA: " << cudaGetErrorString(e); \
  } while (false)
