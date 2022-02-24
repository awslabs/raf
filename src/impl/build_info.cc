/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/build_info.cc
 * \brief Reflect build-time information and expose to the frontend
 */
#include "raf/registry.h"
#ifdef RAF_USE_NCCL
#include <nccl.h>
#endif

namespace raf {
namespace build_info {

std::string GitVersion() {
  return RAF_GIT_VERSION;
}

bool UseCUDA() {
#ifdef RAF_USE_CUDA
  return true;
#else
  return false;
#endif
}

std::string UseCuBLAS() {
  return RAF_USE_CUBLAS;
}

std::string UseCuDNN() {
  return RAF_USE_CUDNN;
}

std::string UseLLVM() {
  return RAF_USE_LLVM;
}

std::string UseMPI() {
  return RAF_USE_MPI;
}

bool UseNCCL() {
#ifdef RAF_USE_NCCL
  return true;
#else
  return false;
#endif
}

int NCCLVersion() {
#ifdef RAF_USE_NCCL
  return NCCL_VERSION_CODE;
#else
  return 0;
#endif
}
std::string UseCUTLASS() {
  return RAF_USE_CUTLASS;
}

std::string CudaVersion() {
  return RAF_CUDA_VERSION;
}

std::string CudnnVersion() {
  return RAF_CUDNN_VERSION;
}

std::string CmakeBuildType() {
  return RAF_CMAKE_BUILD_TYPE;
}

RAF_REGISTER_GLOBAL("raf.build_info.git_version").set_body_typed(GitVersion);
RAF_REGISTER_GLOBAL("raf.build_info.cuda_version").set_body_typed(CudaVersion);
RAF_REGISTER_GLOBAL("raf.build_info.use_cuda").set_body_typed(UseCUDA);
RAF_REGISTER_GLOBAL("raf.build_info.use_cublas").set_body_typed(UseCuBLAS);
RAF_REGISTER_GLOBAL("raf.build_info.use_cudnn").set_body_typed(UseCuDNN);
RAF_REGISTER_GLOBAL("raf.build_info.cudnn_version").set_body_typed(CudnnVersion);
RAF_REGISTER_GLOBAL("raf.build_info.cmake_build_type").set_body_typed(CmakeBuildType);
RAF_REGISTER_GLOBAL("raf.build_info.use_llvm").set_body_typed(UseLLVM);
RAF_REGISTER_GLOBAL("raf.build_info.use_mpi").set_body_typed(UseMPI);
RAF_REGISTER_GLOBAL("raf.build_info.use_nccl").set_body_typed(UseNCCL);
RAF_REGISTER_GLOBAL("raf.build_info.use_cutlass").set_body_typed(UseCUTLASS);
RAF_REGISTER_GLOBAL("raf.build_info.nccl_version").set_body_typed(NCCLVersion);
}  // namespace build_info
}  // namespace raf
