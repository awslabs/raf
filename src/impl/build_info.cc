/*!
 * Copyright (c) 2019 by Contributors
 * \file src/impl/build_info.cc
 * \brief Reflect build-time information and expose to the frontend
 */
#include "mnm/registry.h"

namespace mnm {
namespace build_info {

std::string GitVersion() {
  return MNM_GIT_VERSION;
}

bool UseCUDA() {
#ifdef MNM_USE_CUDA
  return true;
#else
  return false;
#endif
}

std::string UseCuDNN() {
  return MNM_USE_CUDNN;
}

std::string UseLLVM() {
  return MNM_USE_CUDNN;
}

std::string UseMPI() {
  return MNM_USE_MPI;
}

std::string UseNCCL() {
  return MNM_USE_NCCL;
}

std::string UseCUTLASS() {
  return MNM_USE_CUTLASS;
}

MNM_REGISTER_GLOBAL("mnm.build_info.git_version").set_body_typed(GitVersion);
MNM_REGISTER_GLOBAL("mnm.build_info.use_cuda").set_body_typed(UseCUDA);
MNM_REGISTER_GLOBAL("mnm.build_info.use_cudnn").set_body_typed(UseCuDNN);
MNM_REGISTER_GLOBAL("mnm.build_info.use_llvm").set_body_typed(UseLLVM);
MNM_REGISTER_GLOBAL("mnm.build_info.use_mpi").set_body_typed(UseMPI);
MNM_REGISTER_GLOBAL("mnm.build_info.use_nccl").set_body_typed(UseNCCL);
MNM_REGISTER_GLOBAL("mnm.build_info.use_cutlass").set_body_typed(UseCUTLASS);
}  // namespace build_info
}  // namespace mnm
