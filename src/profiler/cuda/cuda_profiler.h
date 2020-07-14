/*!
 * Copyright (c) 2019 by Contributors
 * \file src/common/cuda_profiler.h
 * \brief async profiler for cuda operations
 */
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <memory>
#include "mnm/registry.h"
#include "mnm/profiler.h"

#define WITH_CUDA_PROFILER(CTX, NAME, CAT, ARGS, CODE_SNIPPET)                               \
  {                                                                                             \
    bool profiling = profiler::Profiler::Get()->IsProfiling();                                  \
    std::vector<profiler::ProfilerCudaHelper>& profiling_helpers =                              \
        profiler::CudaProfiler::Get()->GetProfilingHelpers();                                   \
    if (profiling) {                                                                            \
      profiler::ProfilerCudaHelper phelper(profiling, CTX.device_id, CTX.device_type, NAME, CAT, \
                                          ARGS);                                                \
      profiling_helpers.push_back(phelper);                                                      \
      profiling_helpers.back().start();                                                         \
      cudaEventRecord(profiling_helpers.back().start_event, 0);                                 \
      CODE_SNIPPET                                                                              \
      cudaEventRecord(profiling_helpers.back().end_event, 0);                                   \
    } else {                                                                                    \
      CODE_SNIPPET                                                                              \
    }                                                                                           \
  }

namespace mnm {
namespace profiler {

class ProfilerCudaHelper : public ProfilerBaseHelper {
 public:
  ProfilerCudaHelper(bool profiling, uint32_t dev_id, mnm::DevType dev_type, std::string name,
                     std::string categories, std::vector<std::string> args = {})
      : ProfilerBaseHelper(profiling, dev_id, dev_type, name, categories, args) {
  }
  void start();
  void stop();

 public:
  cudaEvent_t start_event;
  cudaEvent_t end_event;
};

class CudaProfiler {
 public:
  CudaProfiler();
  ~CudaProfiler();
  static CudaProfiler* Get(std::shared_ptr<CudaProfiler>* sp = nullptr);
  std::vector<ProfilerCudaHelper>& GetProfilingHelpers();
  void CollectCudaStat();

 private:
  std::vector<ProfilerCudaHelper> cuda_helpers;
};

}  // namespace profiler
}  // namespace mnm
