/*!
 * Copyright (c) 2021 by Contributors
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

#define WITH_CUDA_PROFILER_LEVEL(LEVEL, CTX, NAME, CAT, ARGS, CODE_SNIPPET)                        \
  {                                                                                                \
    bool profiling = mnm::profiler::Profiler::Get()->IsProfiling(LEVEL);                           \
    if (profiling) {                                                                               \
      std::shared_ptr<mnm::profiler::CudaProfilerHelper> phelper(                                  \
          new mnm::profiler::CudaProfilerHelper(CTX.device_id, CTX.device_type, NAME, CAT, ARGS)); \
      mnm::profiler::CudaProfiler::Get()->AddProfilerHelper(phelper);                              \
      cudaEventRecord(phelper->start_event, 0);                                                    \
      CODE_SNIPPET                                                                                 \
      cudaEventRecord(phelper->end_event, 0);                                                      \
    } else {                                                                                       \
      CODE_SNIPPET                                                                                 \
    }                                                                                              \
  }

#define WITH_CUDA_PROFILER(CTX, NAME, CAT, ARGS, CODE_SNIPPET) \
  WITH_CUDA_PROFILER_LEVEL(1, CTX, NAME, CAT, ARGS, CODE_SNIPPET)

namespace mnm {
namespace profiler {

class CudaProfilerHelper : public ProfilerHelper {
 public:
  CudaProfilerHelper(int dev_id, mnm::DevType dev_type, std::string name, std::string categories,
                     std::vector<std::string> args = {});

  void start();
  void stop();

 public:
  cudaEvent_t start_event;
  cudaEvent_t end_event;
};

class CudaProfiler {
 public:
  ~CudaProfiler();
  static CudaProfiler* Get();
  /*! \brief Start the CUDA profiler. start_event_ and start_time_ will be initialized if the CUDA
   *    profiler has not started yet. */
  void start();
  /*! \brief Stop the CUDA profiler. */
  void stop();
  /*! \brief Get a CUDA event. */
  cudaEvent_t GetCudaEvent();
  /*!
   * \brief Release a CUDA event back to CUDA profiler.
   * \param event The event to release.
   */
  void ReleaseCudaEvent(cudaEvent_t event);
  /*!
   * \brief Get the elapsed time in microsecond given an event
   * \param event The event.
   * \return The elapsed time in microsecond.
   */
  uint64_t GetElapsedTimeInMicrosec(cudaEvent_t event);
  /*!
   * \brief Add a profiler helper.
   * \param helper The profiler helper
   */
  void AddProfilerHelper(std::shared_ptr<CudaProfilerHelper> helper);
  /*! \brief Collect CUDA stats. */
  void CollectCudaStat();

 private:
  CudaProfiler();

  /*! \brief The list of CUDA profiler helpers. */
  std::vector<std::shared_ptr<CudaProfilerHelper>> cuda_helpers_;
  /*! \brief The CUDA event pool. */
  std::vector<cudaEvent_t> cuda_events_;
  /*! \brief The start event of the CUDA profiler. */
  cudaEvent_t start_event_;
  /*! \brief The start time of the CUDA profiler in us. */
  uint64_t start_time_ = 0;
  /*! \brief Indicate whether the CUDA profiler starts. */
  bool started_ = false;
};

}  // namespace profiler
}  // namespace mnm
