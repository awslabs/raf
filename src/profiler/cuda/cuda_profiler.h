/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/profiler/cuda/cuda_profiler.h
 * \brief async profiler for cuda operations
 */
#pragma once

#include <vector>
#include <string>
#include <memory>
#include "raf/registry.h"
#include "raf/profiler.h"
#include "raf/event_pool.h"

#if RAF_USE_CUDA

#define WITH_CUDA_PROFILER_LEVEL(LEVEL, CTX, STREAM, NAME, CAT, ARGS, CODE_SNIPPET)         \
  {                                                                                         \
    bool _profiling = raf::profiler::Profiler::Get()->IsProfiling(LEVEL);                   \
    if (_profiling) {                                                                       \
      auto& _pool = raf::profiler::CudaProfiler::Get()->HelperPool();                       \
      auto _phelper_index = _pool.size();                                                   \
      _pool.push_back(raf::profiler::CudaProfilerHelper(CTX.device_id(), CTX.device_type(), \
                                                        STREAM, NAME, CAT, ARGS));          \
      _pool[_phelper_index].start();                                                        \
      CODE_SNIPPET                                                                          \
      _pool[_phelper_index].stop();                                                         \
    } else {                                                                                \
      CODE_SNIPPET                                                                          \
    }                                                                                       \
  }

#define WITH_CUDA_PROFILER(CTX, STREAM, NAME, CAT, ARGS, CODE_SNIPPET) \
  WITH_CUDA_PROFILER_LEVEL(1, CTX, STREAM, NAME, CAT, ARGS, CODE_SNIPPET)

#else

#define WITH_CUDA_PROFILER(CTX, STREAM, NAME, CAT, ARGS, CODE_SNIPPET) \
  { CODE_SNIPPET }
#define WITH_CUDA_PROFILER_LEVEL(LEVEL, CTX, STREAM, NAME, CAT, ARGS, CODE_SNIPPET) \
  { CODE_SNIPPET }

#endif

namespace raf {
namespace profiler {

using raf::event_pool::Event;

class CudaProfilerHelper : public ProfilerHelper {
 public:
  CudaProfilerHelper(int dev_id, raf::DevType dev_type, void* stream, std::string name,
                     std::string categories, std::vector<std::string> args = {});

  void start() {
    cuda_api->EventRecordOnStream(start_event->data(), stream);
  }
  void stop() {
    cuda_api->EventRecordOnStream(end_event->data(), stream);
  }
  void collect();

 public:
  static std::shared_ptr<device_api::DeviceAPI> cuda_api;
  void* stream;
  std::shared_ptr<Event> start_event;
  std::shared_ptr<Event> end_event;
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
  /*!
   * \brief Get the elapsed time in microsecond given an event
   * \param event The event.
   * \return The elapsed time in microsecond.
   */
  uint64_t GetElapsedTimeInMicrosec(std::shared_ptr<Event> event);
  /*! \brief Collect CUDA stats. */
  void CollectCudaStat();
  /*! \brief Clear CUDA stats. */
  void ClearCudaStat();

  std::vector<CudaProfilerHelper>& HelperPool() {
    return cuda_helpers_;
  }

 private:
  CudaProfiler();

  /*! \brief The list of CUDA profiler helpers. */
  std::vector<CudaProfilerHelper> cuda_helpers_;
  /*! \brief The start event of the CUDA profiler. */
  std::shared_ptr<Event> start_event_;
  /*! \brief The start time of the CUDA profiler in us. */
  uint64_t start_time_ = 0;
  /*! \brief Indicate whether the CUDA profiler starts. */
  bool started_ = false;
};

}  // namespace profiler
}  // namespace raf
