/*!
 * Copyright (c) 2019 by Contributors
 * \file src/common/cuda_profiler.cc
 * \brief async profiler for cuda operations
 */
#include "./cuda_profiler.h"
#include "../../common/cuda_utils.h"

namespace mnm {
namespace profiler {

CudaProfilerHelper::CudaProfilerHelper(int dev_id, mnm::DevType dev_type, std::string name,
                                       std::string categories, std::vector<std::string> args)
    : ProfilerHelper(dev_id, dev_type, name, categories, args) {
  start_event = CudaProfiler::Get()->GetCudaEvent();
  end_event = CudaProfiler::Get()->GetCudaEvent();
}

void CudaProfilerHelper::start() {
  start_time_ = ProfileStat::NowInMicrosec();
}

void CudaProfilerHelper::stop() {
  CUDA_CALL(cudaEventSynchronize(end_event));
  auto cuda_profiler = CudaProfiler::Get();
  start_time_ = cuda_profiler->GetElapsedTimeInMicrosec(start_event);
  end_time_ = cuda_profiler->GetElapsedTimeInMicrosec(end_event);
  cuda_profiler->ReleaseCudaEvent(start_event);
  cuda_profiler->ReleaseCudaEvent(end_event);
  start_event = nullptr;
  end_event = nullptr;
  Profiler::Get()->AddNewProfileStat(categories_, name_, start_time_, end_time_, args);
}

CudaProfiler::CudaProfiler() {
  CUDA_CALL(cudaEventCreate(&start_event_));
}

CudaProfiler::~CudaProfiler() {
  CollectCudaStat();
  CUDA_CALL(cudaEventDestroy(start_event_));
  for (auto event : cuda_events_) {
    CUDA_CALL(cudaEventDestroy(event));
  }
}

CudaProfiler* CudaProfiler::Get() {
  static CudaProfiler cuda_profiler;
  return &cuda_profiler;
}

void CudaProfiler::start() {
  if (!started_) {
    CUDA_CALL(cudaEventRecord(start_event_, 0));
    CUDA_CALL(cudaEventSynchronize(start_event_));
    start_time_ = ProfileStat::NowInMicrosec();
    started_ = true;
  }
}

void CudaProfiler::stop() {
  started_ = false;
  start_time_ = 0;
}

cudaEvent_t CudaProfiler::GetCudaEvent() {
  cudaEvent_t event;
  if (cuda_events_.empty()) {
    CUDA_CALL(cudaEventCreate(&event));
  } else {
    event = cuda_events_.back();
    cuda_events_.pop_back();
  }
  return event;
}

void CudaProfiler::ReleaseCudaEvent(cudaEvent_t event) {
  cuda_events_.push_back(event);
}

uint64_t CudaProfiler::GetElapsedTimeInMicrosec(cudaEvent_t event) {
  float elapsedTime;
  CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start_event_, event));
  return start_time_ + static_cast<uint64_t>(elapsedTime * 1000);
}

void CudaProfiler::AddProfilerHelper(std::shared_ptr<CudaProfilerHelper> helper) {
  cuda_helpers_.push_back(helper);
}

void CudaProfiler::CollectCudaStat() {
  for (int i = 0; i < cuda_helpers_.size(); i++) {
    cuda_helpers_[i]->stop();
  }
  cuda_helpers_.clear();
  stop();
}

void CollectCudaProfile() {
  LOG(INFO) << "Collecting cuda profiling";
  CudaProfiler::Get()->CollectCudaStat();
}

MNM_REGISTER_GLOBAL("mnm.profiler.CollectCudaProfile").set_body_typed(CollectCudaProfile);

}  // namespace profiler
}  // namespace mnm
