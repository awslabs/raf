/*!
 * Copyright (c) 2019 by Contributors
 * \file src/common/cuda_profiler.cc
 * \brief async profiler for cuda operations
 */
#include "./cuda_profiler.h"

namespace mnm {
namespace profiler {

void ProfilerCudaHelper::start() {
  if (profiling_) {
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);
    start_time_ = ProfileStat::NowInMicrosec();
  }
}

void ProfilerCudaHelper::stop() {
  if (profiling_) {
    cudaEventSynchronize(end_event);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start_event, end_event);
    end_time_ = start_time_ + uint64_t(elapsedTime * 1000);
    Profiler::Get()->AddNewProfileStat(categories_, name_, start_time_, end_time_, args);
  }
}

CudaProfiler::CudaProfiler() {
}

CudaProfiler::~CudaProfiler() {
  CollectCudaStat();
}

CudaProfiler* CudaProfiler::Get(std::shared_ptr<CudaProfiler>* sp) {
  static std::mutex mtx;
  static std::shared_ptr<CudaProfiler> prof = nullptr;
  if (!prof) {
    std::unique_lock<std::mutex> lk(mtx);
    if (!prof) {
      prof = std::make_shared<CudaProfiler>();
    }
  }
  if (sp) {
    *sp = prof;
  }
  return prof.get();
}

std::vector<ProfilerCudaHelper>& CudaProfiler::GetProfilingHelpers() {
  return cuda_helpers;
}

void CudaProfiler::CollectCudaStat() {
  for (int i = 0; i < cuda_helpers.size(); i++) {
    cuda_helpers[i].stop();
  }
  cuda_helpers.clear();
}

void CollectCudaProfile() {
  LOG(INFO) << "Collecting cuda profiling";
  CudaProfiler::Get()->CollectCudaStat();
}

MNM_REGISTER_GLOBAL("mnm.profiler.CollectCudaProfile").set_body_typed(CollectCudaProfile);

}  // namespace profiler
}  // namespace mnm
