/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/profiler/cuda/cuda_profiler.cc
 * \brief async profiler for cuda operations
 */
#include "./cuda_profiler.h"
#include "../../common/cuda_utils.h"

namespace raf {
namespace profiler {

std::shared_ptr<device_api::DeviceAPI> CudaProfilerHelper::cuda_api =
    device_api::DeviceAPI::Get(DevType::kCUDA());

CudaProfilerHelper::CudaProfilerHelper(int dev_id, raf::DevType dev_type, void* stream,
                                       std::string name, std::string categories,
                                       std::vector<std::string> args)
    : ProfilerHelper(dev_id, dev_type, std::move(name), std::move(categories), std::move(args)),
      stream(stream) {
  Device device(dev_type, dev_id);
  start_event = event_pool::EventPool::Get(device)->GetEvent();
  end_event = event_pool::EventPool::Get(device)->GetEvent();
}

void CudaProfilerHelper::collect() {
  cuda_api->WaitEvent(end_event->data());
  auto cuda_profiler = CudaProfiler::Get();
  start_time_ = cuda_profiler->GetElapsedTimeInMicrosec(start_event);
  end_time_ = cuda_profiler->GetElapsedTimeInMicrosec(end_event);
  Profiler::Get()->AddNewProfileStat(categories_, name_, start_time_, end_time_, args);
}

CudaProfiler::CudaProfiler() {
  // Each cuda event is associated with a cuda context (each device has a default cuda context when
  // we use the cuda runtime api). Thus, if we want to support multi-device profiling on a single
  // machine, we need a parameter to specify the cuda device id. But for now, we only support the
  // profiling on the first cuda device.
  int device_id = 0;
  auto api = device_api::DeviceAPI::Get(DevType::kCUDA());
  if (api->GetDeviceCount() > 1) {
    LOG(WARNING) << "Multi-GPU detected, current CUDA profiler only supports single GPU profiling.";
  }
  start_event_ = event_pool::EventPool::Get(Device(DevType::kCUDA(), device_id))->GetEvent();
}

CudaProfiler::~CudaProfiler() {
}

CudaProfiler* CudaProfiler::Get() {
  static CudaProfiler cuda_profiler;
  return &cuda_profiler;
}

void CudaProfiler::start() {
  static auto api = device_api::DeviceAPI::Get(DevType::kCUDA());
  if (!started_) {
    api->WaitDevice(Device(DevType::kCUDA(), 0));
    api->EventRecordOnStream(start_event_->data(), nullptr);
    start_time_ = ProfileStat::NowInMicrosec();
    api->WaitEvent(start_event_->data());
    started_ = true;
  }
}

void CudaProfiler::stop() {
  started_ = false;
  start_time_ = 0;
}

uint64_t CudaProfiler::GetElapsedTimeInMicrosec(std::shared_ptr<Event> event) {
  static auto api = device_api::DeviceAPI::Get(DevType::kCUDA());
  float elapsedTime = api->EventElapsedTimeInMilliSeconds(start_event_->data(), event->data());
  return start_time_ + static_cast<uint64_t>(elapsedTime * 1000);
}

void CudaProfiler::CollectCudaStat() {
  for (int i = 0; i < cuda_helpers_.size(); i++) {
    cuda_helpers_[i].collect();
  }
  stop();
}

void CudaProfiler::ClearCudaStat() {
  cuda_helpers_.clear();
}

void CollectCudaProfile() {
  CudaProfiler::Get()->CollectCudaStat();
}

void ClearCudaProfile() {
  CudaProfiler::Get()->ClearCudaStat();
}

RAF_REGISTER_GLOBAL("raf.profiler.CollectCudaProfile").set_body_typed(CollectCudaProfile);
RAF_REGISTER_GLOBAL("raf.profiler.ClearCudaProfile").set_body_typed(ClearCudaProfile);

}  // namespace profiler
}  // namespace raf
