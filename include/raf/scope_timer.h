/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file scope_timer.h
 * \brief Scope timer that times the execution time of a code scope (e.g., function).
 */
#pragma once
#include "raf/profiler.h"

namespace raf {
namespace scope_timer {

class ScopeTimerPool {
 public:
  explicit ScopeTimerPool() {
  }

  ~ScopeTimerPool() {
  }

  static std::shared_ptr<ScopeTimerPool> Get() {
    static registry::PerDeviceStore<ScopeTimerPool, false>* pool =
        new registry::PerDeviceStore<ScopeTimerPool, false>();
    std::shared_ptr<ScopeTimerPool>& ret = pool->Get(Device(DevType::kCPU(), 0));
    if (ret == nullptr) {
      std::lock_guard<std::mutex> lock(pool->mutex_);
      if (ret == nullptr) {
        ret = std::make_shared<ScopeTimerPool>();
      }
    }
    return ret;
  }

  void AddSample(std::string name, float timed) {
    std::lock_guard<std::mutex> lock(mutex);
    time_pool[name].push_back(timed);
  }

  void DumpReport() {
    LOG(INFO) << "Scope Timer Report:";
    for (auto& kv : time_pool) {
      float total = 0;
      float max_sample = 0;
      for (auto& t : kv.second) {
        total += t;
        max_sample = std::max(max_sample, t);
      }
      float avg = total / kv.second.size();
      LOG(INFO) << kv.first << ": " << total << "s (max " << max_sample << "s, avg " << avg
                << "s) from " << kv.second.size() << " samples";
    }
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(mutex);
    time_pool.clear();
  }

  std::vector<float> GetSamplesByName(std::string name) {
    return time_pool[name];
  }

 public:
  std::unordered_map<std::string, std::vector<float>> time_pool;
  std::mutex mutex;
};

/*!
 * \brief A helper class and macro to profile the execution time of a scope (e.g., function).
 * This is used for debugging/profiling purpose. For example:
 * void some_func() {
 *   RAF_TIMED_SEC("some_func")
 *   // do something;
 * }
 * The profiled time is then the life time of the created TimeSection object, and will be
 * recorded in the global scope_timer_pool and could be dumped later.
 */
class TimedSection {
 public:
  explicit TimedSection(std::string name, bool flush, std::shared_ptr<ScopeTimerPool> pool)
      : name_(name), flush_(flush), pool_(pool), start_(profiler::ProfileStat::NowInMicrosec()) {
  }

  ~TimedSection() {
    auto now = profiler::ProfileStat::NowInMicrosec();
    float timed = (now - start_) / 1e6;
    pool_->AddSample(name_, timed);
    if (flush_) {
      LOG(INFO) << "Timed " << name_ << ": " << timed << "s";
    }
  }

 private:
  std::string name_;
  bool flush_;
  std::shared_ptr<ScopeTimerPool> pool_;
  uint64_t start_;
};
#define RAF_TIMED(name, flush)                                     \
  auto scope_timer_pool = raf::scope_timer::ScopeTimerPool::Get(); \
  raf::scope_timer::TimedSection timed_section(name, flush, scope_timer_pool);

}  // namespace scope_timer
}  // namespace raf
