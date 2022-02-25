/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/profiler/memory_profiler.cc
 * \brief Memory profiler implementation
 */
#include "raf/registry.h"
#include "raf/memory_profiler.h"
#include "raf/memory_pool.h"

namespace raf {
namespace memory_profiler {

MemoryProfiler::~MemoryProfiler() {
}

MemoryProfiler* MemoryProfiler::Get() {
  static MemoryProfiler prof;
  return &prof;
}

void MemoryProfiler::Record(const Device& device, const std::string& tag) {
  auto pool_size = memory_pool::Memory::GetPoolSize(device);
  auto trace = MemoryTrace(pool_size.first, pool_size.second, tag);
  auto device_str = std::string(device.c_str());
  memory_stats_[device_str].traces.push_back(trace);
  if (trace > memory_stats_[device_str].traces[memory_stats_[device_str].max_trace_idx]) {
    memory_stats_[device_str].max_trace_idx = memory_stats_[device_str].traces.size() - 1;
  } else if (trace.allocated < memory_stats_[device_str].traces.back().allocated) {
    // GC was triggered if the current allocated memory is smaller than the previous one.
    memory_stats_[device_str].num_gc++;
  }
}

void MemoryProfiler::Reset() {
  memory_stats_.clear();
}

Map<String, FloatImm> MemoryProfiler::GetMaxMemoryInfo(const Device& device) {
  float max_used = 0;
  float max_allocated = 0;
  float max_idx = -1;
  int num_gc = 0;
  auto device_str = std::string(device.c_str());

  if (memory_stats_.count(device_str) != 0) {
    max_idx = memory_stats_[device_str].max_trace_idx;
    auto max_trace = memory_stats_[device_str].traces[max_idx];
    max_used = max_trace.used;
    max_allocated = max_trace.allocated;
    num_gc = memory_stats_[device_str].num_gc;
  }

  Map<String, FloatImm> ret;
  ret.Set("max_used", FloatImm(DataType::Float(32), max_used));
  ret.Set("max_allocated", FloatImm(DataType::Float(32), max_allocated));
  ret.Set("max_trace_idx", FloatImm(DataType::Float(32), max_idx));
  ret.Set("num_gc", FloatImm(DataType::Float(32), num_gc));
  return ret;
}

std::string MemoryProfiler::GetMemoryTrace(const Device& device) {
  auto device_str = std::string(device.c_str());
  if (memory_stats_.count(device_str) == 0) {
    return "";
  }

  std::ostringstream os;

  // Display the memory trace of used memory instead of the total allocated memory.
  os << "Numbers are the in MBs." << std::endl;
  os << std::setw(6) << std::left << "#Trace\t" << std::setw(80) << std::left << "#Tag";
  os << "\t" << std::setw(15) << std::left << "used";
  os << "\t" << std::setw(15) << std::left << "alloc";
  os << std::endl;

  for (size_t i = 0; i < memory_stats_[device_str].traces.size(); ++i) {
    auto curr_trace = memory_stats_[device_str].traces[i];
    std::string tag = curr_trace.tag;
    os << std::setw(6) << std::left << i << "\t" << std::setw(80) << std::left << tag;
    os << "\t" << std::setw(15) << std::left << curr_trace.used;
    os << "\t" << std::setw(15) << std::left << curr_trace.allocated;
    os << std::endl;
  }
  return os.str();
}

void EnableMemoryProfiler() {
  MemoryProfiler::Get()->SetProfile(true);
}

void DisableMemoryProfiler() {
  MemoryProfiler::Get()->SetProfile(false);
}

void ResetMemoryProfiler() {
  MemoryProfiler::Get()->Reset();
}

Map<String, FloatImm> GetMaxMemoryInfo(const Device& device) {
  return MemoryProfiler::Get()->GetMaxMemoryInfo(device);
}

std::string GetMemoryTrace(const Device& device) {
  return MemoryProfiler::Get()->GetMemoryTrace(device);
}

RAF_REGISTER_GLOBAL("raf.memory_profiler.EnableMemoryProfiler")
    .set_body_typed(EnableMemoryProfiler);
RAF_REGISTER_GLOBAL("raf.memory_profiler.DisableMemoryeProfiler")
    .set_body_typed(DisableMemoryProfiler);
RAF_REGISTER_GLOBAL("raf.memory_profiler.ResetMemoryProfiler").set_body_typed(ResetMemoryProfiler);
RAF_REGISTER_GLOBAL("raf.memory_profiler.GetMaxMemoryInfo").set_body_typed(GetMaxMemoryInfo);
RAF_REGISTER_GLOBAL("raf.memory_profiler.GetMemoryTrace").set_body_typed(GetMemoryTrace);

}  // namespace memory_profiler
}  // namespace raf
