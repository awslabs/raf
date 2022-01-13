/*!
 * Copyright (c) 2022 by Contributors
 * \file src/profiler/op_profiler.cc
 * \brief A simple profiler with caching to profile ops during compilation
 */

#include "mnm/op_profiler.h"
#include "mnm/ir.h"
#include "../op/dialect/tvm/tvm_utils.h"
#include <chrono>

namespace mnm {
namespace op_profiler {

using namespace mnm::op;
using namespace mnm::value;

// Profile one op and return its latency in microseconds on the target device.
float OpProfiler::ProfileOp(const Expr& op) {
  if (auto call_node = op.as<CallNode>()) {
    auto call = GetRef<Call>(call_node);

    auto key = HashCall(call);

    // Directly return the profiled latency if cache hit.
    if (latency_cache_.count(key) > 0) {
      return latency_cache_[key];
    }

    // JIT the op.
    CallValues call_values = CreateDummyCallValues(call, device_);
    OpEnvPtr op_env = Dispatch(call_values);

    // Create dummy input and dummy output for profiling.
    std::vector<Value> dummy_inputs;
    for (auto arg : call_node->args) {
      if (auto const_node = arg.as<RelayConstantNode>()) {
        const auto casted_const_node = static_cast<const ConstantNode*>(const_node);
        CHECK_NOTNULL(casted_const_node);
        dummy_inputs.push_back(Downcast<Value>(casted_const_node->value));
      } else {
        dummy_inputs.push_back(CreateDummyValueFromType(arg->checked_type(), device_));
      }
    }
    auto dummy_output = CreateDummyValueFromType(op->checked_type(), device_);
    std::vector<value::Value> actual_dummy_inputs;
    for (int k : op_env->arg_indices) {
      actual_dummy_inputs.push_back(dummy_inputs[k]);
    }

    // Profile the op
    float cost = RunOp_(op_env, actual_dummy_inputs, dummy_output);

    // Add the profiled cost to the cache.
    latency_cache_[key] = cost;
    return cost;
  }

  // Non-call ops (e.g., tuple) do not invoke compute kernel and have zero cost.
  return 0.0f;
}

// Create a static CPUOpProfiler for the target CPU device and return a pointer to it.
CPUOpProfiler* CPUOpProfiler::Make(const Device& device, int32_t warmup_tripcount,
                                   int32_t exec_tripcount) {
  LOG(FATAL) << "CPUOpProfiler is currently under development and testing!";
  static CPUOpProfiler cpu_profiler = CPUOpProfiler(device, warmup_tripcount, exec_tripcount);
  return &cpu_profiler;
}

// Run the op on the CPU device, return the profiled execution time in microseconds
float CPUOpProfiler::RunOp_(const OpEnvPtr& op_env, const std::vector<value::Value>& dummy_inputs,
                            const Value& dummy_output) {
  // Warm up first
  for (int i = 0; i < profile_warmup_tripcount_; i++) {
    op_env->Execute(dummy_inputs, dummy_output);
  }

  std::chrono::time_point<std::chrono::system_clock> m_starttime;
  std::chrono::time_point<std::chrono::system_clock> m_endtime;
  m_starttime = std::chrono::system_clock::now();
  // Set up timer and run
  for (int i = 0; i < profile_exec_tripcount_; i++) {
    op_env->Execute(dummy_inputs, dummy_output);
  }
  m_endtime = std::chrono::system_clock::now();
  float elapsed_time =
      std::chrono::duration_cast<std::chrono::microseconds>(m_endtime - m_starttime).count();
  // Convert to microseconds
  return elapsed_time / profile_exec_tripcount_ * 1000.0f;
}

#ifdef MNM_USE_CUDA
// Create a static CPUOpProfiler for the target CUDA device and return a pointer to it.
CUDAOpProfiler* CUDAOpProfiler::Make(const Device& device, int32_t warmup_tripcount,
                                     int32_t exec_tripcount) {
  static CUDAOpProfiler cuda_profiler = CUDAOpProfiler(device, warmup_tripcount, exec_tripcount);
  return &cuda_profiler;
}

// Run the op on the CUDA device, return the profiled execution time in microseconds
float CUDAOpProfiler::RunOp_(const OpEnvPtr& op_env, const std::vector<value::Value>& dummy_inputs,
                             const Value& dummy_output) {
  CUDA_CALL(cudaDeviceSynchronize());

  // Warm up first
  for (int i = 0; i < profile_warmup_tripcount_; i++) {
    op_env->Execute(dummy_inputs, dummy_output);
  }

  // Set up timer and run
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaEventRecord(start_event_, nullptr));
  for (int i = 0; i < profile_exec_tripcount_; i++) {
    op_env->Execute(dummy_inputs, dummy_output);
  }
  CUDA_CALL(cudaEventRecord(end_event_, nullptr));
  CUDA_CALL(cudaDeviceSynchronize());
  float elapsed_time;
  CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start_event_, end_event_));
  // Convert to microseconds
  return elapsed_time / profile_exec_tripcount_ * 1000.0f;
}
#endif
}  // namespace op_profiler
}  // namespace mnm
