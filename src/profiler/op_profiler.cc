/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/profiler/op_profiler.cc
 * \brief A simple profiler with caching to profile ops during compilation
 */

#include "raf/op_profiler.h"
#include "raf/ir_ext.h"
#include "../op/dialect/tvm/tvm_utils.h"
#include "../requests.h"
#include <chrono>

namespace raf {
namespace op_profiler {

using namespace raf::op;
using namespace raf::value;

OpProfiler* OpProfiler::Get(const Device& device) {
  CHECK_EQ(device.device_id(), 0) << "Multi-device profiling is not supported yet";
  if (device.device_type() == DevType::kCPU()) {
    static CPUOpProfiler profiler = CPUOpProfiler(device);
    return &profiler;
  } else if (device.device_type() == DevType::kCUDA()) {
#ifdef RAF_USE_CUDA
    static CUDAOpProfiler profiler = CUDAOpProfiler(device);
    return &profiler;
#else
    LOG(FATAL) << "CUDA is not enabled";
#endif
  }
}

OpWithData::OpWithData(const Device device, const Expr& op, const int stream_id)
    : stream_id(stream_id) {
  // Nothing to do with non-call nodes.
  if (!op->IsInstance<CallNode>()) {
    return;
  }

  // JIT the op.
  auto call = Downcast<Call>(op);
  CallValues call_values = CreateDummyCallValues(call, device);
  op_env = Dispatch(call_values);

  // Create the dummy inputs and outputs.
  output = CreateDummyValueFromType(op->checked_type(), device);
  std::vector<Value> temp_inputs;
  for (auto arg : call->args) {
    if (auto const_node = arg.as<RelayConstantNode>()) {
      const auto casted_const_node = static_cast<const ConstantNode*>(const_node);
      CHECK_NOTNULL(casted_const_node);
      temp_inputs.push_back(Downcast<Value>(casted_const_node->value));
    } else {
      temp_inputs.push_back(CreateDummyValueFromType(arg->checked_type(), device));
    }
  }
  for (int k : op_env->arg_indices) {
    inputs.push_back(temp_inputs[k]);
  }

  // Allocate the workspace.
  workspace_size = 0;
  std::shared_ptr<requests::Requests> requests = op_env->GetRequests();
  for (size_t i = 0; i < requests->workspace.size(); i++) {
    requests::Requests::WorkspaceRequest& entry = requests->workspace[i];
    auto buf = memory_pool::Memory::Alloc(entry.device, entry.nbytes);
    entry.memory = buf;
    *entry.dest = buf->data;
    workspace_size += entry.nbytes;
  }
}

OpWithData::~OpWithData() {
  if (op_env == nullptr) {
    return;
  }

  // Free the workspace.
  std::shared_ptr<requests::Requests> requests = op_env->GetRequests();
  for (size_t i = 0; i < requests->workspace.size(); ++i) {
    requests::Requests::WorkspaceRequest& entry = requests->workspace[i];
    if (entry.nbytes > 0 && entry.memory != nullptr) {
      *entry.dest = nullptr;
      entry.memory.reset();
    }
  }

  // Free the input and output buffers.
  try {
    inputs.clear();
  } catch (dmlc::Error& e) {
    return;
  }
}

OpEnvPtr OpProfiler::GetOpEnv(const Expr& op) {
  if (auto call_node = op.as<CallNode>()) {
    auto call = GetRef<Call>(call_node);
    auto call_hash_key = HashKeyToStr(HashCall(call));
    if (op_env_cache_.count(call_hash_key)) {
      return op_env_cache_[call_hash_key];
    }
  }
  return nullptr;
}

std::pair<std::vector<float>, float> OpProfiler::ProfileOpGroup(const std::vector<Expr>& ops,
                                                                const std::vector<int>& stream_ids,
                                                                int32_t warmup, int32_t exec_number,
                                                                int32_t repeat) {
  // Check cache and skip profiling if hit.
  auto key = HashKeyToStr(HashGroup(ops, stream_ids) << warmup << exec_number << repeat);

  // Directly return the profiled latency if cache hit.
  if (latency_and_workspace_size_cache_.count(key) > 0) {
    return latency_and_workspace_size_cache_[key];
  }

  // Prepare ops for profiling.
  std::vector<OpWithDataPtr> ops_with_data;
  int64_t total_workspace_size = 0;
  for (size_t i = 0; i < ops.size(); ++i) {
    auto op = ops[i];
    auto stream_id = stream_ids.empty() ? -1 : stream_ids[i];
    auto single_op_with_data = std::make_shared<OpWithData>(device_, op, stream_id);
    ops_with_data.push_back(single_op_with_data);
    // Currently using the sum of the workspace sizes of all ops as workspace size
    // Might need refactoring
    total_workspace_size += single_op_with_data->workspace_size;
  }

  // Profiling.
  std::vector<float> cost = RunOpGroup(ops_with_data, warmup, exec_number, repeat);

  // Add the result to the cache.
  latency_and_workspace_size_cache_[key] =
      std::move(std::make_pair(std::move(cost), total_workspace_size));
  return latency_and_workspace_size_cache_[key];
}

// Profile one op and return its latency in microseconds on the target device.
std::pair<std::vector<float>, float> OpProfiler::ProfileOp(const Expr& op, int32_t warmup,
                                                           int32_t exec_number, int32_t repeat) {
  if (auto call_node = op.as<CallNode>()) {
    auto call = GetRef<Call>(call_node);
    auto call_hash_key = HashCall(call);
    auto call_key_str = HashKeyToStr(call_hash_key);
    auto key = HashKeyToStr(call_hash_key << warmup << exec_number << repeat);

    // Directly return the profiled latency if cache hit.
    if (latency_and_workspace_size_cache_.count(key) > 0) {
      return latency_and_workspace_size_cache_[key];
    }

    // Build the op and generate dummy input data for profiling.
    OpWithDataPtr op_with_data = std::make_shared<OpWithData>(device_, op);

    // Only catch the built OpEnv for other use cases. Note that op_with_data cannot be cached
    // and must be released in this function; otherwise we may encounter OOM during profiling.
    op_env_cache_[call_key_str] = op_with_data->op_env;

    // Profile the op.
    std::vector<float> cost = RunOp(op_with_data, warmup, exec_number, repeat);
    int64_t workspace_size = op_with_data->workspace_size;

    // Add the profiled cost to the cache.
    latency_and_workspace_size_cache_[key] =
        std::move(std::make_pair(std::move(cost), workspace_size));
    return latency_and_workspace_size_cache_[key];
  }

  // Non-call ops (e.g., tuple) do not invoke compute kernel, have zero cost, and do not have
  // workspace size
  return std::make_pair(std::vector<float>(repeat, 0.0), 0.0f);
}

std::vector<float> CPUOpProfiler::RunOp(const OpWithDataPtr& op_with_data, int32_t warmup,
                                        int32_t exec_number, int32_t repeat) {
  if (!op_with_data->profilable()) {
    return std::vector<float>(repeat, 0.0);
  }

  // Warm up first
  for (int i = 0; i < warmup; i++) {
    op_with_data->op_env->Execute(op_with_data->inputs, op_with_data->output);
  }

  std::vector<float> elapsed_times;
  std::chrono::time_point<std::chrono::system_clock> m_starttime;
  std::chrono::time_point<std::chrono::system_clock> m_endtime;
  for (size_t r = 0; r < repeat; ++r) {
    m_starttime = std::chrono::system_clock::now();
    // Set up timer and run
    for (int i = 0; i < exec_number; i++) {
      op_with_data->op_env->Execute(op_with_data->inputs, op_with_data->output);
    }
    m_endtime = std::chrono::system_clock::now();
    float elapsed_time =
        std::chrono::duration_cast<std::chrono::microseconds>(m_endtime - m_starttime).count();
    elapsed_times.push_back(elapsed_time / exec_number);
  }
  return elapsed_times;
}

std::vector<float> CPUOpProfiler::RunOpGroup(const std::vector<OpWithDataPtr>& op_with_datas,
                                             int32_t warmup, int32_t exec_number, int32_t repeat) {
  // CPU does not support multi-stream, so just run ops one-by-one.

  // Warm up
  for (int i = 0; i < warmup; i++) {
    for (auto op_with_data : op_with_datas) {
      op_with_data->op_env->Execute(op_with_data->inputs, op_with_data->output);
    }
  }

  // Profiling
  std::vector<float> elapsed_times;
  std::chrono::time_point<std::chrono::system_clock> m_starttime;
  std::chrono::time_point<std::chrono::system_clock> m_endtime;
  for (size_t r = 0; r < repeat; ++r) {
    m_starttime = std::chrono::system_clock::now();
    // Set up timer and run
    for (int i = 0; i < exec_number; i++) {
      for (auto op_with_data : op_with_datas) {
        op_with_data->op_env->Execute(op_with_data->inputs, op_with_data->output);
      }
    }
    m_endtime = std::chrono::system_clock::now();
    float elapsed_time =
        std::chrono::duration_cast<std::chrono::microseconds>(m_endtime - m_starttime).count();
    elapsed_times.push_back(elapsed_time / exec_number);
  }
  return elapsed_times;
}

#ifdef RAF_USE_CUDA
// Run the op on the CUDA device, return the profiled execution time in microseconds
std::vector<float> CUDAOpProfiler::RunOp(const OpWithDataPtr& op_with_data, int32_t warmup,
                                         int32_t exec_number, int32_t repeat) {
  if (!op_with_data->profilable()) {
    return std::vector<float>(repeat, 0.0);
  }
  CUDA_CALL(cudaDeviceSynchronize());

  // Warm up first
  for (int i = 0; i < warmup; i++) {
    op_with_data->op_env->Execute(op_with_data->inputs, op_with_data->output);
  }

  // Set up timer and run
  std::vector<float> elapsed_times;
  for (size_t r = 0; r < repeat; ++r) {
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaEventRecord(start_event_, nullptr));
    for (int i = 0; i < exec_number; ++i) {
      op_with_data->op_env->Execute(op_with_data->inputs, op_with_data->output);
    }
    CUDA_CALL(cudaEventRecord(end_event_, nullptr));
    CUDA_CALL(cudaDeviceSynchronize());
    float elapsed_time;
    CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start_event_, end_event_));
    // Convert to microseconds
    elapsed_times.push_back(1000.0 * elapsed_time / exec_number);
  }
  return elapsed_times;
}

std::vector<float> CUDAOpProfiler::RunOpGroup(const std::vector<OpWithDataPtr>& op_with_datas,
                                              int32_t warmup, int32_t exec_number, int32_t repeat) {
  std::vector<float> elapsed_times;

  // Create streams.
  streams_[-1] = nullptr;
  for (auto op_with_data : op_with_datas) {
    if (op_with_data->stream_id == -1) {
      continue;
    }
    if (streams_.count(op_with_data->stream_id) == 0) {
      streams_[op_with_data->stream_id];
      CUDA_CALL(cudaStreamCreate(&streams_[op_with_data->stream_id]));
    }
  }
  CUDA_CALL(cudaDeviceSynchronize());

  // Warm up first
  for (int i = 0; i < warmup; i++) {
    for (auto op_with_data : op_with_datas) {
      if (!op_with_data->profilable()) {
        continue;
      }

      // Set stream.
      auto curr_stream = streams_[op_with_data->stream_id];
      cuda_api_->SetStream(device_, curr_stream);
      raf::op::cudnn::SetStream(curr_stream);
      raf::op::cublas::SetStream(curr_stream);

      // Issue kernel.
      op_with_data->op_env->Execute(op_with_data->inputs, op_with_data->output);
    }
  }

  // Profiling.
  for (int i = 0; i < repeat; ++i) {
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaEventRecord(start_event_, nullptr));
    for (int j = 0; j < exec_number; ++j) {
      for (auto op_with_data : op_with_datas) {
        if (!op_with_data->profilable()) {
          continue;
        }

        // Set stream.
        auto curr_stream = streams_[op_with_data->stream_id];
        cuda_api_->SetStream(device_, curr_stream);
        raf::op::cudnn::SetStream(curr_stream);
        raf::op::cublas::SetStream(curr_stream);

        // Issue kernel.
        op_with_data->op_env->Execute(op_with_data->inputs, op_with_data->output);
      }
    }
    CUDA_CALL(cudaEventRecord(end_event_, nullptr));
    CUDA_CALL(cudaDeviceSynchronize());
    float elapsed_time;
    CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start_event_, end_event_));
    // Convert to microseconds
    elapsed_times.push_back(1000.0 * elapsed_time / exec_number);
  }

  cuda_api_->SetStream(device_, nullptr);
  raf::op::cudnn::SetStream(nullptr);
  raf::op::cublas::SetStream(nullptr);
  return elapsed_times;
}
#endif

RAF_REGISTER_GLOBAL("raf.op_profiler.Profile")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* ret) {
      CHECK_GE(args.size(), 2U) << "Expected (expr, device, <warmup>, <exec>, <repeat>)";
      Expr expr = args[0];
      Device device = args[1];
      int warmup = (args.size() >= 3) ? args[2] : 10;
      int exec_number = (args.size() >= 4) ? args[3] : 10;
      int repeat = (args.size() == 5) ? args[4] : 1;
      auto profiler = OpProfiler::Get(device);

      Map<String, ObjectRef> results;
      Array<FloatImm> lat_results;
      auto lat_and_workspace_size = profiler->ProfileOp(expr, warmup, exec_number, repeat);
      for (float lat : lat_and_workspace_size.first)
        lat_results.push_back(FloatImm(DataType::Float(32), lat));
      results.Set("latency", lat_results);
      results.Set("workspace_size", FloatImm(DataType::Float(32), lat_and_workspace_size.second));
      *ret = results;
    });

RAF_REGISTER_GLOBAL("raf.op_profiler.ProfileGroup")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* ret) {
      CHECK_GE(args.size(), 2U)
          << "Expected (exprs, device, <stream_ids>, <warmup>, <exec>, <repeat>)";
      Array<Expr> exprs = args[0];
      Device device = args[1];
      Array<Integer> stream_ids_container = (args.size() >= 3) ? args[2] : Array<Integer>();
      std::vector<int> stream_ids;
      for (auto stream_id : stream_ids_container) {
        stream_ids.push_back(stream_id.IntValue());
      }

      int warmup = (args.size() >= 4) ? args[3] : 10;
      int exec_number = (args.size() >= 5) ? args[4] : 10;
      int repeat = (args.size() == 6) ? args[5] : 1;
      auto profiler = OpProfiler::Get(device);

      Map<String, ObjectRef> results;
      Array<FloatImm> lat_results;
      auto lat_and_workspace_size = profiler->ProfileOpGroup(
          std::vector<Expr>(exprs.begin(), exprs.end()), stream_ids, warmup, exec_number, repeat);
      for (auto lat : lat_and_workspace_size.first) {
        lat_results.push_back(FloatImm(DataType::Float(32), lat));
      }
      results.Set("latency", lat_results);
      results.Set("workspace_size", FloatImm(DataType::Float(32), lat_and_workspace_size.second));
      *ret = results;
    });

RAF_REGISTER_GLOBAL("raf.op_profiler.ResetCache").set_body_typed([](const Device& device) {
  auto profiler = OpProfiler::Get(device);
  return profiler->Reset();
});

RAF_REGISTER_GLOBAL("raf.op_profiler.GetCacheSize").set_body_typed([](const Device& device) {
  auto profiler = OpProfiler::Get(device);
  return profiler->GetLatencyCacheSize();
});

}  // namespace op_profiler
}  // namespace raf
