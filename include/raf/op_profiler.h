/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file op_profiler.h
 * \brief A simple profiler with caching to profile ops using dummy inputs. This is supposed to be
 * used by compilation passes for reference.
 */

#pragma once
#include "raf/cache.h"
#include "raf/ir_ext.h"
#include "op.h"
#include "op_utils.h"
#include <unordered_map>

#ifdef RAF_USE_CUDA
#include "../../src/common/cuda_utils.h"
#include "../../src/op/dialect/cudnn/cudnn_utils.h"
#include "../../src/op/dialect/cublas/cublas_utils.h"
#endif

namespace raf {
namespace op_profiler {

using namespace raf::op;
using namespace raf::value;

using LatencyAndWorkspaceMapT =
    std::unordered_map<std::string, std::pair<std::vector<float>, int64_t>>;
using OpEnvMapT = std::unordered_map<std::string, OpEnvPtr>;

/*! \brief A class to JIT op, create dummy input data, and allocate memory buffers for profiling. */
class OpWithData {
 public:
  OpEnvPtr op_env = nullptr;
  int stream_id = -1;
  std::vector<Value> inputs;
  Value output;
  int64_t workspace_size = 0;  // Workspace memory size in bytes.

  OpWithData(const Device device, const Expr& op, const int stream_id = -1);

  ~OpWithData();

  bool profilable() const {
    return op_env != nullptr;
  }
};

using OpWithDataPtr = std::shared_ptr<OpWithData>;

/*! \brief Abstract base class for a profiler to profile per-op latency during compilation. */
class OpProfiler {
 public:
  /*!
   * \brief Dispatch to op profiler according to the given device type.
   * \param device The target device.
   * \return The op profiler pointer.
   */
  static OpProfiler* Get(const Device& device);

  virtual ~OpProfiler() {
  }

  /*!
   * \brief Profile one op and return (1) its latency in microseconds, and (2) the workspace size in
   * bytes.
   * \param op The op to be profiled.
   * \param warmup The number of warmup iterations. Default 10.
   * \param exec_number The number of execution iterations. Default 10.
   * \param repeat The number of repeat iterations. Default 1.
   * \return A tuple containing (1) the latency in microseconds, and
   * (2) workspace size of the op in bytes. If cache hits, the latency and
   * workspace size are retrieved from the cache and no execution is performed on the device.
   */
  std::pair<std::vector<float>, float> ProfileOp(const Expr& op, int32_t warmup = 10,
                                                 int32_t exec_number = 10, int32_t repeat = 1);

  /*!
   * \brief Profile a group of ops and return (1) the total latency in microseconds, and (2) the
   * total workspace size of this group of ops in bytes. If stream_ids are presented, each op will
   * be launched on the corresponding stream, which enables async execution.
   * \param ops The ops to be profiled.
   * \param stream_ids The stream id for each op, or default stream if empty.
   * \param warmup The number of warmup iterations. Default 10.
   * \param exec_number The number of execution iterations. Default 10.
   * \param repeat The number of repeat iterations. Default 1.
   * \return A tuple containing (1) the latency in microseconds, and (2) the sum of workspace sizes
   * of all ops in bytes. If cache hits, the latency and workspace size are retrieved from the cache
   * and no execution is performed on the device.
   */
  std::pair<std::vector<float>, float> ProfileOpGroup(const std::vector<Expr>& ops,
                                                      const std::vector<int>& stream_ids = {},
                                                      int32_t warmup = 10, int32_t exec_number = 10,
                                                      int32_t repeat = 1);

  /*!
   * \brief Return the OpEnv of the given op if it has been profiled.
   * \param op The op to be queried.
   * \return The OpEnv pointer of the given op if it has been profiled; otherwise nullptr.
   */
  OpEnvPtr GetOpEnv(const Expr& op);

  /*!
   * \brief Get the current size of latency cache.
   */
  int GetLatencyCacheSize() {
    return latency_and_workspace_size_cache_.size();
  }

  /*!
   * \brief Reset the latency cache.
   */
  void Reset() {
    latency_and_workspace_size_cache_.clear();
    op_env_cache_.clear();
  }

 protected:
  OpProfiler(const Device& device) : device_(device) {
  }

  /*! \brief The target device. */
  Device device_;
  /*! \brief A cache to store the latency of profiled ops in microseconds. Cache key is
   * the byte string hash of a call node. */
  LatencyAndWorkspaceMapT latency_and_workspace_size_cache_;
  /*! \brief A cache to store built OpEnv. */
  OpEnvMapT op_env_cache_;

 private:
  /*!
   * \brief Generate a byte string hash for the given call node using its op as well as
   * argument and return types.
   *
   * \param call The call node to be hashed.
   * \return The hashed key.
   */
  HashKey HashCall(const Call& call) {
    HashKey key;

    // Hash op name. Note that we directly use the object address as the key
    // because all fused op closures have the same name at this stage.
    if (auto op_node = call->op.as<OpNode>()) {
      key << op_node->name;
    } else if (auto fn_node = call->op.as<FunctionNode>()) {
      key << uint64_t(ObjectPtrHash()(GetRef<Function>(fn_node)));
    } else {
      LOG(FATAL) << "OpProfiler does not deal with " << call->op->GetTypeKey();
      throw;
    }

    // Hash argument and return types.
    for (auto arg : call->args) {
      key << raf::ir::AsText(arg->checked_type(), false);
    }
    key << raf::ir::AsText(call->checked_type(), false);
    return key;
  }

  /*!
   * \brief Generate a byte string hash for the given group node using their op, arguments,
   * return types and stream ids.
   *
   * \param ops The group to be hashed.
   * \param stream_ids The stream IDs.
   * \return The hashed key.
   */
  HashKey HashGroup(const std::vector<Expr>& ops, const std::vector<int> stream_ids = {}) {
    HashKey key;
    std::vector<int> processed_stream_ids = stream_ids;

    if (processed_stream_ids.empty()) {
      for (auto op : ops) {
        processed_stream_ids.push_back(-1);
      }
    }
    CHECK_EQ(processed_stream_ids.size(), ops.size())
        << "The length of stream_ids does not match ops";

    for (size_t i = 0; i < ops.size(); ++i) {
      auto op = ops[i];
      if (auto call_node = op.as<CallNode>()) {
        key << HashCall(GetRef<Call>(call_node));
      } else {
        // For non-call nodes, we simply hash their type.
        key << raf::ir::AsText(op->checked_type(), false);
      }
      key << processed_stream_ids[i];
    }
    return key;
  }

  inline std::string HashKeyToStr(const HashKey& key) {
    return std::string(key.byte_vector.begin(), key.byte_vector.end());
  }

  /*!
   * \brief The function that actually executes the op on the device.
   * \param op_with_data The executable op with data.
   * \param warmup The number of warmup iterations. Default 10.
   * \param exec_number The number of execution iterations. Default 10.
   * \param repeat The number of repeat iterations. Default 1.
   * \return The measured execution time of this op in microseconds.
   */
  virtual std::vector<float> RunOp(const OpWithDataPtr& op_with_data, int32_t warmup = 10,
                                   int32_t exec_number = 10, int32_t repeat = 1) = 0;

  /*!
   * \brief The function that actually executes a op group on the device.
   * \param op_with_datas The executable op with data.
   * \param warmup The number of warmup iterations. Default 10.
   * \param exec_number The number of execution iterations. Default 10.
   * \param repeat The number of repeat iterations. Default 1.
   * \return The measured execution time of executing an entire group in microseconds.
   */
  virtual std::vector<float> RunOpGroup(const std::vector<OpWithDataPtr>& op_with_datas,
                                        int32_t warmup = 10, int32_t exec_number = 10,
                                        int32_t repeat = 1) = 0;
};

/*! \brief A profiler to profile per-op latency on CPU during compilation. */
class CPUOpProfiler : public OpProfiler {
 public:
  CPUOpProfiler(const Device& device) : OpProfiler(device) {
    CHECK_EQ(device.device_type(), DevType::kCPU()) << "CPUOpProfiler only supports CPU devices!";
  }

  virtual ~CPUOpProfiler() {
  }

 private:
  /*!
   * \brief The function that actually executes the op on the device.
   * \param op_with_data The executable op with data.
   * \param warmup The number of warmup iterations. Default 10.
   * \param exec_number The number of execution iterations. Default 10.
   * \param repeat The number of repeat iterations. Default 1.
   * \return The measured execution time of this op in microseconds.
   */
  virtual std::vector<float> RunOp(const OpWithDataPtr& op_with_data, int32_t warmup = 10,
                                   int32_t exec_number = 10, int32_t repeat = 1);

  /*!
   * \brief The function that actually executes a op group on the device.
   * \param op_with_datas The executable op with data.
   * \param warmup The number of warmup iterations. Default 10.
   * \param exec_number The number of execution iterations. Default 10.
   * \param repeat The number of repeat iterations. Default 1.
   * \return The measured execution time of executing an entire group in microseconds.
   */
  virtual std::vector<float> RunOpGroup(const std::vector<OpWithDataPtr>& op_with_datas,
                                        int32_t warmup = 10, int32_t exec_number = 10,
                                        int32_t repeat = 1);
};

#ifdef RAF_USE_CUDA
/*! \brief A profiler to profile per-op latency on a CUDA device during compilation. */
class CUDAOpProfiler : public OpProfiler {
 public:
  CUDAOpProfiler(const Device& device) : OpProfiler(device) {
    CHECK_EQ(device.device_type(), DevType::kCUDA())
        << "CUDAOpProfiler only supports CUDA devices!";
    cuda_api_ = tvm::runtime::DeviceAPI::Get(device_);
    CUDA_CALL(cudaSetDevice(device.device_id()));
    CUDA_CALL(cudaEventCreate(&start_event_));
    CUDA_CALL(cudaEventCreate(&end_event_));
  }

  virtual ~CUDAOpProfiler() {
    CUDA_CALL(cudaEventDestroy(start_event_));
    CUDA_CALL(cudaEventDestroy(end_event_));
    for (auto id_n_stream : streams_) {
      if (id_n_stream.second != nullptr) {
        CUDA_CALL(cudaStreamDestroy(id_n_stream.second));
      }
    }
  }

 private:
  /*!
   * \brief The function that actually executes the op on the device.
   * \param op_with_data The executable op with data.
   * \param warmup The number of warmup iterations. Default 10.
   * \param exec_number The number of execution iterations. Default 10.
   * \param repeat The number of repeat iterations. Default 1.
   * \return The measured execution time of this op in microseconds.
   */
  virtual std::vector<float> RunOp(const OpWithDataPtr& op_with_data, int32_t warmup = 10,
                                   int32_t exec_number = 10, int32_t repeat = 1);

  /*!
   * \brief The function that actually executes a op group on the device.
   * \param op_with_datas The executable op with data.
   * \param warmup The number of warmup iterations. Default 10.
   * \param exec_number The number of execution iterations. Default 10.
   * \param repeat The number of repeat iterations. Default 1.
   * \return The measured execution time of executing an entire group in microseconds.
   */
  virtual std::vector<float> RunOpGroup(const std::vector<OpWithDataPtr>& op_with_datas,
                                        int32_t warmup = 10, int32_t exec_number = 10,
                                        int32_t repeat = 1);

  /*! \brief CUDA events to time the execution of ops. */
  cudaEvent_t start_event_;
  cudaEvent_t end_event_;
  /*! \brief Stream ID to CUDA stream. */
  std::unordered_map<int, cudaStream_t> streams_;
  /*! \brief CUDA device API. */
  tvm::runtime::DeviceAPI* cuda_api_;
};
#endif

}  // namespace op_profiler
}  // namespace raf
