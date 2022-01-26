/*!
 * Copyright (c) 2022 by Contributors
 * \file op_profiler.h
 * \brief A simple profiler with caching to profile ops using dummy inputs. This is supposed to be
 * used by compilation passes for reference.
 */

#pragma once
#include "mnm/cache.h"
#include "op.h"
#include "op_utils.h"
#include <unordered_map>

#ifdef MNM_USE_CUDA
#include "../../src/common/cuda_utils.h"
#include "../../src/op/dialect/cudnn/cudnn_utils.h"
#include "../../src/op/dialect/cublas/cublas_utils.h"
#endif

namespace mnm {
namespace op_profiler {

using namespace mnm::op;
using namespace mnm::value;

using LatencyMapT = std::unordered_map<std::string, float>;

/*! \brief Abstract base class for a profiler to profile per-op latency during compilation. */
class OpProfiler {
 public:
  virtual ~OpProfiler() {
  }

  /*!
   * \brief Profile one op and return its latency in microseconds.
   * \param op The op to be profiled.
   * \return The latency in microseconds from profiling. If cache hits, the
   * latency is retrieved from the cache and no execution is performed on the
   * device.
   */
  virtual float ProfileOp(const Expr& op);

 protected:
  OpProfiler(const Device& device, int32_t warmup_tripcount, int32_t exec_tripcount)
      : device_(device),
        profile_warmup_tripcount_(warmup_tripcount),
        profile_exec_tripcount_(exec_tripcount) {
  }

  /*! \brief The target device. */
  Device device_;

  /*! \brief A cache to store the latency of profiled ops in microseconds. Cache key is
   * the byte string hash of a call node. */
  LatencyMapT latency_cache_;

  /*! \brief The number of times to warmup the op. */
  int32_t profile_warmup_tripcount_;

  /*! \brief The number of times to actually execute the op. */
  int32_t profile_exec_tripcount_;

 private:
  /*!
   * \brief Generate a byte string hash for the given call node using its op as well as
   * argument and return types.
   *
   * \param call The call node to be hashed.
   * \return The hashed key in a string.
   */
  std::string HashCall(const Call& call) {
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
      key << mnm::ir::AsText(arg->checked_type(), false);
    }
    key << mnm::ir::AsText(call->checked_type(), false);
    return std::string(key.byte_vector.begin(), key.byte_vector.end());
  }

  /*!
   * \brief The function that actually executes the op on the device.
   * \param op_env Pointer to the OpEnv for this op.
   * \param dummy_inputs A vector of dummy inputs for this op.
   * \param dummy_output Space reserved for the dummy output of this op.
   * \return The measured execution time of this op in microseconds.
   */
  virtual float RunOp_(const OpEnvPtr& op_env, const std::vector<value::Value>& dummy_inputs,
                       const Value& dummy_output) = 0;
};

/*! \brief A profiler to profile per-op latency on CPU during compilation. */
class CPUOpProfiler : private OpProfiler {
 public:
  /*! \brief Create a static CPUOpProfiler for the target CPU device and return a pointer to it. */
  static CPUOpProfiler* Make(const Device& device, int32_t warmup_tripcount = 10,
                             int32_t exec_tripcount = 10);

  virtual ~CPUOpProfiler() {
  }

  /*!
   * \brief Profile one op and return its latency in microseconds.
   * \param op The op to be profiled.
   * \return The latency in microseconds from profiling. If cache hits, the
   * latency is retrieved from the cache and no execution is performed on the
   * device.
   */
  virtual float ProfileOp(const Expr& op) {
    return OpProfiler::ProfileOp(op);
  }

 private:
  CPUOpProfiler(const Device& device, int32_t warmup_tripcount, int32_t exec_tripcount)
      : OpProfiler(device, warmup_tripcount, exec_tripcount) {
    CHECK_EQ(device.device_type(), DevType::kCPU()) << "CPUOpProfiler only supports CPU devices!";
  }

  /*!
   * \brief Execute the op on the target CPU device.
   * \param op_env Pointer to the OpEnv for this op.
   * \param dummy_inputs A vector of dummy inputs for this op.
   * \param dummy_output Space reserved for the dummy output of this op.
   * \return The measured execution time of this op in microseconds.
   */
  virtual float RunOp_(const OpEnvPtr& op_env, const std::vector<value::Value>& dummy_inputs,
                       const Value& dummy_output);
};

#ifdef MNM_USE_CUDA
/*! \brief A profiler to profile per-op latency on a CUDA device during compilation. */
class CUDAOpProfiler : private OpProfiler {
 public:
  /*! \brief Create a static CUDAOpProfiler for the target CUDA device and return a pointer to it.
   */
  static CUDAOpProfiler* Make(const Device& device, int32_t warmup_tripcount = 10,
                              int32_t exec_tripcount = 10);

  /*!
   * \brief Profile one op and return its latency in microseconds.
   * \param op The op to be profiled.
   * \return The latency in microseconds from profiling. If cache hits, the
   * latency is retrieved from the cache and no execution is performed on the
   * device.
   */
  virtual float ProfileOp(const Expr& op) {
    return OpProfiler::ProfileOp(op);
  }

  virtual ~CUDAOpProfiler() {
    CUDA_CALL(cudaEventDestroy(start_event_));
    CUDA_CALL(cudaEventDestroy(end_event_));
  }

 private:
  CUDAOpProfiler(const Device& device, int32_t warmup_tripcount, int32_t exec_tripcount)
      : OpProfiler(device, warmup_tripcount, exec_tripcount) {
    CHECK_EQ(device.device_type(), DevType::kCUDA())
        << "CUDAOpProfiler only supports CUDA devices!";
    CUDA_CALL(cudaSetDevice(device.device_id()));
    CUDA_CALL(cudaEventCreate(&start_event_));
    CUDA_CALL(cudaEventCreate(&end_event_));
  }

  /*!
   * \brief Execute the op on the target CPU device.
   * \param op_env Pointer to the OpEnv for this op.
   * \param dummy_inputs A vector of dummy inputs for this op.
   * \param dummy_output Space reserved for the dummy output of this op.
   * \return The measured execution time of this op in microseconds.
   */
  virtual float RunOp_(const OpEnvPtr& op_env, const std::vector<value::Value>& dummy_inputs,
                       const Value& dummy_output);

  /*! \brief CUDA events to time the execution of ops. */
  cudaEvent_t start_event_;
  cudaEvent_t end_event_;
};
#endif

}  // namespace op_profiler
}  // namespace mnm
