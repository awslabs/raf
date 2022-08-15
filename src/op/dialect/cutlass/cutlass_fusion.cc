/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/cutlass/cutlass_fusion.cc
 * \brief Implementation of cutlass dispatch for fused functions
 */
#include <limits>

#include "raf/cache.h"
#include "raf/value.h"
#include "raf/profiler.h"
#include "raf/registry.h"
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/pass.h"
#include "tvm/ir/type_functor.h"
#include "tvm/relay/dataflow_pattern.h"
#include "./timer.h"
#include "./gemm.h"
#include "./conv.h"

namespace raf {
namespace op {
namespace cutlass {

using namespace raf::ir;
using namespace raf::value;
using raf::registry::TypedPackedFunc;

/*! \brief The persist cache entry of the best CUTLASS tuned config. */
class CUTLASSConfigCacheEntry {
 public:
  explicit CUTLASSConfigCacheEntry() {
  }

  CUTLASSConfigCacheEntry(std::shared_ptr<TunableConfig> config) : config_(config) {
  }

  std::shared_ptr<TunableConfig> GetConfig() {
    return config_;
  }

  static CUTLASSConfigCacheEntry Load(const std::string path) {
    // Not support deserialization yet.
    throw;
  }

  bool Save(const std::string& path) {
    // Not support serialization yet.
    return false;
  }

 private:
  /*! \brief The tunable config. */
  std::shared_ptr<TunableConfig> config_;
};

MetaPersistCache<CUTLASSConfigCacheEntry> CacheConfig("cutlass_fusion_config");

HashKey HashFusedFunc(const Function& func) {
  HashKey key;
  key << raf::ir::AsText(func, true);
  return key;
}

OpEnv* Tune(const op::CallValues& call, OpEnv* op_env) {
  CutlassOpEnv* env = static_cast<CutlassOpEnv*>(op_env);
  auto key = HashFusedFunc(Downcast<ClosureValue>(call->callee)->func);
  std::shared_ptr<TunableConfig> best;

  if (const auto* compiled = CacheConfig.Get(key.byte_vector)) {
    CUTLASSConfigCacheEntry entry = *compiled;
    best = entry.GetConfig();
  } else {
    std::vector<std::shared_ptr<TunableConfig>> tunable = env->ListTunableConfigs();
    const int number = 10, repeat = 1, min_repeat_ms = 0, cooldown_interval_ms = 0,
              repeats_to_cooldown = 1, limit_zero_time_iterations = 100;
    double min_time = std::numeric_limits<double>::max();
    for (auto& config : tunable) {
      env->SetTunableConfig(config);
      env->Init(call);
      Array<FloatValue> result = TimeEvaluator(
          TypedPackedFunc<void()>([&]() { env->Execute(call); }), call->device, number, repeat,
          min_repeat_ms, limit_zero_time_iterations, cooldown_interval_ms, repeats_to_cooldown)();
      CHECK_EQ(result.size(), 1U);
      if (result[0]->value < min_time) {
        min_time = result[0]->value;
        best = config;
      }
    }
    CacheConfig.Set(key.byte_vector, CUTLASSConfigCacheEntry(best));
  }

  env->SetTunableConfig(best);
  env->Init(call);
  return env;
}

/*!
 * \brief Dispatch fused functions to CUTLASS. CUTLASS only supports certain
 *        patterns of functions, and when the pattern is not supported, nullptr
 *        is returned so the fused function can be built by TVM.
 *        Patterns supported by CUTLASS:
 *          - gemm_op(a, b)
 *          - gemm_op(a, b) + bias
 *          - epilogue_op(gemm_op(a, b) + bias)
 *        where gemm_op = matmul | matmul_nt | matmul_tn | matmul_tt | dense |
 *                        batch_matmul | batch_matmul_nt | batch_matmul_tn | batch_matmul_tt
 *              epilogue_op = relu
 * \param call the call value to be dispatched
 * \return the CUTLASS OpEnv. nullptr if not supported by CUTLASS
 */
OpEnv* FusedFuncBuild(const op::CallValues& call) {
  using FMaker = std::function<OpEnv*(const CallValues& call)>;
  Function func = Downcast<ClosureValue>(call->callee)->func;
  auto attr = func->GetAttr<String>(attr::kPatternName);
  ICHECK(attr.defined()) << "No pattern name marked for the function";
  std::string pattern_name = attr.value();
  OpEnv* env = nullptr;
  auto fmake_tune = [&env, &call](FMaker maker) {
    env = maker(call);
    if (!env->HasError()) {
      Tune(call, env);
    }
  };
  if (!pattern_name.compare(0, 6, "matmul") || !pattern_name.compare(0, 12, "batch_matmul")) {
    fmake_tune(CutlassMatmulOpEnv::make);
  } else if (!pattern_name.compare(0, 4, "conv")) {
    fmake_tune(CutlassConv2dOpEnv::make);
  } else {
    LOG(FATAL) << "Unknown cutlass fusion pattern: " << pattern_name;
  }

  return env;
}

RAF_OP_ENV_MAKER("raf.op.cutlass._fused_op", FusedFuncBuild);

}  // namespace cutlass
}  // namespace op
}  // namespace raf
