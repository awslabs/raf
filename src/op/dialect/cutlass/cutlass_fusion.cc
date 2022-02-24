/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/cutlass/cutlass_fusion.cc
 * \brief Implementation of cutlass dispatch for fused functions
 */
#include <limits>

#include "raf/value.h"
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

OpEnv* Tune(const op::CallValues& call, OpEnv* op_env) {
  CutlassOpEnv* env = static_cast<CutlassOpEnv*>(op_env);
  std::vector<std::unique_ptr<TunableConfig>> tunable = env->ListTunableConfigs();
  const int number = 10, repeat = 1, min_repeat_ms = 0;
  std::unique_ptr<TunableConfig> best;
  double min_time = std::numeric_limits<double>::max();
  for (auto& i : tunable) {
    env->SetTunableConfig(i);
    env->Init(call);
    Array<FloatValue> result = TimeEvaluator(TypedPackedFunc<void()>([&]() { env->Execute(call); }),
                                             call->device, number, repeat, min_repeat_ms)();
    CHECK_EQ(result.size(), 1U);
    if (result[0]->value < min_time) {
      min_time = result[0]->value;
      best = std::move(i);
    }
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
    if (env) {
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
