/*!
 * Copyright (c) 2021 by Contributors
 * \file ./src/op/dispatch/cutlass/cutlass_fusion.cc
 * \brief Implementation of cutlass dispatch for fused functions
 */
#include "mnm/value.h"
#include "mnm/registry.h"
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/ir_ext.h"
#include "mnm/pass.h"
#include "tvm/ir/type_functor.h"
#include "tvm/relay/dataflow_pattern.h"
#include "./gemm.h"

namespace mnm {
namespace op {
namespace cutlass {

using namespace mnm::ir;
using namespace mnm::value;

/*!
 * \brief Dispatch fused functions to CUTLASS. CUTLASS only supports certain
 *        patterns of functions, and when the pattern is not supported, nullptr
 *        is returned so the fused function can be built by TVMJIT.
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
  Function func = Downcast<ClosureValue>(call->callee)->func;
  std::vector<std::function<OpEnv*(const CallValues& call)>> makers = {CutlassMatmulOpEnv::make};
  OpEnv* env = nullptr;
  for (const auto& maker : makers) {
    env = maker(call);
    if (env) {
      break;
    }
  }
  return env;
}

MNM_FUNC_DISPATCH_PLEVEL(FusedFuncBuild, DevType::kCUDA(), "cutlass", 30);

}  // namespace cutlass
}  // namespace op
}  // namespace mnm
