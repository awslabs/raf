/*!
 * Copyright (c) 2021 by Contributors
 * \file ./src/op/dispatch/cutlass/gemm.cc
 * \brief Implementation of cutlass gemm dispatch
 */
#include "./gemm.h"

#include "mnm/value.h"
#include "mnm/registry.h"
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/ir_ext.h"
#include "mnm/pass.h"
#include "tvm/ir/type_functor.h"
#include "tvm/relay/dataflow_pattern.h"
#include "./cutlass_utils.h"
#include "./pattern_utils.h"
#include "./gemm_utils.h"
#include "../../schema/ufunc.h"
#include "../../schema/nn.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace cutlass {

using namespace mnm::ir;
using namespace mnm::value;
using mnm::registry::PackedFunc;
using mnm::registry::TypedPackedFunc;

std::tuple<bool, bool> GetTranspose(const Op& op) {
  const static std::vector<Op> transpose_a_ops = {
      Op::Get("mnm.op.matmul_tn"), Op::Get("mnm.op.matmul_tt"), Op::Get("mnm.op.batch_matmul_tn"),
      Op::Get("mnm.op.batch_matmul_tt")};
  const static std::vector<Op> transpose_b_ops = {
      Op::Get("mnm.op.dense"), Op::Get("mnm.op.matmul_nt"), Op::Get("mnm.op.matmul_tt"),
      Op::Get("mnm.op.batch_matmul_nt"), Op::Get("mnm.op.batch_matmul_tt")};
  bool transpose_a =
      std::find(transpose_a_ops.begin(), transpose_a_ops.end(), op) != transpose_a_ops.end();
  bool transpose_b =
      std::find(transpose_b_ops.begin(), transpose_b_ops.end(), op) != transpose_b_ops.end();
  return {transpose_a, transpose_b};
}

bool IsBatch(const Op& op) {
  const static std::vector<Op> batched_ops = {
      Op::Get("mnm.op.batch_matmul"), Op::Get("mnm.op.batch_matmul_nt"),
      Op::Get("mnm.op.batch_matmul_tn"), Op::Get("mnm.op.batch_matmul_tt")};
  return std::find(batched_ops.begin(), batched_ops.end(), op) != batched_ops.end();
}

// Generates gelu(pat).
DFPattern GELU(DFPattern pat) {
  auto erf = IsOp("mnm.op.erf");
  auto inv_sqrt_2 = IsVar("");
  auto inv_2 = IsVar("");
  return Multiply()(pat, Add()(inv_2, Multiply()(erf({Multiply()(pat, inv_sqrt_2)}), inv_2)));
}

bool CutlassMatmulOpEnv::Pattern(const CallValues& cv) {
  Expr expr = Downcast<ClosureValue>(cv->callee)->func->body;
  const static std::vector<std::string> matmul_ops = {
      "mnm.op.dense",           "mnm.op.matmul",          "mnm.op.matmul_nt",
      "mnm.op.matmul_tn",       "mnm.op.matmul_tt",       "mnm.op.batch_matmul",
      "mnm.op.batch_matmul_nt", "mnm.op.batch_matmul_tn", "mnm.op.batch_matmul_tt"};
  const static std::vector<std::string> epilogue_ops = {"mnm.op.relu"};
  auto matmul = IsOps(matmul_ops);
  auto add = IsOp("mnm.op.add");
  auto epilogue = IsOps(epilogue_ops);
  auto konst1 = IsConstant();
  auto konst2 = IsConstant();
  auto x1 = IsVar("");
  auto x2 = IsVar("");
  auto beta = IsVar("");
  auto bias = IsVar("");
  DFPattern pat = matmul({x1, x2});
  DFPattern scaled_bias = bias || Multiply()(beta, bias);
  DFPattern with_bias = Add()(pat, scaled_bias);
  pat = pat || with_bias;
  DFPattern with_epilogue = epilogue({pat});
  pat = pat || with_epilogue;
  // TODO(@hzfan): after we have mnm.op.gelu, add it to epilogue_ops
  DFPattern with_epilogue_gelu = GELU(pat);
  pat = pat || with_epilogue_gelu;

  if (!MatchPattern(pat, expr)) {
    return false;
  }

  // RewritePatterns serves as a visitor here: it does not rewrite, instead information
  // is recorded for later process.
  TypedPackedFunc<Expr(const Expr&, const Expr&, const Map<DFPattern, Array<Expr>>&)> func(
      [&](const Expr& pre, const Expr& post, const Map<DFPattern, Array<Expr>>& node_map) {
        Op matmul_op = GetPattern<Op>(node_map, matmul);
        a_ = GetPattern<Var>(node_map, x1);
        b_ = GetPattern<Var>(node_map, x2);
        with_bias_ = GetPattern<Expr>(node_map, scaled_bias).defined();
        epilogue_op_ = GetEpilogueKind(GetPattern<Op>(node_map, epilogue));
        // TODO(@hzfan): remove this if after we have mnm.op.gelu
        if (GetPattern<Expr>(node_map, with_epilogue_gelu).defined()) {
          epilogue_op_ = EpilogueKindExt::kLinearCombinationGELU;
        }
        batched_ = IsBatch(matmul_op);
        std::tie(transpose_a_, transpose_b_) = GetTranspose(matmul_op);
        if (with_bias_) {
          beta_ = GetPattern<Var>(node_map, beta);
          bias_ = GetPattern<Var>(node_map, bias);
        }
        return post;
      });
  DFPatternCallback cb(pat, func.operator PackedFunc(), false);
  RewritePatterns({cb}, expr);
  return true;
}

bool CutlassMatmulOpEnv::IsValid(const CallValues& cv) {
  bool flag = a_.defined() && b_.defined() && (!with_bias_ || bias_.defined());
  if (beta_.defined()) {
    DLTensor* x = GetValue<TensorValue>(cv, beta_);
    flag = flag && ((x->ndim == 0) || (x->ndim == 1 && x->shape[0] == 1));
  }
  return flag;
}

void CutlassMatmulOpEnv::Init(const CallValues& cv) {
  DLTensor* a = GetValue<TensorValue>(cv, a_);
  DLTensor* b = GetValue<TensorValue>(cv, b_);
  DLTensor* c = cv->out;
  DLTensor* bias = with_bias_ ? GetValue<TensorValue>(cv, bias_) : c;
  bool batched = batched_;
  int batch = batched ? std::max(a->shape[0], b->shape[0]) : -1;
  int m = c->shape[batched + 1];
  int n = c->shape[batched + 0];
  int k = transpose_b_ ? b->shape[batched + 1] : b->shape[batched + 0];
  CHECK(DType(c->dtype) == DType(bias->dtype));

  int lda = a->shape[batched + 1];
  int ldb = b->shape[batched + 1];
  int ldc = c->shape[batched + 1];
  int ldbias = bias->ndim < 2 ? 0 : bias->shape[bias->ndim - 1];
  int batch_stride_a = batched ? (a->shape[0] == 1 ? 0 : a->shape[1] * a->shape[2]) : -1;
  int batch_stride_b = batched ? (b->shape[0] == 1 ? 0 : b->shape[1] * b->shape[2]) : -1;
  int batch_stride_c = batched ? (c->shape[1] * c->shape[2]) : -1;
  int batch_stride_bias = batched ? (bias->ndim < 3 ? 0 : bias->shape[1] * bias->shape[2]) : -1;
  const float* ptra = static_cast<float*>(a->data);
  const float* ptrb = static_cast<float*>(b->data);
  const float* ptrbias = static_cast<float*>(bias->data);
  float* ptrc = static_cast<float*>(c->data);

  const auto layouta = transpose_a_ ? LayoutTypeID::kRowMajor : LayoutTypeID::kColumnMajor;
  const auto layoutb = transpose_b_ ? LayoutTypeID::kRowMajor : LayoutTypeID::kColumnMajor;

  DType accumulation_dtype = GetAccumulationDType(DType(c->dtype));
  DType scalar_dtype = accumulation_dtype;
  const void* alpha = const_addr<1>(cudaDataType_t(scalar_dtype));
  const void* beta = with_bias_ ? const_addr<1>(cudaDataType_t(scalar_dtype))
                                : const_addr<0>(cudaDataType_t(scalar_dtype));
  if (beta_.defined()) {
    beta_ptr_ = shared_addr(cudaDataType_t(scalar_dtype),
                            GetScalarValueData<float>(GetValue<TensorValue>(cv, beta_)));
    beta = beta_ptr_.get();
  }

  InitGemmOperation(batched ? GemmUniversalMode::kBatched : GemmUniversalMode::kGemm, m, n, k,
                    GetNumericTypeID(accumulation_dtype), GetNumericTypeID(scalar_dtype), alpha,
                    GetNumericTypeID(b->dtype), layoutb, ptrb, ldb, GetNumericTypeID(a->dtype),
                    layouta, ptra, lda, beta, GetNumericTypeID(c->dtype), ptrbias, ldbias, ptrc,
                    ldc, batched ? batch : tunable_.split_k_slices, batch_stride_b, batch_stride_a,
                    batch_stride_bias, batch_stride_c, epilogue_op_, tunable_.kernel_name);
  arg_indices = GetArgIndices(
      cv, with_bias_ ? std::vector<Var>({a_, b_, bias_}) : std::vector<Var>({a_, b_}));
}

OpEnv* CutlassMatmulOpEnv::make(const CallValues& cv) {
  std::unique_ptr<CutlassMatmulOpEnv> op_env(std::make_unique<CutlassMatmulOpEnv>(cv));
  if (!op_env->Pattern(cv) || !op_env->IsValid(cv)) {
    return nullptr;
  }
  op_env->Init(cv);
  return op_env.release();
}

void CutlassMatmulOpEnv::Execute(const std::vector<Value>& inputs, Value output) {
  DLTensor* x1 = Downcast<TensorValue>(inputs[0]);
  DLTensor* x2 = Downcast<TensorValue>(inputs[1]);
  DLTensor* out = Downcast<TensorValue>(output);
  DLTensor* bias = with_bias_ ? Downcast<TensorValue>(inputs[2]) : out;
  arguments_.A = x2->data;
  arguments_.B = x1->data;
  arguments_.C = bias->data;
  arguments_.D = out->data;
  CUTLASS_CALL(operation_->run(&arguments_, host_workspace_, workspace_, stream_));
}

}  // namespace cutlass
}  // namespace op
}  // namespace mnm
