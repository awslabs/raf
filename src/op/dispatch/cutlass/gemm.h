/*!
 * Copyright (c) 2021 by Contributors
 * \file ./src/op/dispatch/cutlass/gemm.h
 * \brief Implementation of cutlass gemm dispatch
 */
#include "mnm/value.h"
#include "mnm/registry.h"
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/ir_ext.h"
#include "mnm/pass.h"
#include "tvm/ir/type_functor.h"
#include "tvm/relay/dataflow_pattern.h"
#include "./gemm_utils.h"

namespace mnm {
namespace op {
namespace cutlass {

using namespace mnm::ir;
using namespace mnm::value;

/*! \brief OpEnv for the following pattern:
 * epilogue_op_(alpha * matmul_like_ops(a_, b_) + beta * bias_),
 * where 1) matmul_like_ops can be matmul, matmul_nt, matmul_tn, matmul_tt,
 * batch_matmul, batch_matmul_nt, batch_matmul_tn, batch_matmul_tt or dense;
 * 2) epilogue_op_ can be clamp, relu, gelu, sigmoid or identity;
 * 3) alpha and beta are arbitrary scalars.
 */
class CutlassMatmulOpEnv : public CutlassGemmOpEnv {
 public:
  explicit CutlassMatmulOpEnv(const CallValues& cv) : CutlassGemmOpEnv(cv) {
  }

  bool Pattern(const CallValues& cv);

  void Init(const CallValues& cv) override;

  void Execute(const std::vector<Value>& inputs, Value output);

  bool IsValid();

  static OpEnv* make(const CallValues& cv);

 private:
  /*! \brief matmul operand a */
  Var a_;
  /*! \brief matmul operand b */
  Var b_;
  /*! \brief bias added to matmul result */
  Var bias_;
  /*! \brief whether to transpose a */
  bool transpose_a_;
  /*! \brief whether to transpose b */
  bool transpose_b_;
  /*! \brief whether to use bias */
  bool with_bias_;
  /*! \brief batch matmul or not */
  bool batched_;
  /*! \brief epilogue operator like relu, gelu, etc. */
  EpilogueKind epilogue_op_;
};

}  // namespace cutlass
}  // namespace op
}  // namespace mnm
