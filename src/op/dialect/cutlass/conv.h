/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/cutlass/conv.h
 * \brief Implementation of cutlass convolution dispatch
 */
#include "cutlass/library/operation_table.h"
#include "raf/value.h"
#include "raf/registry.h"
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/pass.h"
#include "raf/op_utils.h"
#include "tvm/ir/type_functor.h"
#include "tvm/relay/dataflow_pattern.h"

#include "./cutlass_utils.h"
#include "./conv_utils.h"
#include "./pattern_utils.h"
#include "../../schema/ufunc.h"
#include "../../schema/nn.h"
#include "../../../common/shape_utils.h"
#include "../../regs/value2schema.h"

namespace raf {
namespace op {
namespace cutlass {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::op::schema;
using raf::op::regs::value2schema::TupleInt;
using raf::registry::PackedFunc;
using raf::registry::TypedPackedFunc;

/*! \brief OpEnv for the following pattern:
 * epilogue_op_(alpha * conv2d(a_, b_) + beta * bias_),
 * where 1) epilogue_op_ can be relu or identity, and can be extended to clamp, gelu and sigmoid
 * 2) alpha and beta are arbitrary scalars.
 */
class CutlassConv2dOpEnv : public CutlassConvOpEnv {
 public:
  explicit CutlassConv2dOpEnv(const CallValues& cv) : CutlassConvOpEnv(cv) {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cutlass.conv2d"));
  }

  bool Pattern(const CallValues& cv);

  void Init(const CallValues& cv) override;

  void Execute(const std::vector<Value>& inputs, Value output);

  bool IsValid(const CallValues& cv);

  static OpEnv* make(const CallValues& cv);

 private:
  /*! \brief conv operand x */
  Var x_;
  /*! \brief conv operand w */
  Var w_;
  /*! \brief bias added to conv result */
  Var bias_;
  /*! \brief convolution stride */
  std::vector<int64_t> stride_;
  /*! \brief convolution padding */
  std::vector<int64_t> padding_;
  /*! \brief convolution dilation */
  std::vector<int64_t> dilation_;
  /*! \brief input layout */
  std::string layout_;
  /*! \brief kernel layout */
  std::string kernel_layout_;
  /*! \brief output layout */
  std::string out_layout_;
  /*! \brief whether to use bias */
  bool with_bias_;
  /*! \brief epilogue operator like relu, gelu, etc. */
  EpilogueKindExt epilogue_op_;
};

}  // namespace cutlass
}  // namespace op
}  // namespace raf
