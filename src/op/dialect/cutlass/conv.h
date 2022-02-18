/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file ./src/op/dialect/cutlass/conv.h
 * \brief Implementation of cutlass convolution dispatch
 */
#include "cutlass/library/operation_table.h"
#include "mnm/value.h"
#include "mnm/registry.h"
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/ir_ext.h"
#include "mnm/pass.h"
#include "mnm/op_utils.h"
#include "tvm/ir/type_functor.h"
#include "tvm/relay/dataflow_pattern.h"

#include "./cutlass_utils.h"
#include "./conv_utils.h"
#include "./pattern_utils.h"
#include "../../schema/ufunc.h"
#include "../../schema/nn.h"
#include "../../../common/shape_utils.h"
#include "../../regs/value2schema.h"

namespace mnm {
namespace op {
namespace cutlass {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::op::schema;
using mnm::op::regs::value2schema::TupleInt;
using mnm::registry::PackedFunc;
using mnm::registry::TypedPackedFunc;

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
    return TruncateName(GetUniqueName("mnm.op.cutlass.conv2d"));
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
}  // namespace mnm
