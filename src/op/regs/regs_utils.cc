/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/regs/regs_utils.cc
 * \brief Helpers for operator registry
 */
#include "raf/tensor.h"
#include "raf/value.h"
#include "raf/binding.h"
#include "./regs_utils.h"
#include "../schema/list_args.h"

namespace raf {
namespace op {
namespace regs {

using namespace raf::value;
using namespace raf::ir;
using binding::BindNDArray;
using binding::GradTape;
using registry::TVMArgValue;

class UsedVars : public ir::ExprVisitor {
 public:
  explicit UsedVars(std::vector<const ExprNode*>* vars) : vars(vars) {
  }
  void VisitExpr_(const VarNode* op) final {
    vars->push_back(op);
  }
  std::vector<const ExprNode*>* vars;
};

void CollectVars(const Expr& expr, std::vector<const ExprNode*>* vars) {
  UsedVars(vars).VisitExpr(expr);  // NOLINT(*)
  std::sort(vars->begin(), vars->end());
}

}  // namespace regs
}  // namespace op
}  // namespace raf
