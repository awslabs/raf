/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/regs/regs_utils.cc
 * \brief Helpers for operator registry
 */
#include "mnm/tensor.h"
#include "mnm/value.h"
#include "mnm/binding.h"
#include "./regs_utils.h"
#include "../schema/list_args.h"

namespace mnm {
namespace op {
namespace regs {

using namespace mnm::value;
using namespace mnm::ir;
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
}  // namespace mnm
