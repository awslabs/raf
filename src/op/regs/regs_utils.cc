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
using binding::GradTape;
using binding::BindNDArray;
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

ObjectRef DeTuple(const Value& value) {
  if (value->IsInstance<ScalarValueObj>()) {
    return value;
  }
  if (value->IsInstance<TensorValueObj>()) {
    return BindNDArray(value);
  }
  if (const auto* tuple = value.as<TupleValueObj>()) {
    Array<ObjectRef> result;
    int n = static_cast<int>(tuple->fields.size());
    for (int i = 0; i < n; ++i) {
      Value sub_value = tuple->fields[i];
      if (sub_value->op_env == nullptr) {
        sub_value->op_env = tuple->op_env;
      }
      result.push_back(DeTuple(sub_value));
    }
    return std::move(result);
  }
  LOG(FATAL) << "ValueError: cannot de-tuple " << value->GetTypeKey();
  throw;
}

ObjectRef DeStruct(Value value, ClosureValue bp, Array<ObjectRef> prev_tapes) {
  if (value->IsInstance<ScalarValueObj>()) {
    return std::move(value);
  }
  GradTape tape = GradTape::make(
      /*dy=*/binding::BindNDArray({}),
      /*bp=*/std::move(bp),
      /*prev_tapes=*/std::move(prev_tapes));
  if (value->IsInstance<TensorValueObj>()) {
    return BindNDArray(std::move(value), std::move(tape));
  }
  if (const auto* tuple = value.as<TupleValueObj>()) {
    Array<ObjectRef> result;
    int n = static_cast<int>(tuple->fields.size());
    Var dy = VarNode::make("dy", {});
    std::vector<Expr> grads(n, MakeConstant(NoGradValue::make()));
    for (int i = 0; i < n; ++i) {
      Value sub_value = tuple->fields[i];
      if (sub_value->op_env == nullptr) {
        sub_value->op_env = tuple->op_env;
      }
      grads[i] = dy;
      result.push_back(DeStruct(
          /*value=*/sub_value,
          /*bp=*/ClosureValue::make({}, FunctionNode::make({dy}, TupleNode::make(grads), {}, {})),
          /*prev_tapes*/ {tape}));
    }
    return std::move(result);
  }
  LOG(FATAL) << "ValueError: cannot de-tuple " << value->GetTypeKey();
  throw;
}

}  // namespace regs
}  // namespace op
}  // namespace mnm
