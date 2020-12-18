/*!
 * Copyright (c) 2019 by Contributors
 * \file inline_backward.cc
 * \brief inlining backward graph in the forward pass
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/binding.h"
#include "./common.h"

namespace mnm {
namespace pass {
namespace inline_backward {

using namespace mnm::ir;

class InlineBackwardFunc : public ExprMutator {
 public:
  explicit InlineBackwardFunc(Function func) : func_(func) {
  }

  Expr VisitExpr_(const VarNode* var_node) final {
    auto var = GetRef<Var>(var_node);
    return std::move(var);
  }

  Expr VisitExpr_(const LetNode* let_node) final {
    auto value = VisitExpr(let_node->value);
    auto body = VisitExpr(let_node->body);
    auto vn = let_node->var.as<VarNode>();
    if (body == let_node->var && !is_closure_) {
      /* for return */
      auto ret_fields = value.as<TupleNode>()->fields;
      old_ret_.Assign(ret_fields.begin(), ret_fields.end() - 1);
    } else if (!let_node->value.as<FunctionNode>()) {
      ell_->exprs.push_back(value);
      ell_->vars.push_back(let_node->var);
    }
    return Let(let_node->var, value, body);
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    return ExprMutator::VisitExpr_(call_node);
  }

  Expr VisitExpr_(const FunctionNode* func_node) final {
    is_closure_ = true;
    VisitExpr(func_node->body);
    is_closure_ = false;
    auto func = GetRef<Function>(func_node);
    closure_params_ = func_node->params;
    return Function(func_node->params, func_node->body, func_node->ret_type, {});
  }

  Function Inline() {
    VisitExpr(func_->body);
    auto new_params = func_->params;
    for (auto closure_param : closure_params_) {
      new_params.push_back(closure_param);
    }
    std::reverse(ell_->vars.begin(), ell_->vars.end());
    std::reverse(ell_->exprs.begin(), ell_->exprs.end());
    old_ret_.push_back(ell_->vars.back());
    Var gradient = MakeVar("gradient", {});
    ell_->vars.push_back(gradient);
    ell_->exprs.push_back(Tuple(old_ret_));
    ell_->ret = gradient;
    return Function(new_params, ell_->AsExpr(), {}, {});
  }

 private:
  Function func_;
  Array<Expr> old_ret_;
  Array<Var> closure_params_;
  std::unique_ptr<ExplicitLetList> ell_{std::make_unique<ExplicitLetList>()};
  bool is_closure_ = false;
};

Function InlineBackward(Function func) {
  return InlineBackwardFunc(func).Inline();
}

MNM_REGISTER_GLOBAL("mnm.pass_.InlineBackward").set_body_typed(InlineBackward);
}  // namespace inline_backward
}  // namespace pass
}  // namespace mnm
