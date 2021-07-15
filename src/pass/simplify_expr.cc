/*!
 * Copyright (c) 2021 by Contributors
 * \file simplify_expr.cc
 * \brief Simplifies the commonly seen patterns.
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "mnm/value.h"

namespace mnm {
namespace pass {
namespace simplify_expr {

using namespace mnm::ir;
using namespace mnm::op;
using namespace mnm::value;

class ExprSimplifier : public ExprMutator {
 public:
  explicit ExprSimplifier() : sum_op_(Op::Get("mnm.op.sum")) {
  }

 private:
  Expr SimplifySum(const CallNode* call_node) {
    // Simplify sum when the reduce axis are empty
    if (call_node->args.size() > 1) {
      const Expr& axis = call_node->args[1];
      if (auto const_node = axis.as<ConstantNode>()) {
        auto data = const_node->value;
        if (data.defined()) {
          if (auto tuple_data = data.as<TupleValueObj>()) {
            if (tuple_data->fields.size() == 0) {
              return call_node->args[0];
            }
          }
        }
      }
    }
    return GetRef<Expr>(call_node);
  }

 public:
  Expr Simplify(const Expr& expr) {
    return this->Mutate(expr);
  }

  Expr VisitExpr_(const VarNode* var_node) final {
    Var var = GetRef<Var>(var_node);
    if (mapping_.count(var)) {
      return mapping_.at(var);
    }
    return var;
  }

  Expr VisitExpr_(const CallNode* node) final {
    Expr expr = ExprMutator::VisitExpr_(node);
    if (node->op.same_as(sum_op_)) {
      return SimplifySum(expr.as<CallNode>());
    }
    return Downcast<Call>(expr);
  }

  Expr VisitExpr_(const LetNode* let_node) final {
    auto pre_visit = [this](const LetNode* op) {
      auto var = op->var;
      auto value = this->Mutate(op->value);
      // Simplify let %a = %b .. scenarios
      if (auto value_var = value.as<VarNode>()) {
        mapping_[var] = GetRef<Var>(value_var);
      }
    };
    auto post_visit = [this](const LetNode* op) {
      auto var = op->var;
      auto value = this->Mutate(op->value);
      auto body = this->Mutate(op->body);

      if (auto value_var = value.as<VarNode>()) {
        this->memo_[GetRef<Expr>(op)] = body;
      } else {
        this->memo_[GetRef<Expr>(op)] = Let(var, value, body);
      }
    };
    ExpandANormalForm(let_node, pre_visit, post_visit);
    return memo_[GetRef<Expr>(let_node)];
  }

 private:
  const Op& sum_op_;
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> mapping_;
};
}  // namespace simplify_expr

Pass SimplifyExpr() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(simplify_expr::ExprSimplifier().Simplify(f));
  };
  return CreateMNMFunctionPass(pass_func, 0, "SimplifyExpr", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.SimplifyExpr").set_body_typed(SimplifyExpr);

}  // namespace pass
}  // namespace mnm
