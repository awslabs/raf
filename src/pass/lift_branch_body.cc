/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file lift_branch_body.cc
 * \brief Lift branch body pass eases the automatic differentiation for If nodes. This pass lifts
 * the true and false branches of an if node into global functions. While lifting, both the global
 * functions have same number of free vars. This ensures that AD will the same return type for
 * igrads on both branches.
 */
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "./common.h"

namespace raf {
namespace pass {
namespace lift_branch_body {

using namespace raf::ir;
using namespace raf::op;

class BranchBodyLift : public MixedModeMutator {
 public:
  explicit BranchBodyLift(IRModule module) : unique_name_counter_(0) {
    module_ = ir::IRModule(module->functions);
  }

  Expr VisitExpr_(const LetNode* let_node) final {
    if (auto func = let_node->value.as<FunctionNode>()) {
      if (!func->HasNonzeroAttr(attr::kPrimitive)) {
        closure_vars_.insert(let_node->var.get());
      }
    }
    auto value = VisitExpr(let_node->value);
    auto body = VisitExpr(let_node->body);
    return Let(let_node->var, value, body);
  }

  Expr VisitExpr_(const IfNode* if_node) final {
    Expr cond = VisitExpr(if_node->cond);
    Expr true_branch = VisitExpr(if_node->true_branch);
    Expr false_branch = VisitExpr(if_node->false_branch);

    // Check that true and false branches are not function calls
    // TODO (janimesh) - Skip the check and avoid the transformation if the branches are already
    // function calls
    CHECK(!is_function_call(true_branch)) << "Branch is already a function";
    CHECK(!is_function_call(false_branch)) << "Branch is already a function";

    // Collect the free vars on both branches. For AD, we would need to return the
    // closure of IfNode such that the input grads from both branches should be of
    // same datatype. So, collect the union of the free vars on both branches.
    Array<Var> all_vars = Concat(pass::FreeVars(true_branch), pass::FreeVars(false_branch));
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> free_vars_set(all_vars.begin(),
                                                                         all_vars.end());

    Array<Expr> free_vars(free_vars_set.begin(), free_vars_set.end());
    Array<Var> func_params;
    for (auto free_var : free_vars) {
      Var fv = Downcast<Var>(free_var);
      // If there is a closure and it is called in the branch bodies. The closure var will also be
      // lifted. This will cause problems if the closure was recrusive. So, we should first run
      // Lambda lift and then apply LiftBranchBody
      CHECK(closure_vars_.count(fv.get()) == 0)
          << "There are closure calls in the branch bodies. Apply LambdaLift pass";
      func_params.push_back(raf::ir::Var(fv));
    }

    // Create new functions for the true and false branches.
    auto true_branch_type = if_node->true_branch->checked_type_;
    Function true_branch_func = CreateGlobalFunc(func_params, true_branch, true_branch_type);
    GlobalVar true_branch_gvar = GlobalVar("true_branch_" + std::to_string(unique_name_counter_));
    module_->Add(true_branch_gvar, true_branch_func);
    auto new_true_branch = Call(true_branch_gvar, free_vars);

    auto false_branch_type = if_node->false_branch->checked_type_;
    Function false_branch_func = CreateGlobalFunc(func_params, false_branch, false_branch_type);
    GlobalVar false_branch_gvar = GlobalVar("false_branch_" + std::to_string(unique_name_counter_));
    module_->Add(false_branch_gvar, false_branch_func);
    auto new_false_branch = Call(false_branch_gvar, free_vars);

    If new_if = If(cond, new_true_branch, new_false_branch);
    unique_name_counter_++;
    return new_if;
  }

  ir::IRModule Lift() {
    auto glob_funcs = module_->functions;
    for (auto pair : glob_funcs) {
      if (auto* n = pair.second.as<FunctionNode>()) {
        auto func = GetRef<Function>(n);
        func = Function(func->params, VisitExpr(func->body), func->ret_type, func->type_params,
                        func->attrs);
        module_->Add(pair.first, func, true);
      }
    }
    return module_;
  }

 private:
  bool is_function_call(const Expr& expr) {
    if (auto* call_node = expr.as<CallNode>()) {
      return call_node->op.as<GlobalVarNode>() != nullptr;
    }
    return false;
  }

 private:
  // initialized in constructor
  IRModule module_;
  std::unordered_set<const VarNode*> closure_vars_;
  int unique_name_counter_;
};

}  // namespace lift_branch_body

Pass LiftBranchBody() {
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        return lift_branch_body::BranchBodyLift(mod).Lift();
      },
      0, "LiftBranchBody", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.LiftBranchBody").set_body_typed(LiftBranchBody);

}  // namespace pass
}  // namespace raf
