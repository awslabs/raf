/*!
 * Copyright (c) 2021 by Contributors
 * \file lift_branch_body.cc
 * \brief Lift branch body pass eases the automatic differentiation for If nodes. This pass lifts
 * the true and false branches of an if node into global functions. While lifting, both the global
 * functions have same number of free vars. This ensures that AD will the same return type for
 * igrads on both branches.
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"

namespace mnm {
namespace pass {
namespace lift_branch_body {

using namespace mnm::ir;
using namespace mnm::op;

class BranchBodyLift : public MixedModeMutator {
 public:
  explicit BranchBodyLift(IRModule module) : unique_name_counter_(0) {
    module_ = ir::IRModule(module->functions);
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
      func_params.push_back(mnm::ir::Var(Downcast<Var>(free_var)));
    }

    // both true and false branch will have same ret type, set the return type of the global
    // funcs to simplify InferType.
    auto if_node_ret_type = if_node->true_branch->checked_type_;

    // Create new functions for the true and false branches.
    Function true_branch_func = Function(func_params, true_branch, if_node_ret_type, {});
    GlobalVar true_branch_gvar = GlobalVar("true_branch_" + std::to_string(unique_name_counter_));
    module_->Add(true_branch_gvar, true_branch_func);
    auto new_true_branch = Call(true_branch_gvar, free_vars);
    new_true_branch->checked_type_ = if_node_ret_type;

    Function false_branch_func = Function(func_params, false_branch, if_node_ret_type, {});
    GlobalVar false_branch_gvar = GlobalVar("false_branch_" + std::to_string(unique_name_counter_));
    module_->Add(false_branch_gvar, false_branch_func);
    auto new_false_branch = Call(false_branch_gvar, free_vars);
    new_false_branch->checked_type_ = if_node_ret_type;

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
  int unique_name_counter_;
};

}  // namespace lift_branch_body

ir::IRModule LiftBranchBody(ir::IRModule mod) {
  return lift_branch_body::BranchBodyLift(mod).Lift();
}

MNM_REGISTER_GLOBAL("mnm.pass_.LiftBranchBody").set_body_typed(LiftBranchBody);

}  // namespace pass
}  // namespace mnm
