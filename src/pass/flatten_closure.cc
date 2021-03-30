/*!
 * Copyright (c) 2021 by Contributors
 * \file flatten_closure.cc
 * \brief This is applied after Lambda lifting. Lambda lifting pass lifts the closures to global
 * scope, but the lifted global function still has the closure within. This makes AD harder. This
 * pass flattens the global functions that are marked Closure, and then changes the call sites
 * accordingly. This helps AD pass where it is difficult to handle closures.
 */
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <sstream>
#include <utility>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "./let_list.h"
#include "./common.h"

namespace mnm {
namespace pass {
/*
  Lambda lifting pass lifts the closures into the global variables, but as of now it keeps the
  closures inside the lifted function. For example,
  ###############
  Original module
  ###############
  def @main(%y: Tensor[(1, 100), float32], %z: Tensor[(1, 100), float32]) {
    let %a1 = let %a2 = fn (%x: Tensor[(1, 100), float32]) {
      let %a3 = mnm.op.tanh(%x);
      let %a4 = mnm.op.tanh(%y);
      let %a5 = mnm.op.add(%a3, %a4, -114514, -114514);
      %a5
    };
    %a2; <-- a2 is the closure
    let %a6 = %a1(%z);
    %a6
  }


  ####################
  After Lambda lifting
  ####################
  def @main(%y: Tensor[(1, 100), float32], %z: Tensor[(1, 100), float32]) {
    let %a1 = @lifted_name17350894363744824019(%y);
    let %a6 = %a1(%z);
    %a6
  }

  --> Lambda is lifted to global var, but it has the closure inside
  def @lifted_name17350894363744824019(%y, Closure=1) {
  fn (%x: Tensor[(1, 100), float32]) {
      let %a3 = mnm.op.tanh(%x);
      let %a4 = mnm.op.tanh(%y);
      let %a5 = mnm.op.add(%a3, %a4, -114514, -114514);
      %a5
    }
  }

  This pass flattens the closure such that the lifted closure just has the func body. This also
  changes the call sites.
  #####################
  After Flatten Closure
  #####################
  def @main(%y: Tensor[(1, 100), float32], %z: Tensor[(1, 100), float32]) {
    let %a6 = @lifted_name17350894363744824019(%z, %y);
    %a6
  }

  def @lifted_name17350894363744824019(%x: Tensor[(1, 100), float32], %y) {
    let %a3 = mnm.op.tanh(%x);
    let %a4 = mnm.op.tanh(%y);
    let %a5 = mnm.op.add(%a3, %a4, -114514, -114514);
    %a5
  }

*/
namespace flatten_closure {

using namespace mnm::ir;
using namespace mnm::op;

class ClosureFlattener : public MixedModeMutator {
 public:
  explicit ClosureFlattener(IRModule module) {
    module_ = IRModule(module->functions);
  }

  Function FlatGlobalFunc(const Function& func) {
    auto closure_node = func->body.as<FunctionNode>();
    CHECK(closure_node);
    auto new_body = closure_node->body;
    auto all_vars = Concat(closure_node->params, func->params);
    return Function(all_vars, new_body, {}, {});
  }

  IRModule Flatten() {
    std::vector<std::pair<ir::GlobalVar, ir::Function>> updated_funcs;

    // The pass can be broken down into 2 components
    // 1) Flat the global functions that are marked closure.
    // 2) Go through all the function bodies and change call sites. Now, the call sites needs to
    // call with both free vars and captured vars Change call sites and collect the functions that
    // are marked closure Flat the functions
    for (auto kv : module_->functions) {
      if (kv.second.as<ir::FunctionNode>()) {
        Function func = tvm::runtime::Downcast<ir::Function>(kv.second);
        if (func->HasNonzeroAttr("Closure")) {
          lifted_gvars_.insert(kv.first);
          auto flat_func = FlatGlobalFunc(func);
          module_->Add(kv.first, flat_func, true);
        }
      }
    }

    // Change the call site
    for (auto kv : module_->functions) {
      if (kv.second.as<ir::FunctionNode>()) {
        auto expr = this->Mutate(kv.second);
        auto func = tvm::runtime::Downcast<ir::Function>(expr);
        updated_funcs.emplace_back(kv.first, func);
      }
    }

    for (const auto& it : updated_funcs) {
      module_->Add(it.first, it.second, true);
    }
    return module_;
  }

  Expr VisitExpr_(const LetNode* let_node) final {
    auto var = let_node->var;
    auto value = Mutate(let_node->value);
    if (auto call_node = value.as<CallNode>()) {
      if (auto gvar_node = call_node->op.as<GlobalVarNode>()) {
        // Check if the value is actually a global var call which is lifted gvar
        // If yes, save the let var and captured var to replace the call site later on
        auto gvar = GetRef<GlobalVar>(gvar_node);
        auto global_func = module_->Lookup(gvar);
        if (lifted_gvars_.count(gvar)) {
          auto captured_vars = call_node->args;
          var_to_closure_captured_vars_[var] = captured_vars;
          var_to_closure_lifted_gvar_[var] = gvar;
          return Mutate(let_node->body);
        }
      } else if (auto var_node = call_node->op.as<VarNode>()) {
        // Check if the call node op is var node which is the saved lifted gvar earlier
        // If yes, replace the call with the lifted gvar and args with free vars + captured vars
        auto local_var = GetRef<Var>(var_node);
        if (var_to_closure_lifted_gvar_.count(local_var)) {
          auto gvar = var_to_closure_lifted_gvar_.at(local_var);
          auto captured_vars = var_to_closure_captured_vars_.at(local_var);
          auto all_vars = Concat(call_node->args, captured_vars);
          auto new_call = Call(gvar, all_vars, call_node->attrs, call_node->type_args);
          auto body = Mutate(let_node->body);
          return Let(var, new_call, body);
        }
      }
    }

    auto body = Mutate(let_node->body);
    return Let(var, value, body);
  }

 private:
  // initialized in constructor
  IRModule module_;
  /* \brief The mapping from let var to the lifted gvar extracted from the value binding */
  std::unordered_map<Var, GlobalVar, ObjectPtrHash, ObjectPtrEqual> var_to_closure_lifted_gvar_;
  /* \brief The mapping from let var to the captured variable used in the lifted gvar call */
  std::unordered_map<Var, Array<Expr>, ObjectPtrHash, ObjectPtrEqual> var_to_closure_captured_vars_;
  std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual> lifted_gvars_;
};

}  // namespace flatten_closure

IRModule FlattenClosure(IRModule mod) {
  return flatten_closure::ClosureFlattener(mod).Flatten();
}

MNM_REGISTER_GLOBAL("mnm.pass_.FlattenClosure").set_body_typed(FlattenClosure);

}  // namespace pass
}  // namespace mnm
