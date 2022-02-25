/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
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
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "./let_list.h"
#include "./common.h"

namespace raf {
namespace pass {
/*
  Lambda lifting pass lifts the closures into the global variables, but as of now it keeps the
  closures inside the lifted function if the closure uses variables other than the arguments.
  In the following example, the closure uses %y, which does not in its argument list.

  ###############
  Original module
  ###############
  def @main(%y: Tensor[(1, 100), float32], %z: Tensor[(1, 100), float32]) {
    let %a1 = fn (%x: Tensor[(1, 100), float32]) {
      let %a2 = raf.op.tanh(%x);
      let %a3 = raf.op.tanh(%y);
      let %a4 = raf.op.add(%a3, %a4, nullptr, nullptr);
      %a4
    };
    let %a5 = %a1; <-- This may be caused by Relay ToANF pass to avoid nested let bindings
    let %a6 = %a5(%z);
    %a6
  }

  ####################
  After Lambda lifting
  ####################
  def @main(%y: Tensor[(1, 100), float32], %z: Tensor[(1, 100), float32]) {
    let %a1 = @lifted_name17350894363744824019(%y);
    let %a5 = %a1;
    let %a6 = %a5(%z);
    %a6
  }

  // Lambda is lifted, but the closure is preserved to keep the semantic of using %y.
  def @lifted_name17350894363744824019(%y, Closure=1) {
    fn (%x: Tensor[(1, 100), float32]) {
        let %a2 = raf.op.tanh(%x);
        let %a3 = raf.op.tanh(%y);
        let %a4 = raf.op.add(%a2, %a3, nullptr, nullptr);
        %a5
      }
  }

  This pass flats the closure such that the lifted closure just has the func body.
  #####################
  After Flatten Closure
  #####################
  // Now we only need one caller with all required arguments.
  def @main(%y: Tensor[(1, 100), float32], %z: Tensor[(1, 100), float32]) {
    let %a6 = @lifted_name17350894363744824019(%z, %y);
    %a6
  }

  // The lifted function is flattened to have only the func body.
  def @lifted_name17350894363744824019(%x: Tensor[(1, 100), float32], %y) {
    let %a3 = raf.op.tanh(%x);
    let %a4 = raf.op.tanh(%y);
    let %a5 = raf.op.add(%a3, %a4, -114514, -114514);
    %a5
  }
*/
namespace flatten_closure {

using namespace raf::ir;
using namespace raf::op;

template <typename T>
using StdMap = std::unordered_map<Var, T, ObjectPtrHash, ObjectPtrEqual>;

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
    return Function(all_vars, new_body, closure_node->ret_type, {});
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
    // We target to the following pattern:
    //   let %a1 = @lifted_name(%y); // Calling global var with captured vars.
    //   let %a2 = %a1;              // It is possible to have redundant let.
    //   let %a3 = %a2(%x);          // Calling the lifted function with arguments.
    // And transform them to:
    //   let %x1 = @lifted_name(%y, %x); // The lifted function should have been flatten.
    auto var = let_node->var;
    auto value = Mutate(let_node->value);
    if (auto var_node = value.as<VarNode>()) {
      auto local_var = GetRef<Var>(var_node);
      if (var_to_closure_captured_vars_.count(local_var) > 0) {
        // In this case, var is redundant (i.e., %a2) so we map %a2 -> %a1 and remove this binding.
        var_to_closure_captured_vars_[var] = var_to_closure_captured_vars_[local_var];
        var_to_closure_lifted_gvar_[var] = var_to_closure_lifted_gvar_[local_var];
        return Mutate(let_node->body);
      }
    } else if (auto call_node = value.as<CallNode>()) {
      if (auto gvar_node = call_node->op.as<GlobalVarNode>()) {
        // Save the let var and captured var to replace the call site later on.
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
  /*! \brief The mapping from let var to the lifted gvar extracted from the value binding */
  StdMap<GlobalVar> var_to_closure_lifted_gvar_;
  /*! \brief The mapping from let var to the captured variable used in the lifted gvar call */
  StdMap<Array<Expr>> var_to_closure_captured_vars_;
  /*! \brief The set of lifted global vars. */
  std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual> lifted_gvars_;
};

}  // namespace flatten_closure

Pass FlattenClosure() {
  TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m, PassContext pc) {
    return flatten_closure::ClosureFlattener(m).Flatten();
  };
  return CreateModulePass(pass_func, 1, "FlattenClosure", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.FlattenClosure").set_body_typed(FlattenClosure);

}  // namespace pass
}  // namespace raf
