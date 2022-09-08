/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/full_inline.cc
 * \brief Recursively inline calls to global functions. Assume LambdaLift() has
 * run before this pass.
 */
#include <unordered_set>
#include <vector>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include "raf/op.h"
#include "raf/ir_ext.h"
#include "raf/pass.h"
#include "raf/binding.h"
#include "./common.h"
#include "./let_list.h"
#include "./call_graph.h"

namespace raf {
namespace pass {
namespace full_inline {

/*!
 *  \brief This pass inlines non-recursive function calls. If the module contains recursive calls
 *  then this pass will be skipped and a warning message will be printed. After this pass, all but
 *  the top-level function will be removed from the module. Assumes the LambdaLift() pass has run
 *  before this pass and the IR in all functions is in ANF.
 *
 *  The InlineClosure() pass has similar functionality, except that it can only handle cases where
 *  each function is invoked once.
 */
using namespace raf::ir;
using namespace raf::op;
using namespace raf::pass::call_graph;
using VarMap = std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual>;

/*!
 *  \brief Perform full inlining on a function. The returned function body will
 *  have no calls to global functions.
 */
class FullInliner : public ExprMutator {
 public:
  explicit FullInliner(const IRModule& mod, const Function& func,
                       const LocalVarGVarMap& local_var_to_gvar)
      : mod_(mod), func_(func), local_var_to_gvar_(local_var_to_gvar) {
    scopes_.emplace_back(new LetList);
    var_map_ = std::make_shared<VarMap>();
  }

  /*! \brief This basically serves as a driver to visit other nodes. */
  Expr VisitExpr_(const LetNode* let_node) {
    scopes_.emplace_back(new LetList);
    auto scope = scopes_.back().get();

    Expr body;
    do {
      curr_let_ = let_node->var;
      auto new_value = VisitExpr(let_node->value);
      // Proceed to the next node
      // Notice that we will keep the let statement even if it is just assigning a global var
      //   let x = %some_global_var
      // This won't cause an issue in later passes because we will run DCE afterwards.
      scope->Push(let_node->var, new_value);
      body = let_node->body;
      let_node = body.as<LetNode>();
    } while (let_node);

    auto new_body = VisitExpr(body);
    auto ret = scopes_.back()->Get(new_body);
    scopes_.pop_back();
    return ret;
  }

  /*! \brief Replace the vars on the RHS of let statements if necessary. */
  Expr VisitExpr_(const VarNode* var_node) {
    auto old_var = GetRef<Var>(var_node);
    if (var_map_->count(old_var)) {
      return (var_map_->find(old_var))->second;
    }
    return old_var;
  }

  /*! \brief Handles call nodes and actually inlines functions. Only global functions are inlined.
   */
  Expr VisitExpr_(const CallNode* call_node) {
    LetList* curr_scope = scopes_.back().get();

    // First replace the call arguments if necessary
    Array<Expr> new_args;
    for (auto old_arg : call_node->args) {
      // Assuming all arguments are Vars here
      if (auto old_arg_var_node = old_arg.as<VarNode>())
        new_args.push_back(VisitExpr_(old_arg_var_node));
      else
        new_args.push_back(old_arg);
    }

    // By default, return the call with arguments changed
    auto call = GetRef<Call>(call_node);
    Expr ret_expr = Call(call->op, new_args, call->attrs, call->type_args);
    ret_expr->checked_type_ = call->checked_type();

    // Check the op, if it is a global function, inline it
    auto op = call_node->op;
    if (auto gvar_node = op.as<GlobalVarNode>()) {
      // Case 1: the called function maps to a global var
      GlobalVar gvar = GetRef<GlobalVar>(gvar_node);
      ret_expr = FindAndInlineFunc(gvar, new_args, curr_scope);
    } else if (auto var_node = op.as<VarNode>()) {
      // Case 2: the called function maps to a local var, this can happen in the following case:
      //   let x = %some_global_var;
      //   let y = x(...)
      Var op_var = GetRef<Var>(var_node);
      if (local_var_to_gvar_.count(op_var)) {
        auto gvar = local_var_to_gvar_.find(op_var)->second;
        ret_expr = FindAndInlineFunc(gvar, new_args, curr_scope);
      }
    }
    return ret_expr;
  }

  /*! \brief Top-level entry point. */
  Function Run() {
    return Function(func_->params, this->VisitExpr(func_->body), func_->ret_type,
                    func_->type_params, func_->attrs, func_->span);
  }

 private:
  /*! \brief Given a global var, find the function and inline it. */
  Expr FindAndInlineFunc(const GlobalVar& gvar, const Array<Expr>& new_args, LetList* curr_scope) {
    CHECK(mod_->functions.count(gvar))
        << "Called global function " << gvar << " is not in the module!";
    auto base_func = (mod_->functions)[gvar];
    if (auto called_func_node = base_func.as<FunctionNode>()) {
      Function called_func = GetRef<Function>(called_func_node);
      Var ret_var = InlineFunc(called_func, new_args, curr_scope);
      var_map_->insert(std::make_pair(curr_let_, ret_var));
      return Downcast<Expr>(ret_var);
    } else {
      LOG(FATAL) << "Called global var " << gvar << " is not a function!";
    }
  }

  /*! \brief Inline the called function into the current function body. */
  Var InlineFunc(const Function& f, const Array<Expr>& new_args, LetList* curr_scope) {
    CHECK_EQ(new_args.size(), f->params.size())
        << "The function should have " << f->params.size() << " parameters, but the arg list has"
        << new_args.size() << " elements!";

    // A local var map, not to be confused with the one in this class
    // This var map only stores var mappings in the called function
    std::shared_ptr<VarMap> var_map_in_func = std::make_shared<VarMap>();
    for (size_t i = 0; i < new_args.size(); i++) {
      var_map_in_func->insert(std::make_pair(f->params[i], new_args[i]));
      // Delete the arguments from the internal memo to force revisiting nodes
      this->memo_.erase(f->params[i]);
    }

    // We want to leverage VisitExpr() to substitute the vars automatically
    // Cheap trick here: let the var_map_ pointer point to the var map we just created, and
    // change it back when we exit this function
    std::shared_ptr<VarMap> tmp = var_map_;
    var_map_ = var_map_in_func;

    // Assume the called function is in ANF
    std::unique_ptr<ExplicitLetList> ell = ExplicitLetList::make(f->body);
    const std::vector<Var>& vars = ell->vars;
    const std::vector<Expr>& exprs = ell->exprs;
    size_t n = vars.size();

    for (size_t i = 0; i < n; i++) {
      Var let_var = vars[i];
      Expr expr = exprs[i];
      // Same here, force revisiting nodes
      this->memo_.erase(let_var);
      this->memo_.erase(expr);

      // We only change the vars on the RHS here, so we don't want it to trigger our routine
      // of handling calls
      Expr new_expr;
      if (auto call_node = expr.as<CallNode>()) {
        new_expr = ExprMutator::VisitExpr_(call_node);
      } else {
        new_expr = VisitExpr(expr);
      }

      auto new_let_var = curr_scope->Push(new_expr);
      var_map_->insert(std::make_pair(let_var, new_let_var));
    }

    // Return the ret var of the called function, it will be assigned to the let
    // var associated with the original call node
    Expr ret;
    if (n > 0) {
      ret = VisitExpr(ell->ret);
    } else {
      // Although we assume the IR is ANF, it is possible to have a function like:
      // fn (%in) { %in; }, which is treat as a special case of ANF.
      ret = VisitExpr(f->body);
    }
    // Change the var map back
    var_map_ = tmp;
    return Downcast<Var>(ret);
  }

  void DebugDumpVarMap() {
    LOG(INFO) << "Current var map: ";
    for (auto pair : *(var_map_.get())) {
      LOG(INFO) << pair.first << " -> " << pair.second;
    }
  }
  /*! \brief The module we are operating on. */
  const IRModule& mod_;
  /*! \brief The function we are operating on. */
  const Function& func_;
  /*! \brief The map from local vars to global vars. This one map contains information about all
   *  functions in the module. */
  const LocalVarGVarMap& local_var_to_gvar_;
  /*!
   *  \brief A map to store var mapping after inlining functions. The map stores mapping from
   *  vars in the original function to the returned vars of the inlined functions. For example,
   *    let a = call(foo, ...)    -->         let ...
   *                                          let ...
   *                                          let b = <the stmt before ret in foo>
   *  then the map will have an entry (a, b).
   */
  std::shared_ptr<VarMap> var_map_;
  /*! \brief The scope stack of the let list. */
  std::vector<std::unique_ptr<LetList>> scopes_;
  /*! \brief The current let var. */
  Var curr_let_;
};

/*!
 * \brief Top-level function of this inlining pass. It does the following:
 * 1. Construct the call graph. Currently we skip this pass when we find any loop in the call graph.
 * 2. Process functions according to reverse topological order and inline everything.
 */
IRModule Inline(const IRModule& mod) {
  CallGraphConstructor cgc;
  CallGraph cg;
  bool construction_success;
  std::tie(cg, construction_success) = cgc.ConstructCallGraph(mod);
  if (!construction_success) {
    LOG(WARNING) << "Call graph cannot be constructed, skip inlining pass. ";
    return mod;
  }
  CallGraphNodeList funcs;
  bool is_acyclic;
  std::tie(funcs, is_acyclic) = cg.ReverseTopologicalSortAndCheckCycle();
  if (!is_acyclic) {
    LOG(WARNING) << "Call graph is cyclic, skip inlining pass.";
    return mod;
  }

  auto localvar2funcs = cgc.GetLocalVarFuncMap();
  for (auto f : funcs) {
    // Only process functions that call other functions
    if ((f->callees).size() > 0) {
      auto new_func = FullInliner(mod, f->func, localvar2funcs).Run();
      mod->Update(f->gvar, new_func);
    }
  }

  // Remove all functions except the entry function
  // If the control flow reaches here it means we don't have recursion so this should be safe
  for (auto f : funcs) {
    if (f->RefCount() > 0) mod->Remove(f->gvar);
  }
  return mod;
}

}  // namespace full_inline

Pass FullInline() {
  auto inline_pass = CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) { return full_inline::Inline(mod); }, 0,
      "FullInline", {});
  // Run LambdaLift, FullInline, and DCE in a pass sequence to avoid misuse of this pass.
  return RAFSequential({LambdaLift(), inline_pass, DeadCodeElimination()}, "FullInline");
}

RAF_REGISTER_GLOBAL("raf.pass_.FullInline").set_body_typed(FullInline);

}  // namespace pass
}  // namespace raf
