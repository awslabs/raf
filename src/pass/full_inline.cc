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
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/binding.h"
#include "./common.h"
#include "./let_list.h"

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

struct CallGraphNode;
using CallGraphNodeSet = std::unordered_set<CallGraphNode*>;
using CallGraphNodeMap = std::unordered_map<GlobalVar, CallGraphNode*, ObjectPtrHash, ObjectPtrEqual>;
using LocalVarFuncMap = std::unordered_map<Var, CallGraphNode*, ObjectPtrHash, ObjectPtrEqual>;
using CallGraphNodeList = std::vector<CallGraphNode*>;
using VarMap = std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual>;

/*! 
 *  \brief Classes to store the call graph and get some basic information about each function in the
 *  call graph. This is a simplified version of the TVM CallGraph class. We cannot use it directly
 *  because:
 *  - TVM's file structure: we cannot find call_graph.h from our include path and we don't want to
 *    change TVM's file hierarchy. 
 *  - TVM's call graph does not seem to be able to detect loops in the call graph. In such cases we
 *    choose to skip this pass entirely. 
 */

/*! \brief One node of the call graph. Represents one function in the module. */
struct CallGraphNode {
  CallGraphNode(const GlobalVar& gv, const Function& f): gvar(gv), func(f) {}
  /*! \brief The function. */
  Function func;
  /*! \brief The global var associated with the function. */
  GlobalVar gvar;
  /*! \brief Functions that called this function. */
  CallGraphNodeSet callers;
  /*! \brief Functions that are called by this function. */
  CallGraphNodeSet callees;
  /*! \brief Number of functions that called this function. Used to determine whether this function
   *  is an entry function. */
  size_t RefCount() { return callers.size(); }
};

/*! \brief The whole call graph. */
class CallGraph {
 public:
  CallGraph() {}
  /*! \brief Add a node to the call graph. */
  void AddNode(const GlobalVar& gv, CallGraphNode* cg_node) {
    funcs_[gv] = cg_node;
  }
  /*! 
   *  \brief Run topological sort and return a sorted list of functions. The other
   *  field of the returned pair is whether the call graph is acyclic or not. The
   *  returned list is in reverse topological sort (i.e., children first, parent last).
   */
  std::pair<CallGraphNodeList, bool> ReverseTopologicalSortAndCheckCycle() {
    CallGraphNodeList sorted_funcs;
    CallGraphNode* start_node = nullptr;

    // Find the entry functions, insert them into the list directly
    // We will start with these functions when traversing the call graph
    size_t entry_node_cnt = 0;
    for (auto it : funcs_) {
      CallGraphNode* cg_node = it.second;
      if (cg_node->RefCount() == 0) {
        start_node = cg_node;
        entry_node_cnt ++;
      }
    }
    CHECK_EQ(entry_node_cnt, 1UL) << "Not handling more than one entry nodes for now!";

    // Run DFS to detect cycles and do topological sorting at the same time

    // Set of visited nodes
    CallGraphNodeSet visited;
    // Set of nodes on the path
    CallGraphNodeSet path;
    // Stack for DFS, we use the non-recursive implementation
    CallGraphNodeList stack;

    visited.insert(start_node);
    stack.push_back(start_node);
    path.insert(start_node);

    while (!stack.empty()) {
      CallGraphNode* curr_node = stack.back();
      visited.insert(curr_node);
      path.insert(curr_node);

      // Check all children and count the number of unvisited neighbors
      int num_unvisited = 0;
      for (auto child : curr_node->callees) {
        // Push a children onto the stack if it is not visited
        if (!visited.count(child)) {
          stack.push_back(child);
          num_unvisited ++;
        } else if (path.count(child)) {
          // If a child is visited and it's on the path, then we have a cycle
          // in the call graph, return directly
          return std::pair<CallGraphNodeList, bool>(CallGraphNodeList(), false);
        }
      }

      // Only pop a node from stack and add it to our final list when all of its
      // children have been visited
      if (num_unvisited == 0) {
        sorted_funcs.push_back(curr_node);
        stack.pop_back();
        path.erase(curr_node);
      }
    }

    return std::pair<CallGraphNodeList, bool>(sorted_funcs, true);
  }

 private:
  /*! \brief A map from global vars to global functions in the module. */
  CallGraphNodeMap funcs_;
};

/*! \brief Construct the call graph from module. */
class CallGraphConstructor {
 public:
  /*! \brief Construct the call graph from module. */
  CallGraph ConstructCallGraph(const IRModule& mod) {
    // Get all functions, insert the call graph nodes into the set
    auto funcs_in_mod = mod->functions;
    for (auto it : funcs_in_mod) {
      auto gvar = it.first;
      auto fn = it.second;
      if (auto fn_node = fn.as<FunctionNode>()) {
        CallGraphNode* new_fn_node = new CallGraphNode(gvar, GetRef<Function>(fn_node));
        funcs_[gvar] = new_fn_node;
      }
    }
    // Go through each function and link the call graph nodes together
    for (auto it : funcs_) {
      auto gvar = it.first;
      auto cg_node = it.second;
      AnalyzeFunc(cg_node);
    }
    // Put the nodes into the call graph and return it
    CallGraph cg;
    for (auto it : funcs_) {
      cg.AddNode(it.first, it.second);
    }
    return cg;
  }

  // Some interfaces for accessing the two maps constructed by this class, we need them later
  
  /*! \brief Get the mapping from global vars to functions. */
  const CallGraphNodeMap& GetGlobalVarFuncMap() { return funcs_; }

  /*! \brief Get the mapping from local vars to functions. */
  const LocalVarFuncMap& GetLocalVarFuncMap() { return local_var_to_funcs_; }

 private:
  /*! \brief Analyze a function and connect it with its callers/callees in the call graph. */
  void AnalyzeFunc(CallGraphNode* cg_node) {
    auto func = cg_node->func;
    auto ell = ExplicitLetList::make(func->body);
    const std::vector<Var>& vars = ell->vars;
    const std::vector<Expr>& exprs = ell->exprs;
    size_t n = vars.size();

    for (size_t i = 0; i < n; i ++) {
      Var let_var = vars[i];
      Expr expr = exprs[i];
      if (auto gvar_node = expr.as<GlobalVarNode>()) {
        // Case 1: the expr is a global var which may refer to a function
        auto gvar = GetRef<GlobalVar>(gvar_node);
        if (funcs_.count(gvar))
          local_var_to_funcs_[let_var] = funcs_[gvar];
      } else if (auto call_node = expr.as<CallNode>()) {
        // Case 2: call node, check if it is calling a global function
        auto op = call_node->op;
        if (auto gvar_node = op.as<GlobalVarNode>()) {
          // Case 2.1: the call node is directly calling the global function
          auto gvar = GetRef<GlobalVar>(gvar_node);
          CHECK_GT(funcs_.count(gvar), 0) << "The called global function " << gvar << " cannot be found!";
          (cg_node->callees).insert(funcs_[gvar]);
          funcs_[gvar]->callers.insert(cg_node);
        } else if (auto var_node = op.as<VarNode>()) {
          // Case 2.2: the call node is calling a local var, which may point to a global var
          auto local_var = GetRef<Var>(var_node);
          // The local var points to a global function if and only if it is stored in 
          // the local var-func map
          if (local_var_to_funcs_.count(local_var)) {
            (cg_node->callees).insert(local_var_to_funcs_[local_var]);
            local_var_to_funcs_[local_var]->callers.insert(cg_node);
          }
        } 
      }
      // We are not handling tuples and more than one levels of direct-assign for now. 
    }
  }

  /*! \brief A map from global vars to global functions in the module. */
  CallGraphNodeMap funcs_;
  /*! \brief A separate map from local vars to global functions. */
  LocalVarFuncMap local_var_to_funcs_;
};


/*! 
 *  \brief Perform full inlining on a function. The returned function body will
 *  have no calls to global functions. 
 */
class FullInliner: public ExprMutator {
 public: 
  explicit FullInliner(const Function& func, const CallGraphNodeMap& gvar_to_funcs,
                       const LocalVarFuncMap& local_var_to_funcs): 
    func_(func),
    gvar_to_funcs_(gvar_to_funcs),
    local_var_to_funcs_(local_var_to_funcs) {
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

  /*! \brief Handles call nodes and actually inlines functions. Only global functions are inlined. */
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
      CHECK(gvar_to_funcs_.count(gvar)) << "Called global function " << gvar << " is not in the map!";
      Function called_func = (gvar_to_funcs_.find(gvar))->second->func;
      Var ret_var = InlineFunc(called_func, new_args, curr_scope);
      var_map_->insert(std::make_pair(curr_let_, ret_var));
      ret_expr = Downcast<Expr>(ret_var);
    } else if (auto var_node = op.as<VarNode>()) {
      // Case 2: the called function maps to a local var, this can happen in the following case:
      //   let x = %some_global_var;
      //   let y = x(...)
      Var op_var = GetRef<Var>(var_node);
      if (local_var_to_funcs_.count(op_var)) {
        Function called_func = (local_var_to_funcs_.find(op_var))->second->func;
        Var ret_var = InlineFunc(called_func, new_args, curr_scope);
        var_map_->insert(std::make_pair(curr_let_, ret_var));
        ret_expr = Downcast<Expr>(ret_var);
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
  /*! \brief Inline the called function into the current function body. */
  Var InlineFunc(const Function& f, const Array<Expr>& new_args, LetList* curr_scope) {
    // LOG(INFO) << "Inlining function " << ir::AsText(f);
    CHECK_EQ(new_args.size(), f->params.size()) 
      << "The function should have " << f->params.size() << " parameters, but the arg list has"
      << new_args.size() << " elements!";
     
    // A local var map, not to be confused with the one in this class
    // This var map only stores var mappings in the called function
    std::shared_ptr<VarMap> var_map_in_func = std::make_shared<VarMap>();
    for (size_t i = 0; i < new_args.size(); i ++) {
      var_map_in_func->insert(std::make_pair(f->params[i], new_args[i]));
      // LOG(INFO) << "Arg: " << f->params[i] << " -> " << new_args[i];
      // Delete the arguments from the internal memo to force revisiting nodes
      this->memo_.erase(f->params[i]);
    }

    // We want to leverage VisitExpr() to substitute the vars automatically
    // Cheap trick here: let the var_map_ pointer point to the var map we just created, and
    // change it back when we exit this function
    std::shared_ptr<VarMap> tmp = var_map_;
    var_map_ = var_map_in_func;
    // DebugDumpVarMap();

    // Assume the called function is in ANF
    std::unique_ptr<ExplicitLetList> ell = ExplicitLetList::make(f->body);
    const std::vector<Var>& vars = ell->vars;
    const std::vector<Expr>& exprs = ell->exprs;
    size_t n = vars.size();

    for (size_t i = 0; i < n; i ++) {
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

    // Return the ret var of the called function, it will be assigned to the 
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
    // DebugDumpVarMap();
    return Downcast<Var>(ret);
  }

  void DebugDumpVarMap() {
    LOG(INFO) << "Current var map: ";
    for (auto pair : *(var_map_.get())) {
      LOG(INFO) << pair.first << " -> " << pair.second;
    }
  }
  /*! \brief The function we are operating on. */
  const Function& func_;
  /*! \brief The map from global vars to functions. */
  const CallGraphNodeMap& gvar_to_funcs_;
  /*! \brief The map from local vars to functions. This one map contains information about all 
   *  functions in the module. */
  const LocalVarFuncMap& local_var_to_funcs_;
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
  LOG(INFO) << "Inlining module: " << ir::AsText(mod);
  CallGraphConstructor cgc;
  CallGraph cg = cgc.ConstructCallGraph(mod);
  CallGraphNodeList funcs;
  bool is_acyclic;
  std::tie(funcs, is_acyclic) = cg.ReverseTopologicalSortAndCheckCycle();
  if (!is_acyclic) {
    LOG(INFO) << "Call graph is cyclic, skip inlining pass.";
    return mod;
  }

  auto gvar2funcs = cgc.GetGlobalVarFuncMap();
  auto localvar2funcs = cgc.GetLocalVarFuncMap();
  for (auto f : funcs) {
    // Only process functions that call other functions
    if ((f->callees).size() > 0) {
      auto new_func = FullInliner(f->func, gvar2funcs, localvar2funcs).Run();
      mod->Update(f->gvar, new_func);
    }
  }

  // Remove all functions except the entry function
  // If the control flow reaches here it means we don't have recursion so this should be safe
  for (auto f : funcs) {
    if (f->RefCount() > 0) 
      mod->Remove(f->gvar);
  }
  LOG(INFO) << "After inlining: " << ir::AsText(mod);
  return mod;
}

}  // namespace full_inline

Pass FullInline() {
  return CreateModulePass(
    [=](IRModule mod, const PassContext& pass_ctx) {
      return full_inline::Inline(mod);
    }, 
    0, "FullInline", {"LambdaLift"});
}

RAF_REGISTER_GLOBAL("raf.pass_.FullInline").set_body_typed(FullInline);

}  // namespace pass
}  // namespace raf
