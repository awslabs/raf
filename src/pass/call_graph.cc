/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/call_graph.cc
 * \brief A simple call graph implementation for function inlining. This call
 * graph only captures the relationship between global functions, so LambdaLift()
 * should be run before using this call graph.
 */

#include "./call_graph.h"

namespace raf {
namespace pass {
namespace call_graph {

// Utilities of the CallGraph class
std::pair<CallGraphNodeList, bool> CallGraph::ReverseTopologicalSortAndCheckCycle() {
  CallGraphNodeList sorted_funcs;
  CallGraphNode* start_node = nullptr;

  // Find the entry functions, insert them into the list directly
  // We will start with these functions when traversing the call graph
  size_t entry_node_cnt = 0;
  for (auto it : funcs_) {
    CallGraphNode* cg_node = it.second;
    if (cg_node->RefCount() == 0) {
      start_node = cg_node;
      entry_node_cnt++;
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
        num_unvisited++;
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

// Utilities of the CallGraphConstructor class
std::pair<CallGraph, bool> CallGraphConstructor::ConstructCallGraph(const IRModule& mod) {
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
    bool success = AnalyzeFunc(cg_node);
    // If analysis of any function fails, return an empty call graph and a failure indicator
    if (!success) {
      return std::pair<CallGraph, bool>(CallGraph(), false);
    }
  }
  // Put the nodes into the call graph and return it
  CallGraph cg;
  for (auto it : funcs_) {
    cg.AddNode(it.first, it.second);
  }
  return std::pair<CallGraph, bool>(cg, true);
}

bool CallGraphConstructor::AnalyzeFunc(CallGraphNode* cg_node) {
  auto func = cg_node->func;
  auto ell = ExplicitLetList::make(func->body);
  const std::vector<Var>& vars = ell->vars;
  const std::vector<Expr>& exprs = ell->exprs;
  size_t n = vars.size();

  for (size_t i = 0; i < n; i++) {
    Var let_var = vars[i];
    Expr expr = exprs[i];
    if (auto gvar_node = expr.as<GlobalVarNode>()) {
      // Case 1: the expr is a global var which may refer to a function
      auto gvar = GetRef<GlobalVar>(gvar_node);
      if (funcs_.count(gvar)) local_var_to_funcs_[let_var] = gvar;
    } else if (auto call_node = expr.as<CallNode>()) {
      // Case 2: call node, check if it is calling a global function
      auto op = call_node->op;
      if (auto gvar_node = op.as<GlobalVarNode>()) {
        // Case 2.1: the call node is directly calling the global function
        auto gvar = GetRef<GlobalVar>(gvar_node);
        CHECK_GT(funcs_.count(gvar), 0)
            << "The called global function " << gvar << " cannot be found!";
        (cg_node->callees).insert(funcs_[gvar]);
        funcs_[gvar]->callers.insert(cg_node);
      } else if (auto var_node = op.as<VarNode>()) {
        // Case 2.2: the call node is calling a local var, which may point to a global var
        auto local_var = GetRef<Var>(var_node);
        // The local var points to a global function if and only if it is stored in
        // the local var-func map
        if (local_var_to_funcs_.count(local_var)) {
          CallGraphNode* node = funcs_[local_var_to_funcs_[local_var]];
          (cg_node->callees).insert(node);
          node->callers.insert(cg_node);
        } else {
          // We are not handling tuples and more than one levels of direct-assign for now.
          LOG(WARNING)
              << "The called local var does not seem to correspond to a global function! "
              << "Notice that this pass currently does not handle cases where the called op "
              << "is generated by a direct assign or TGI node. ";
          return false;
        }
      }
    }
  }
  return true;
}

}  // namespace call_graph
}  // namespace pass
}  // namespace raf
