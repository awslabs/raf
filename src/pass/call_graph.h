/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/call_graph.h
 * \brief A simple call graph implementation for function inlining. This call
 * graph only captures the relationship between global functions, so LambdaLift()
 * should be run before using this call graph.
 */

#include <unordered_set>
#include <vector>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include "raf/op.h"
#include "raf/ir_ext.h"
#include "raf/pass.h"
#include "./common.h"

namespace raf {
namespace pass {
namespace call_graph {

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
using CallGraphNodeMap =
    std::unordered_map<GlobalVar, CallGraphNode*, ObjectPtrHash, ObjectPtrEqual>;
using LocalVarGVarMap = std::unordered_map<Var, GlobalVar, ObjectPtrHash, ObjectPtrEqual>;
using CallGraphNodeList = std::vector<CallGraphNode*>;

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
  CallGraphNode(const GlobalVar& gv, const Function& f) : gvar(gv), func(f) {
  }
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
  size_t RefCount() {
    return callers.size();
  }
};

/*! \brief The whole call graph. */
class CallGraph {
 public:
  CallGraph() {
  }
  /*! \brief Add a node to the call graph. */
  void AddNode(const GlobalVar& gv, CallGraphNode* cg_node) {
    funcs_[gv] = cg_node;
  }
  /*!
   * \brief Run topological sort and return a sorted list of functions. The other
   * field of the returned pair is whether the call graph is acyclic or not. The
   * returned list is in reverse topological sort (i.e., children first, parent last).
   */
  std::pair<CallGraphNodeList, bool> ReverseTopologicalSortAndCheckCycle();

 private:
  /*! \brief A map from global vars to global functions in the module. */
  CallGraphNodeMap funcs_;
};

/*! \brief Construct the call graph from module. */
class CallGraphConstructor {
 public:
  /*!
   * \brief Construct the call graph from module. The boolean in the return value
   * indicates whether the construction is successful.
   */
  std::pair<CallGraph, bool> ConstructCallGraph(const IRModule& mod);
  /*! \brief Get the mapping from local vars to functions. */
  const LocalVarGVarMap& GetLocalVarFuncMap() {
    return local_var_to_funcs_;
  }

 private:
  /*!
   * \brief Analyze a function and connect it with its callers/callees in the call graph.
   * Returns true on success and false when analysis fails.
   */
  bool AnalyzeFunc(CallGraphNode* cg_node);
  /*! \brief A map from global vars to global functions in the module. */
  CallGraphNodeMap funcs_;
  /*! \brief A separate map from local vars to global functions. */
  LocalVarGVarMap local_var_to_funcs_;
};

}  // namespace call_graph
}  // namespace pass
}  // namespace raf
