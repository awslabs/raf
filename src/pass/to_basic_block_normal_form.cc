/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/to_a_normal_form.cc
 * \brief Convert dataflow graph to A-normal form.
 */
#include <vector>
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/analysis.h"
#include "./convert_utils.h"

namespace raf {
namespace pass {

using raf::analysis::CreateDependencyGraph;
using tvm::relay::CalcScope;

// For basic block normal form, bind expressions only if the original expression's scope
// should be lifted
Expr Fill::ToBasicBlockNormalForm(const Expr& e, const DependencyGraph& dg,
                                  NodeScopeMap* node_scope, ExprSet* lifted) {
  Fill fi(dg, node_scope, lifted);
  auto var = fi.VisitExpr(e);
  return fi.GetScope(e)->let_list->Get(var);
}

Expr ToBasicBlockNormalFormExpr(const Expr& expr) {
  // calculate all the dependency between nodes.
  tvm::support::Arena arena;
  DependencyGraph dg = CreateDependencyGraph(&arena, expr, false);
  /* The scope of the whole expr is global.
   * The scope of any subexpr, is the lowest common ancestor of all incoming edge.
   * We also record the set of expressions whose scope is lifted.
   */
  std::pair<NodeScopeMap, ExprSet> scopes = CalcScope(dg);
  return Fill::ToBasicBlockNormalForm(expr, dg, &scopes.first, &scopes.second);
}

Pass ToBasicBlockNormalForm() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    ICHECK_EQ(FreeVars(f).size(), 0);
    Expr ret = TransformF([&](const Expr& e) { return ToBasicBlockNormalFormExpr(e); }, f);
    ICHECK_EQ(FreeVars(ret).size(), 0)
        << ir::AsText(ret) << "should not has free vars: " << FreeVars(ret);
    return Downcast<Function>(ret);
  };
  return CreateRAFFunctionPass(pass_func, 1, "ToBasicBlockNormalForm", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.ToBasicBlockNormalForm").set_body_typed(ToBasicBlockNormalForm);

}  // namespace pass
}  // namespace raf
