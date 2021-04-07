/*!
 * Copyright (c) 2021 by Contributors
 * \file src/pass/to_a_normal_form.cc
 * \brief Convert dataflow graph to A-normal form.
 */
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "./convert_utils.h"

namespace mnm {
namespace pass {

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
  DependencyGraph dg = DependencyGraph::Create(&arena, expr);
  /* The scope of the whole expr is global.
   * The scope of any subexpr, is the lowest common ancestor of all incoming edge.
   * We also record the set of expressions whose scope is lifted.
   */
  std::pair<NodeScopeMap, ExprSet> scopes = CalcScope(dg);
  return Fill::ToBasicBlockNormalForm(expr, dg, &scopes.first, &scopes.second);
}

IRModule ToBasicBlockNormalForm(IRModule m) {
  tvm::Map<GlobalVar, BaseFunc> updates;
  auto funcs = m->functions;
  for (const auto& it : funcs) {
    ICHECK_EQ(FreeVars(it.second).size(), 0);
    if (const auto* n = it.second.as<FunctionNode>()) {
      if (n->GetAttr<String>(tvm::relay::attr::kCompiler).defined()) continue;
    }
    Expr ret = TransformF([&](const Expr& e) { return ToBasicBlockNormalFormExpr(e); },
                          Downcast<Function>(it.second));
    ICHECK_EQ(FreeVars(ret).size(), 0)
        << AsText(ret) << "should not has free vars: " << FreeVars(ret);
    updates.Set(it.first, Downcast<Function>(ret));
  }

  for (auto pair : updates) {
    m->Add(pair.first, pair.second, true);
  }

  DLOG(INFO) << "ToBBNF: transformed" << std::endl << m;

  return m;
}

MNM_REGISTER_GLOBAL("mnm.pass_.ToBasicBlockNormalForm").set_body_typed(ToBasicBlockNormalForm);

}  // namespace pass
}  // namespace mnm
