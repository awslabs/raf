/*!
 * Copyright (c) 2019 by Contributors
 * \file extract_binding.cc
 * \brief Extracting a relay body from frontend defined binding
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/binding.h"

namespace mnm {
namespace pass {
namespace rename_vars {

using namespace mnm::ir;

struct RenameVarsMutator : public ExprMutator {
 public:
  explicit RenameVarsMutator(const Map<String, Var>& named_vars) {
    for (const auto& iter : named_vars) {
      var_map.Set(iter.second, mnm::ir::Var(iter.first, iter.second->type_annotation));
    }
  }

  Expr VisitExpr_(const VarNode* node) final {
    return var_map.at(GetRef<Var>(node));
  }

  Expr VisitExpr_(const LetNode* node) final {
    const Var& var = node->var;
    CHECK_EQ(var_map.count(var), 0) << "IR is malformed: cannot bind var twice";
    Var new_var = mnm::ir::Var("a" + std::to_string(++num_bound_var), var->type_annotation);
    var_map.Set(var, new_var);
    return mnm::ir::Let(new_var, Mutate(node->value), Mutate(node->body));
  }

  int num_bound_var = 0;
  Map<Var, Expr> var_map;
};

Expr RenameVars(Expr expr, Map<String, Var> named_vars) {
  return RenameVarsMutator(named_vars).Mutate(expr);
}

MNM_REGISTER_GLOBAL("mnm.pass_.RenameVars").set_body_typed(RenameVars);
}  // namespace rename_vars
}  // namespace pass
}  // namespace mnm
