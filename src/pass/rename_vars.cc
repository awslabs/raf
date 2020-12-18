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
      const auto* var = iter.second.as<ExtendedVarNode>();
      var_map.Set(iter.second,
                  mnm::ir::MakeVar(iter.first, iter.second->type_annotation, var->may_share));
    }
  }

  Expr VisitExpr_(const FunctionNode* node) final {
    for (const auto& param : node->params) {
      if (var_map.find(param) == var_map.end()) {
        const auto* var = param.as<ExtendedVarNode>();
        var_map.Set(param,
                    mnm::ir::MakeVar(var->name_hint(), var->type_annotation, var->may_share));
      }
    }
    return ExprMutator::VisitExpr_(node);
  }

  Expr VisitExpr_(const VarNode* node) final {
    return var_map.at(GetRef<Var>(node));
  }

  Expr VisitExpr_(const LetNode* node) final {
    const Var& var = node->var;
    CHECK_EQ(var_map.count(var), 0) << "IR is malformed: cannot bind var twice";
    const auto* vn = var.as<ExtendedVarNode>();
    Var may_share = vn->may_share;
    Var new_var =
        mnm::ir::MakeVar("a" + std::to_string(++num_bound_var), var->type_annotation,
                         may_share.defined() ? Downcast<Var>(var_map.at(may_share)) : may_share);
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
