/*!
 * Copyright (c) 2019 by Contributors
 * \file unbind_constants.cc
 * \brief Make all constants bind to no name
 */
#include "mnm/registry.h"
#include "mnm/ir.h"
#include "mnm/pass.h"

namespace mnm {
namespace pass {
namespace unbind_constants {

using namespace mnm::ir;

struct ConstUnbinder : public ExprMutator {
 public:
  explicit ConstUnbinder(const Map<std::string, Var>& named_vars) {
    for (const auto& iter : named_vars) {
      var_map.Set(iter.second, VarNode::make(iter.first, {}));
    }
  }

  Expr VisitExpr_(const VarNode* node) final {
    return var_map.at(GetRef<Var>(node));
  }

  Expr VisitExpr_(const LetNode* node) final {
    const Var& var = node->var;
    CHECK_EQ(var_map.count(var), 0) << "IR is malformed: cannot bind var twice";
    if (node->value->IsInstance<RelayConstantNode>()) {
      var_map.Set(var, node->value);
      return Mutate(node->body);
    }
    var_map.Set(var, var);
    return LetNode::make(var, Mutate(node->value), Mutate(node->body));
  }

  Map<Var, Expr> var_map;
};

Expr UnbindConstants(Expr func, Map<std::string, Var> named_vars) {
  return ConstUnbinder(named_vars).Mutate(func);
}

MNM_REGISTER_GLOBAL("mnm.pass_.UnbindConstants").set_body_typed(UnbindConstants);

}  // namespace unbind_constants
}  // namespace pass
}  // namespace mnm
