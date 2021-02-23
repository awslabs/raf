/*!
 * Copyright (c) 2021 by Contributors
 * \file src/pass/inline_let.cc
 * \brief Inline the Let stmt when a variable is assigned to another variable or a TupleGetItem
 * expression that can be simplified.
 */
#include <unordered_map>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "mnm/binding.h"

namespace mnm {
namespace pass {
namespace inline_let {

using namespace mnm::ir;
using namespace mnm::op;

class LetInliner : public ExprMutator {
 public:
  LetInliner() {
  }

  Expr VisitExpr_(const LetNode* let) {
    if (let->value->IsInstance<TupleNode>()) {
      tuple_map_.emplace(let->var, Downcast<Tuple>(let->value));
    }
    auto new_value = VisitExpr(let->value);
    bool alias = false;
    if (new_value->IsInstance<VarNode>()) {
      auto alias_var = Downcast<Var>(new_value);
      alias_map_.emplace(let->var.get(), alias_var);
      alias = true;
    }
    auto new_body = VisitExpr(let->body);
    if (alias) {
      return new_body;
    }
    return Let(let->var, new_value, new_body);
  }

  Expr VisitExpr_(const TupleGetItemNode* node) {
    auto tup = VisitExpr(node->tuple);
    auto it = tuple_map_.find(tup);
    if (it != tuple_map_.end()) {
      Tuple tuple = it->second;
      return tuple->fields[node->index];
    }
    return ExprMutator::VisitExpr_(node);
  }

  Expr VisitExpr_(const VarNode* var) {
    auto it = alias_map_.find(var);
    if (it != alias_map_.end()) {
      return it->second;
    }
    return GetRef<Var>(var);
  }

 private:
  /*! \brief Mapping of a var to a tuple. */
  std::unordered_map<Expr, Tuple, ObjectPtrHash, ObjectPtrEqual> tuple_map_;
  /*! \brief Mapping from a var to another var. */
  std::unordered_map<const VarNode*, Var> alias_map_;
};

}  // namespace inline_let

ir::Expr InlineLet(ir::Expr expr) {
  return inline_let::LetInliner().VisitExpr(expr);
}

MNM_REGISTER_GLOBAL("mnm.pass_.InlineLet").set_body_typed(InlineLet);

}  // namespace pass
}  // namespace mnm
