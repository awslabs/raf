/*!
 * Copyright (c) 2020 by Contributors
 * \file flatten_let.cc
 * \brief remove nested let expressions
 */
#include <vector>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "./let_list.h"
#include "./common.h"

/*!
 * before flattening:
 * let v = (let a = p1 + p2, let b = a + p3, b)
 * let ret = v + p4
 * ret
 *
 * after flattening:
 * let a = p1 + p2
 * let v = a + p3
 * let ret = v + p4
 * ret
 */

namespace mnm {
namespace pass {
namespace flatten_let {

using namespace mnm::ir;
using namespace mnm::op;

class LetFlattener : public ExprMutator {
 public:
  Expr VisitExpr_(const VarNode* node) final {
    Var var = GetRef<Var>(node);
    if (vmap_.find(var) != vmap_.end()) {
      return vmap_.at(var);
    }
    return var;
  }

  Expr VisitExpr_(const LetNode* node) final {
    if (node->value.as<LetNode>()) {
      let_var_rec_.push_back(node->var);
      let_body_rec_.push_back(node->body);
      return VisitExpr(node->value);
    }
    Expr value = VisitExpr(node->value);
    return Let(node->var, value, VisitLetBody(node->body));
  }

  Expr VisitLetBody(const Expr& body) {
    if (body.as<LetNode>()) {
      return VisitExpr(body);
    }
    Var var = Downcast<Var>(body);
    if (let_var_rec_.empty()) {
      CHECK(let_body_rec_.empty());
      return VisitExpr(var);
    }
    CHECK(!let_body_rec_.empty());
    vmap_[let_var_rec_.back()] = var;
    let_var_rec_.pop_back();
    Expr next_body = let_body_rec_.back();
    let_body_rec_.pop_back();
    return VisitLetBody(next_body);
  }

 private:
  /*! \brief the var of let whose value is let */
  std::vector<Var> let_var_rec_;
  /*! \brief the body of let whose value is let */
  std::vector<Expr> let_body_rec_;
  /*! \brief replace key to value */
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> vmap_;
};

}  // namespace flatten_let

ir::Module FlattenLet(ir::Module mod) {
  tvm::Map<ir::GlobalVar, ir::Function> functions;
  for (auto& kv : mod->functions) {
    functions.Set(kv.first, tvm::Downcast<ir::Function>(flatten_let::LetFlattener()(kv.second)));
  }
  return ir::Module::make(functions);
}

MNM_REGISTER_GLOBAL("mnm.pass_.FlattenLet").set_body_typed(FlattenLet);

}  // namespace pass
}  // namespace mnm
