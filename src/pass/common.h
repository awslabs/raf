/*!
 * Copyright (c) 2020 by Contributors
 * \file common.h
 * \brief common utilities
 */
#pragma once

#include <vector>
#include "mnm/ir.h"

namespace mnm {
namespace pass {

struct ExplicitLetList {
 public:
  std::vector<ir::Var> vars;
  std::vector<ir::Expr> exprs;
  ir::Var ret;

  ir::Expr AsExpr() {
    CHECK_EQ(vars.size(), exprs.size());
    ir::Expr body = ret;
    int n = exprs.size();
    for (int i = n - 1; i >= 0; --i) {
      body = ir::Let(vars[i], exprs[i], body);
    }
    return body;
  }

  static std::unique_ptr<ExplicitLetList> make(const ir::Expr& node) {
    std::unique_ptr<ExplicitLetList> ell = std::make_unique<ExplicitLetList>();
    Maker(ell.get()).VisitExpr(node);
    return ell;
  }

  struct Maker : public ir::ExprVisitor {
    explicit Maker(ExplicitLetList* ell) : ell(ell) {
    }

    void VisitExpr_(const ir::VarNode* node) final {
      ell->ret = ir::GetRef<ir::Var>(node);
    }

    void VisitExpr_(const ir::LetNode* node) final {
      ell->vars.push_back(node->var);
      ell->exprs.push_back(node->value);
      const ir::Expr& expr = node->body;
      if (expr->IsInstance<ir::LetNode>()) {
        VisitExpr(expr);  // tail call
      } else if (expr->IsInstance<ir::VarNode>()) {
        VisitExpr(expr);
      } else {
        LOG(FATAL) << "ValueError: assumes ANF";
        throw;
      }
    }
    ExplicitLetList* ell;
  };
};

};  // namespace pass
};  // namespace mnm
