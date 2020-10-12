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

using namespace mnm::ir;

struct ExplicitLetList {
 public:
  std::vector<Var> vars;
  std::vector<Expr> exprs;
  Var ret;

  Expr AsExpr() {
    CHECK_EQ(vars.size(), exprs.size());
    Expr body = ret;
    int n = exprs.size();
    for (int i = n - 1; i >= 0; --i) {
      body = Let(vars[i], exprs[i], body);
    }
    return body;
  }

  static std::unique_ptr<ExplicitLetList> make(const Expr& node) {
    std::unique_ptr<ExplicitLetList> ell = std::make_unique<ExplicitLetList>();
    Maker(ell.get()).VisitExpr(node);
    return ell;
  }

  struct Maker : public ExprVisitor {
    explicit Maker(ExplicitLetList* ell) : ell(ell) {
    }
    void VisitExpr_(const LetNode* node) final {
      ell->vars.push_back(node->var);
      ell->exprs.push_back(node->value);
      const Expr& expr = node->body;
      if (expr->IsInstance<LetNode>()) {
        ExprVisitor::VisitExpr(expr);  // tail call
      } else if (expr->IsInstance<VarNode>()) {
        ell->ret = Downcast<Var>(expr);
      } else {
        LOG(FATAL) << "ValueError: DataParallel pass assumes ANF";
        throw;
      }
    }
    ExplicitLetList* ell;
  };
};

};  // namespace pass
};  // namespace mnm
