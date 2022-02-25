/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file bind.cc
 * \brief substitue variables in an expression
 */
#include "raf/op.h"
#include "raf/ir.h"
#include "support/arena.h"

namespace raf {
namespace pass {
namespace bind {

using namespace raf::ir;
using namespace raf::op;

class SubstitueMutator : public ExprMutator {
 public:
  explicit SubstitueMutator(const tvm::Map<Var, Expr>& args_map) : args_map_(args_map) {
  }

  Expr VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      CHECK(!args_map_.count(op->var)) << "Cannot bind an internel variable in let";
      const auto* var = static_cast<const ExtendedVarNode*>(op->var.get());
      if (var->may_share.defined()) {
        Expr may_share = VisitExpr(var->may_share);
        const auto* msv = may_share.as<VarNode>();
        Var new_var =
            MakeVar(var->name_hint(), var->type_annotation, msv ? GetRef<Var>(msv) : Var());
        args_map_.Set(op->var, new_var);
      } else {
        this->VisitExpr(op->var);
      }
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      auto expr = GetRef<Expr>(op);
      const auto* extended_var = static_cast<const ExtendedVarNode*>(op->var.get());
      Var var = Downcast<Var>(this->VisitExpr(op->var));
      auto value = this->VisitExpr(op->value);
      auto body = this->VisitExpr(op->body);
      if (extended_var->may_share.defined()) {
        this->memo_[expr] = Let(Downcast<Var>(args_map_[op->var]), value, body);
      } else {
        if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
          this->memo_[expr] = expr;
        } else {
          this->memo_[expr] = Let(var, value, body);
        }
      }
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    for (Var param : op->params) {
      ICHECK(!args_map_.count(param)) << "Cannnot bind an internal function parameter";
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const VarNode* op) final {
    auto id = GetRef<Var>(op);
    auto it = args_map_.find(id);
    if (it != args_map_.end()) {
      return (*it).second;
    } else {
      return std::move(id);
    }
  }

 private:
  /*! \brief substitute var with expr */
  tvm::Map<Var, Expr> args_map_;
};

class ExprAppendMutator : public ExprMutator {
 public:
  explicit ExprAppendMutator(const Expr& second) : second_(second) {
  }

  Expr VisitExpr_(const LetNode* op) final {
    static auto pre_visit = [this](const LetNode* op) {
      // empty body
    };
    auto post_visit = [this](const LetNode* op) {
      if (op->body.as<VarNode>()) {
        this->memo_[GetRef<Expr>(op)] = Let(op->var, op->value, second_);
      } else {
        Expr body = this->VisitExpr(op->body);
        this->memo_[GetRef<Expr>(op)] = Let(op->var, op->value, body);
      }
    };
    tvm::relay::ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

 private:
  /*! \brief expr to be appended */
  const Expr& second_;
};

}  // namespace bind

ir::Expr Substitute(ir::Expr expr, const tvm::Map<ir::Var, ir::Expr>& args_map) {
  return bind::SubstitueMutator(args_map).VisitExpr(expr);
}

RAF_REGISTER_GLOBAL("raf.pass_.Substitute").set_body_typed(Substitute);

ir::Expr ExprAppend(const ir::Expr& first, const ir::Expr& second) {
  return bind::ExprAppendMutator(second).VisitExpr(first);
}

RAF_REGISTER_GLOBAL("raf.pass_.ExprAppend").set_body_typed(ExprAppend);

}  // namespace pass
}  // namespace raf
