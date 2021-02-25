/*!
 * Copyright (c) 2020 by Contributors
 * \file bind.cc
 * \brief substitue variables in an expression
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/binding.h"
#include "support/arena.h"

namespace mnm {
namespace pass {
namespace bind {

using namespace mnm::ir;
using namespace mnm::op;

class SubstitueMutator : public ExprMutator {
 public:
  explicit SubstitueMutator(const tvm::Map<Var, Expr>& args_map) : args_map_(args_map) {
  }

  Expr VisitExpr_(const LetNode* op) final {
    CHECK(!args_map_.count(op->var)) << "Cannot bind an internel variable in let";
    const auto* var = static_cast<const ExtendedVarNode*>(op->var.get());
    if (var->may_share.defined()) {
      Expr may_share = VisitExpr(var->may_share);
      const auto* msv = may_share.as<VarNode>();
      var->may_share = msv ? GetRef<Var>(msv) : Var();
    }
    return ExprMutator::VisitExpr_(op);
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
  const tvm::Map<Var, Expr>& args_map_;
};

}  // namespace bind

ir::Expr Substitute(ir::Expr expr, const tvm::Map<ir::Var, ir::Expr>& args_map) {
  return bind::SubstitueMutator(args_map).VisitExpr(expr);
}

MNM_REGISTER_GLOBAL("mnm.pass_.Substitute").set_body_typed(Substitute);

}  // namespace pass
}  // namespace mnm
