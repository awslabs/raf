/*!
 * Copyright (c) 2020 by Contributors
 * \file inplace_update.cc
 * \brief inplace array updates
 */
#include <vector>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "./let_list.h"
#include "./common.h"

namespace mnm {
namespace pass {
namespace inplace_update {

using namespace mnm::ir;
using namespace mnm::op;

class InplaceVisitor : public ExprFunctor<void(const Expr& n)> {
 public:
  void VisitExprDefault_(const Object* op) override {
  }

  void VisitExpr_(const TupleGetItemNode* node) override {
    Expr e = binding_[simplify_[Downcast<Var>(node->tuple)]];
    if (const auto* tn = e.as<TupleNode>()) {
      simplify_.Set(let_var_, Downcast<Var>(tn->fields[node->index]));
    }
  }

  void VisitExpr_(const VarNode* node) override {
    simplify_.Set(let_var_, GetRef<Var>(node));
  }

  void operator()(const Function& func) {
    const static Op op = Op::Get("mnm.op._alloc_storage");
    std::unique_ptr<ExplicitLetList> ell_{ExplicitLetList::make(func->body)};
    const auto& vars = ell_->vars;
    const auto& exprs = ell_->exprs;
    CHECK_EQ(vars.size(), exprs.size());
    int n = exprs.size();
    for (const auto& var : func->params) {
      simplify_.Set(var, var);
      binding_.Set(var, var);
    }
    for (int i = 0; i < n; ++i) {
      let_var_ = vars[i];
      simplify_.Set(let_var_, let_var_);
      binding_.Set(let_var_, exprs[i]);
      VisitExpr(exprs[i]);
    }
    for (int i = 0; i < n; ++i) {
      const auto* var = static_cast<const ExtendedVarNode*>(vars[i].as<VarNode>());
      if (var->may_share.defined()) {
        Expr e = binding_[simplify_[vars[i]]];
        if (const auto* cn = e.as<CallNode>()) {
          vmap.Set(simplify_[vars[i]], var->may_share);
          auto arg = Downcast<Var>(cn->args[0]);
          auto alloc_storage = Downcast<Call>(binding_[simplify_[arg]]);
          CHECK_EQ(alloc_storage->op, op);
          vmap.Set(arg, Var());
        }
      }
    }
  }

  /*! \brief the variables to be replaced */
  Map<Var, Var> vmap;

 private:
  /*! \brief a variable that is set for each let expr */
  Var let_var_;
  /*! \brief the let binding for each variable */
  Map<Var, Expr> binding_;
  /*! \brief simplify TupleGetItem((a_0, a_1, .., a_n), i) to a_i */
  Map<Var, Var> simplify_;
};

class InplaceRewriter : public ExprMutator {
 public:
  Expr VisitExpr_(const LetNode* node) override {
    // record the origin implementation here:
    // if (visitor_.vmap.count(node->var) > 0) {
    //   return VisitExpr(node->body);
    // }
    // return ExprMutator::VisitExpr_(node);
    auto pre_visit = [this](const LetNode* op) {
      if (visitor_.vmap.count(op->var) == 0) {
        this->VisitExpr(op->var);
        this->VisitExpr(op->value);
      }
    };
    auto post_visit = [this](const LetNode* op) {
      auto expr = GetRef<Expr>(op);
      Expr body = this->VisitExpr(op->body);
      if (visitor_.vmap.count(op->var) > 0) {
        this->memo_[expr] = body;
      } else {
        Var var = Downcast<Var>(this->VisitExpr(op->var));
        Expr value = this->VisitExpr(op->value);
        if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
          this->memo_[expr] = expr;
        } else {
          this->memo_[expr] = Let(var, value, body);
        }
      }
    };
    ExpandANormalForm(node, pre_visit, post_visit);
    return memo_[GetRef<Expr>(node)];
  }

  Expr VisitExpr_(const VarNode* node) override {
    auto k = GetRef<Var>(node);
    if (visitor_.vmap.count(k) > 0) {
      Var v = visitor_.vmap[k];
      if (v.defined()) {
        return v;
      } else {
        LOG(FATAL) << "variable " << k
                   << "represents alloc_storage, which should not be used outside alloc_tensor";
      }
    }
    return k;
  }

  Expr operator()(const Function& func) {
    visitor_(func);
    Expr body = VisitExpr(func->body);
    return Function(func->params, body, func->ret_type, func->type_params, func->attrs);
  }

 private:
  /*! \brief the visitor */
  InplaceVisitor visitor_;
};

}  // namespace inplace_update

ir::IRModule InplaceUpdate(ir::IRModule mod) {
  tvm::Map<ir::GlobalVar, ir::BaseFunc> functions;
  for (auto& kv : mod->functions) {
    functions.Set(kv.first, tvm::Downcast<Function>(
                                inplace_update::InplaceRewriter()(Downcast<Function>(kv.second))));
  }
  return ir::IRModule(functions);
}

MNM_REGISTER_GLOBAL("mnm.pass_.InplaceUpdate").set_body_typed(InplaceUpdate);

}  // namespace pass
}  // namespace mnm
