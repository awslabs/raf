/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file replace_op.cc
 * \brief replace ops for training
 * TODO(@zhen-jia) we will refactor this pass to support more ops like batch_norm.
 */
#include "raf/pass.h"

#include "./common.h"
#include "./let_list.h"
#include "../common/shape_utils.h"

namespace raf {
namespace pass {
namespace switch_train {

class OpReplacer : public ExprMutator {
 public:
  OpReplacer(const Function& func) : func_(func) {
    auto ell = ExplicitLetList::make(func->body);
    for (size_t i = 0; i < ell->vars.size(); ++i) {
      if (IsLayerNormCall(ell->exprs[i])) {
        ln_map_.Set(ell->vars[i], Expr());
      }
    }

    scopes_.emplace_back(new LetList);
  }

  Function Replace() {
    if (ln_map_.empty()) {
      return func_;
    }

    Array<Var> func_params{func_->params};
    auto new_body = this->Mutate(func_->body);
    return Function(func_params, new_body, func_->ret_type, {}, func_->attrs);
  }

  Expr VisitExpr_(const LetNode* node) override {
    scopes_.emplace_back(new LetList);
    auto scope = scopes_.back().get();
    Expr body;
    do {
      auto curr_var = node->var;
      auto value = VisitExpr(node->value);

      if (ln_map_.count(curr_var) > 0) {
        auto new_var = ToApex(scope, curr_var, value);
        ln_map_.Set(curr_var, new_var);
      } else {
        scope->Push(curr_var, value);
      }

      body = node->body;
      node = body.as<LetNode>();
    } while (node);
    auto ret = scopes_.back()->Get(this->Mutate(body));
    scopes_.pop_back();
    return ret;
  }
  Expr VisitExpr_(const CallNode* node) override {
    const Expr& callee = node->op;
    Array<Expr> args;
    for (int i = 0; i < node->args.size(); ++i) {
      auto arg_var = node->args[i];
      if (arg_var->IsInstance<VarNode>() && ln_map_.count(Downcast<Var>(arg_var))) {
        args.push_back(ln_map_[Downcast<Var>(arg_var)]);
      } else {
        args.push_back(arg_var);
      }
    }
    return Call(node->op, args);
  }
  Expr VisitExpr_(const TupleNode* node) override {
    Array<Expr> fields;
    for (auto field : node->fields) {
      auto field_var = field;
      if (field_var->IsInstance<VarNode>() && ln_map_.count(Downcast<Var>(field))) {
        fields.push_back(ln_map_[Downcast<Var>(field)]);
      } else {
        fields.push_back(field);
      }
    }
    return Tuple(fields, node->span);
  }

 private:
  inline bool IsLayerNormCall(const Expr& expr) {
    static const Op& layer_norm_op = Op::Get("mnm.op.layer_norm");
    if (!expr->IsInstance<CallNode>()) {
      return false;
    }
    auto call = Downcast<Call>(expr);
    if (auto node = call->op.as<OpNode>()) {
      return GetRef<Op>(node) == layer_norm_op;
    }
    return false;
  }
  /*! \brief Return the n'th argument of the given call expr. */
  inline Expr GetNArg(const Expr& expr, int n) {
    CHECK(expr->IsInstance<CallNode>());
    auto call = Downcast<Call>(expr);
    CHECK_GE(call->args.size(), n)
        << "Expected at least " << n << " argument, but got " << raf::ir::AsText(expr);
    return call->args[n];
  }

  Var ToApex(LetList* scope, const Var& var, const Expr& value) {
    static const Op& ln_op = Op::Get("raf.op.layer_norm_train");
    auto x = Downcast<Var>(GetNArg(value, 0));
    auto scale = Downcast<Var>(GetNArg(value, 1));
    auto bias = Downcast<Var>(GetNArg(value, 2));
    auto axis = Downcast<Constant>(GetNArg(value, 3));
    auto eps = Downcast<Constant>(GetNArg(value, 4));

    auto ln_var = scope->Push(Call(ln_op, {x, scale, bias, axis, eps}));
    return scope->Push(TupleGetItem(ln_var, 0));
    // return ln_replace;
  }
  /*! \brief The scope stack of the let list. */
  std::vector<std::unique_ptr<LetList>> scopes_;
  Function func_;

  Map<Var, Expr> ln_map_;
  Var grad_tuple_var_;
};

}  // namespace switch_train

Pass ReplaceOP() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return switch_train::OpReplacer(f).Replace(); };
  auto switch_train = CreateRAFFunctionPass(pass_func, 0, "SwithTrainOp", {});
  return RAFSequential({switch_train}, "SwithTrainOp");
}

RAF_REGISTER_GLOBAL("raf.pass_.ReplaceOP").set_body_typed(ReplaceOP);

}  // namespace pass
}  // namespace raf
