/*!
 * Copyright (c) 2021 by Contributors
 * \file inplace_update.cc
 * \brief Annotate the may_share field in the ExtendedVar to share the memory between inputs and
 * outputs and check the validity of memory sharing.
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "tvm/ir/type_functor.h"
#include "./let_list.h"
#include "./common.h"
#include "./liveness_analysis.h"

namespace mnm {
namespace pass {
namespace inplace_update {

/*!
 * \brief InplaceUpdateMutator marks may_share in the variable for ops that have attr
 * TMNMInplaceUpdate so that inputs and outputs may share the memory and perform inplace update.
 * When binary op `add` and `subtract` has inplace arg set to be true, this pass will also mark the
 * output tensor to share the memory with the 1st input tensor.
 *
 * For example, this pass transforms the op batch_norm_train as follows.
 *   let %x1 = batch_norm_train(%x, %mean, %variance, %weight, %bias, float64(0.1), float64(0.001));
 *   let %x2 = %x1.0;
 *   let %x3 = %x1.1;
 *   let %x4 = %x1.2;
 *   let %x5 = (%x2, %x3, %x4);
 *   %x5
 *
 * The transformed IR will become:
 *   let %x1 = batch_norm_train(%x, %mean, %variance, %weight, %bias, float64(0.1), float64(0.001));
 *   let %x2 = %x1.0;
 *   let %x3(share: %mean) = %x1.1;
 *   let %x4(share: %variance) = %x1.2;
 *   let %x5 = (%x2, %x3, %x4);
 *   %x5
 *
 * Another example for inplace update for the subtract op.
 *   let %x1 = substract(%model.x, %gradient, true);
 *   %x1
 *
 *   let %x1(share: %model.x) = substract(%model.x, %gradient, true);
 *   %x1
 */
class InplaceUpdateMutator : public MixedModeMutator {
 public:
  // Mapping from the output index in the expression to a variable to share memory with
  using ExprShareMap = std::unordered_map<int, Var>;
  // Mapping from the output index to the input index to share memory in fused op functions
  using FusedOpShareMap = std::unordered_map<int, int>;

  Expr VisitExpr_(const VarNode* node) final {
    auto var = GetRef<Var>(node);
    auto it = var_update_map_.find(var);
    if (it != var_update_map_.end()) {
      return it->second;
    }
    return var;
  }

  Expr VisitExpr_(const FunctionNode* func) final {
    if (!func->HasNonzeroAttr(attr::kPrimitive)) {
      return ExprMutator::VisitExpr_(func);
    }
    Array<Var> params;
    for (auto param : func->params) {
      params.push_back(Downcast<Var>(Mutate(param)));
    }
    auto body = Mutate(func->body);
    auto new_func =
        Function(params, body, func->ret_type, func->type_params, func->attrs, func->span);
    auto find = expr_share_map_.find(body);
    if (find != expr_share_map_.end()) {
      auto& expr_map = find->second;
      FusedOpShareMap fuse_map;
      // replace the memory sharing var to the param index in the fused function
      for (auto it : expr_map) {
        int out_idx = it.first;
        Var param = it.second;
        int input_idx = -1;
        for (size_t i = 0; i < params.size(); ++i) {
          if (params[i].same_as(param)) {
            input_idx = i;
            break;
          }
        }
        CHECK_GE(input_idx, 0) << "Cannot find the variable " << param
                               << " in the function parameters";
        fuse_map.emplace(out_idx, input_idx);
      }
      fused_op_share_map_.emplace(new_func, fuse_map);
    }
    return new_func;
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    static auto finplace = Op::GetAttrMap<op::TMNMInplaceUpdate>("TMNMInplaceUpdate");
    static auto add_op = Op::Get("mnm.op.add");
    static auto subtract_op = Op::Get("mnm.op.subtract");
    auto call = Downcast<Call>(post);
    auto op = Mutate(call->op);
    if (op.as<OpNode>()) {
      auto opn = Downcast<Op>(op);
      if (finplace.count(opn)) {
        ExprShareMap share;
        for (auto it : finplace[opn]) {
          auto arg = Downcast<Var>(call->args[it.first]);
          auto var = arg.as<ExtendedVarNode>();
          if (var && var->may_share.defined()) {
            arg = var->may_share;
          }
          share.emplace(it.second, arg);
        }
        expr_share_map_.emplace(post, share);
      } else if (opn == add_op || opn == subtract_op) {
        // Currently only support add and subtract in the binary ops for inplace update.
        // If the inplace arg is true, the output tensor will share memory with the 1st input tensor
        CHECK_GT(pre->args.size(), 2);
        if (auto var = pre->args[2].as<ExtendedVarNode>()) {
          ExprShareMap share{{0, GetRef<Var>(var)}};
          expr_share_map_.emplace(post, share);
        }
      }
    } else if (op.as<FunctionNode>()) {
      auto it = fused_op_share_map_.find(op);
      if (it != fused_op_share_map_.end()) {
        auto& fuse_map = it->second;
        ExprShareMap expr_map;
        for (auto it : fuse_map) {
          expr_map.emplace(it.first, Downcast<Var>(call->args[it.second]));
        }
        expr_share_map_.emplace(post, expr_map);
      }
    }
    return post;
  }

  Expr Rewrite_(const TupleGetItemNode* pre, const Expr& post) final {
    auto tup_get = Downcast<TupleGetItem>(post);
    auto tup = tup_get->tuple;
    if (expr_share_map_.count(tup)) {
      auto tup_share = expr_share_map_[tup];
      if (tup_share.count(tup_get->index)) {
        ExprShareMap share;
        share.emplace(0, tup_share[tup_get->index]);
        expr_share_map_.emplace(post, share);
      }
    }
    return post;
  }

  Expr VisitExpr_(const LetNode* node) final {
    auto pre_visit = [this](const LetNode* node) {
      Expr value = this->Mutate(node->value);
      Var var = node->var;
      if (expr_share_map_.count(value)) {
        auto share = expr_share_map_[value];
        if (node->value->checked_type().as<TensorTypeNode>()) {
          CHECK(share.count(0));
          Var new_var = MakeVar(var->name_hint(), var->type_annotation, share[0]);
          var_update_map_.emplace(var, new_var);
        } else {
          // Tuple output, propagate the share map to the binding var
          expr_share_map_[var] = share;
        }
      }
    };
    auto post_visit = [this](const LetNode* node) {
      Var var = Downcast<Var>(this->Mutate(node->var));
      Expr value = this->Mutate(node->value);
      Expr body = this->Mutate(node->body);
      auto expr = GetRef<Expr>(node);
      if (var.same_as(node->var) && value.same_as(node->value) && body.same_as(node->body)) {
        this->memo_[expr] = expr;
      } else {
        this->memo_[expr] = Let(var, value, body);
      }
    };
    ExpandANormalForm(node, pre_visit, post_visit);
    return memo_[GetRef<Expr>(node)];
  }

 private:
  /*! \brief Mapping from expr to the share information of its output. */
  std::unordered_map<Expr, ExprShareMap, ObjectPtrHash, ObjectPtrEqual> expr_share_map_;
  /*! \brief Mapping from fused function to its share information. */
  std::unordered_map<Expr, FusedOpShareMap, ObjectPtrHash, ObjectPtrEqual> fused_op_share_map_;
  /*! \brief Mapping from original var to updated var with share annotation. */
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> var_update_map_;
};

/*
 * The major challenge is as follows:
 * 1. Two kinds of memory sharing are present in our scenario: 1) mandatory
 *    memory sharing determined by VM / interpreter, like %a = (%x0, %x1, %x2).
 *    where %a shares memory with %x0, %x1, %x2. It exists before the introduction
 *    of effect ir. 2) memory sharing newly introduced by effect ir, like
 *    %b = mnm.add(%a, 1). This pass is to tell whether there is a chance
 *    to make %b and %a in the above example share memory.
 *    Typical liveness analysis does not handle mandatory memory sharing as is
 *    denoted by 1).
 * 2. The memory sharing relation (denoted by ~) is not transitive:
 *    Say %a = (%x0, %x1, %x2), %a ~ %x0, %a ~ %x1, %a ~ %x2. But chances
 *    are that %x0 !~ %x1, %x0 !~ %x2, %x1 !~ %x2
 *
 * Note that for plain liveness analysis [1], neither of 1. and
 * 2. holds. So we transform the IR so that plain liveness analysis can
 * be applied. See LivenessAnalysis for more details.
 *
 * Our algorithm works as follows:
 * 1. run liveness analysis and obtain 1) the set of tensor var contained by each original var,
 *    and 2) the set of live tensor vars at each line.
 * 2. inplace write from ty to tx is invalid if and only if there exists a line l, such
 *    that the following two holds simultaneously:
 *    - live(l, x)
 *    - live(l, y)
 *    That is, the inplace write is valid iff the intersection of live(*, x) and live(*, y)
 *    is empty.
 */

class InplaceUpdateValidator : public ExprMutator {
 public:
  InplaceUpdateValidator(const Expr& body, liveness_analysis::LivenessAnalyzer& analyzer,
                         bool enforce_inplace_update)
      : body_(body),
        ell_(ExplicitLetList::make(body)),
        analyzer_(analyzer),
        enforce_inplace_update_(enforce_inplace_update) {
  }

  Expr VisitExpr_(const LetNode* node) override {
    Var var_ = node->var;
    const auto* var = static_cast<const ExtendedVarNode*>(var_.operator->());
    var->may_share = Var();
    Expr value = VisitExpr(node->value);
    Expr body = VisitExpr(node->body);
    return Let(var_, value, body);
  }

  Expr Run() {
    if (!analyzer_.IsSuccess()) {
      return VisitExpr(body_);
    }
    auto& vars = ell_->vars;
    auto& exprs = ell_->exprs;
    CHECK_EQ(vars.size(), exprs.size());
    int n = exprs.size();
    // memory sharing introduced by users
    for (int i = 0; i < n; ++i) {
      const auto* var = static_cast<const ExtendedVarNode*>(vars[i].operator->());
      if (var->may_share.defined()) {
        Var fout = *analyzer_.GetTensorVars(vars[i]).begin();
        Var fin = *analyzer_.GetTensorVars(var->may_share).begin();
        CHECK(fout.defined());
        CHECK(fin.defined());
        fout = analyzer_.Find(fout);
        fin = analyzer_.Find(fin);
        if (fout != fin && analyzer_.Intersect(fout, fin)) {
          if (enforce_inplace_update_) {
            LOG(FATAL) << "inplace update failed.";
          } else {
            // invalidate the inplace update
            var->may_share = Var();
          }
        } else {
          // the inplace update is valid
          analyzer_.Unite(fin, fout);
        }
      }
      if (const auto* node = exprs[i].as<IfNode>()) {
        // handle if branches recursively
        Expr true_branch =
            InplaceUpdateValidator(node->true_branch, analyzer_, enforce_inplace_update_).Run();
        Expr false_branch =
            InplaceUpdateValidator(node->false_branch, analyzer_, enforce_inplace_update_).Run();
        exprs[i] = If(node->cond, true_branch, false_branch);
      } else if (const auto* node = exprs[i].as<FunctionNode>()) {
        exprs[i] = VisitExpr(exprs[i]);
      }
    }
    return ell_->AsExpr();
  }

 private:
  /*! \brief the expression to be analyzed */
  const Expr& body_;
  /*! \brief the explicit let list of func_ */
  std::unique_ptr<ExplicitLetList> ell_{nullptr};
  /*! \brief the analyzer it belongs to */
  liveness_analysis::LivenessAnalyzer& analyzer_;
  /*! \brief Whether to enforce the inplace update */
  bool enforce_inplace_update_;
};

}  // namespace inplace_update

Pass InplaceUpdate() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    auto body = inplace_update::InplaceUpdateMutator().Mutate(f->body);
    return Function(f->params, body, f->ret_type, f->type_params, f->attrs);
  };
  return CreateMNMFunctionPass(pass_func, 1, "InplaceUpdate", {});
}

Pass ValidateInplaceUpdate(bool enforce_inplace_update) {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    auto analyzer = liveness_analysis::LivenessAnalyzer(f);
    analyzer.Run();
    auto body =
        inplace_update::InplaceUpdateValidator(f->body, analyzer, enforce_inplace_update).Run();
    return Function(f->params, body, f->ret_type, f->type_params, f->attrs);
  };
  return CreateMNMFunctionPass(pass_func, 1, "ValidateInplaceUpdate", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.InplaceUpdate").set_body_typed(InplaceUpdate);
MNM_REGISTER_GLOBAL("mnm.pass_.ValidateInplaceUpdate").set_body_typed(ValidateInplaceUpdate);

}  // namespace pass
}  // namespace mnm
