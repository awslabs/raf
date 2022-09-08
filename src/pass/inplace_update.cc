/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file inplace_update.cc
 * \brief Annotate the may_share field in the ExtendedVar to share the memory between inputs and
 * outputs and check the validity of memory sharing.
 */
#include "raf/op.h"
#include "raf/ir_ext.h"
#include "raf/pass.h"
#include "tvm/ir/type_functor.h"
#include "./let_list.h"
#include "./common.h"
#include "./liveness_analysis.h"

namespace raf {
namespace pass {
namespace inplace_update {

using namespace raf::op;

/*! \brief Extract the mapping from the parmeters of fused func to call arg expr. */
class ExtractCallArgs : public MixedModeVisitor {
 public:
  void VisitExpr_(const CallNode* call) final {
    auto func = call->op.as<FunctionNode>();
    if (func && func->HasNonzeroAttr(attr::kPrimitive)) {
      ICHECK_EQ(func->params.size(), call->args.size());
      for (uint i = 0; i < func->params.size(); ++i) {
        param_value_map.emplace(func->params[i], call->args[i]);
      }
    }
  }

  /*! \brief Mapping from fused function parameter to actual argument value. */
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> param_value_map;
};

/*!
 * \brief InplaceUpdateMutator marks may_share in the variable for ops that have attr
 * TRAFInplaceUpdate so that inputs and outputs may share the memory and perform inplace update.
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
  // Mapping from the output index in the expr to a variable that shares the buffer
  //   The output index is always 0 if the expr type is tensor; otherwise, the index indicates
  //   the index of elements in a tuple.
  using ExprShareMap = std::unordered_map<int, Var>;
  // Mapping from the output index to the parameter index that shares the memory for fused functions
  using FusedOpShareMap = std::unordered_map<int, int>;

  Expr Run(const Expr& expr) {
    auto extract = ExtractCallArgs();
    extract.VisitExpr(expr);
    param_value_map_ = extract.param_value_map;
    return Mutate(expr);
  }

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
      // The output(s) of the fused op may need to share memory with input(s) of the fused op.
      // We need to replace the memory-sharing var in the expr_share_map by the param index in the
      // fused function
      auto& expr_share_map = find->second;
      FusedOpShareMap fuse_share_map;
      for (auto it : expr_share_map) {
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
        fuse_share_map.emplace(out_idx, input_idx);
        fused_op_share_map_.emplace(new_func, fuse_share_map);
      }
    }
    return new_func;
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    static auto finplace = Op::GetAttrMap<op::TRAFInplaceUpdate>("TRAFInplaceUpdate");
    static auto add_op = Op::Get("raf.op.add");
    static auto subtract_op = Op::Get("raf.op.subtract");
    auto call = Downcast<Call>(post);
    auto op = Mutate(call->op);
    if (op.as<OpNode>()) {
      auto opn = Downcast<Op>(op);
      if (IsDialectOp(opn)) {
        opn = GetBaseOp(opn);
      }
      if (finplace.count(opn)) {
        ExprShareMap share;
        for (auto it : finplace[opn]) {
          auto arg = Downcast<Var>(call->args[it.first.IntValue()]);
          if (var_tuple_map_.count(arg)) {
            auto tnode = var_tuple_map_[arg].as<TupleNode>();
            // Two tuples share the same storage. The two tuples should have the same
            // number of the tensors. So the input tuple index is also the output tensor
            // index. The tuple is expended.
            for (int i = 0; i < tnode->fields.size(); ++i) {
              share.emplace(i, Downcast<Var>(tnode->fields[i]));
            }
          } else {
            auto var = arg.as<ExtendedVarNode>();
            if (var && var->may_share.defined()) {
              arg = GetLatestVar(var->may_share);
            }
            share.emplace(it.second.IntValue(), arg);
          }
        }
        expr_share_map_.emplace(post, share);
      } else if (opn == add_op || opn == subtract_op) {
        // Currently only support add and subtract in the binary ops for inplace update.
        // If the out_arg is a variable, the output tensor needs to share memory with the out_arg
        ICHECK_GT(pre->args.size(), 2) << "Invalid add/subtract schema";
        auto out_arg = pre->args[2];
        // Extract the actual call arg if the out_arg is a param in the fused function
        auto it = param_value_map_.find(out_arg);
        if (it != param_value_map_.end()) {
          out_arg = it->second;
        }
        if (out_arg.as<ExtendedVarNode>()) {
          // Use the call arg in the share map because the shared var could be used to
          // map to the func parameter if this call node is inside a fused function.
          ExprShareMap share{{0, Downcast<Var>(call->args[2])}};
          expr_share_map_.emplace(post, share);
        }
      }
    } else if (op.as<FunctionNode>()) {
      // Propagate the fused_op_share_map to expr_share_map
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
      Var var = node->var;
      Expr value = this->Mutate(node->value);
      if (node->value.as<TupleNode>()) {
        var_tuple_map_.emplace(var, value);
      }
      if (expr_share_map_.count(value)) {
        auto share_map = expr_share_map_[value];
        if (node->value->checked_type().as<TensorTypeNode>()) {
          CHECK(share_map.count(0));
          Var new_var = MakeVar(var->name_hint(), var->type_annotation, GetLatestVar(share_map[0]));
          new_var->checked_type_ = var->checked_type();
          var_update_map_.emplace(var, new_var);
        } else {
          // Tuple type, just propagate the share map to the binding var
          expr_share_map_[var] = share_map;
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
  /*!
   * \brief Get the up-to-date shared var.
   * If the shared var is already in the update map, meaning that the shared var
   * also shares another var, forming a share chain. In this case, we need to use
   * the up-to-date shared var. For example:
   *
   * let %x1 = raf.op.add(%x, %x, nullptr, nullptr);
   * let %x2(share: %x1) = raf.op.add(%x1, %x, %x1, nullptr);
   * let %x3(share: %x2) = raf.op.add(%x2, %x, %x2, nullptr);
   *
   * In %x3, its shared var should be the new created %x2(share: %x1) instead of
   * the original %x2(share: nullptr).
   * \param var The original shared var.
   * \return The up-to-date shared var, which can be the identical shared var or
   * a new created shared var.
   */
  inline Var GetLatestVar(const Var& var) {
    if (var_update_map_.count(var) > 0) {
      return var_update_map_[var];
    }
    return var;
  }

  /*! \brief Mapping from expr to the share information of its output. */
  std::unordered_map<Expr, ExprShareMap, ObjectPtrHash, ObjectPtrEqual> expr_share_map_;
  /*! \brief Mapping from fused function to its share information. */
  std::unordered_map<Expr, FusedOpShareMap, ObjectPtrHash, ObjectPtrEqual> fused_op_share_map_;
  /*! \brief Mapping from original var to updated var with may_share annotation. */
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> var_update_map_;
  /*! \brief Mapping from var to tuple, which records each var to tuple binding. */
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> var_tuple_map_;
  /*! \brief Mapping from fused function parameter to actual call argument value. */
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> param_value_map_;
};

/*
 * The major challenge is as follows:
 * 1. Two kinds of memory sharing are present in our scenario: 1) mandatory
 *    memory sharing determined by VM / interpreter, like %a = (%x0, %x1, %x2).
 *    where %a shares memory with %x0, %x1, %x2. It exists before the introduction
 *    of effect ir. 2) memory sharing newly introduced by effect ir, like
 *    %b = raf.add(%a, 1). This pass is to tell whether there is a chance
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

/*!
 * \brief A simple mutator to mutate the case:
 * let %a1 = call(...);
 * let %a2(may_share: %w) = add(%a1, 0, %w)
 *
 * to:
 *
 * let %a1(may_share: %w) = call(...);
 *
 * TODO(issue 758): This case should only created by PartitionOptimizerStatus pass, and this pass
 * can be removed once issue 758 is resolved.
 */
class InplaceSimplifer : public ExprMutator {
 public:
  InplaceSimplifer() {
    scopes_.emplace_back(new LetList);
  }

  Function Run(const Function& func) {
    for (auto param : func->params) {
      func_params_.insert(param);
    }
    return Downcast<Function>(this->Mutate(func));
  }

  Expr VisitExpr_(const LetNode* node) {
    scopes_.emplace_back(new LetList);
    auto scope = scopes_.back().get();
    Expr body;
    Var prev_var, curr_var;
    Expr prev_value, curr_value;
    bool ignore_prev = false;
    do {
      curr_var = node->var;
      curr_value = VisitExpr(node->value);

      bool ignore_curr = false;
      if (curr_value->IsInstance<CallNode>() && IsInplaceAddZero(Downcast<Call>(curr_value))) {
        auto add_call = Downcast<Call>(curr_value);
        if (prev_var.defined() && add_call->args[0] == prev_var) {
          // We only consider the consecutive case; otherwise we may have correctness issue.
          prev_var = MakeVar(prev_var->name_hint(), prev_var->type_annotation,
                             Downcast<Var>(add_call->args[2]));
          mutated_vars_.Set(curr_var, prev_var);
          ignore_curr = true;
        }
      }

      if (!ignore_prev && prev_var.defined()) {
        scope->Push(prev_var, prev_value);
      }

      prev_var = curr_var;
      prev_value = curr_value;
      ignore_prev = ignore_curr;

      body = node->body;
      node = body.as<LetNode>();
    } while (node);

    if (!ignore_prev && prev_var.defined()) {
      scope->Push(prev_var, prev_value);
    }

    auto ret = scopes_.back()->Get(this->Mutate(body));
    scopes_.pop_back();
    return ret;
  }

  Expr VisitExpr_(const VarNode* node) {
    auto var = GetRef<Var>(node);
    if (mutated_vars_.count(var) > 0) {
      return mutated_vars_[var];
    }
    return var;
  }

 private:
  bool IsInplaceAddZero(const Call& call) {
    static auto add_op = Op::Get("raf.op.add");

    if (!call->op->IsInstance<OpNode>()) {
      return false;
    }

    auto op = Downcast<Op>(call->op);
    if (op != add_op || call->args.size() < 3) {
      return false;
    }

    // 1st argument cannot be an input parameter.
    if (func_params_.find(Downcast<Var>(call->args[0])) != func_params_.end()) {
      return false;
    }

    // 2nd argument must be 0.
    auto const_node = call->args[1].as<ConstantNode>();
    if (!const_node || !const_node->value->IsInstance<FloatValueObj>()) {
      return false;
    }
    auto value = const_node->value.as<FloatValueObj>()->value;
    if (value != 0) {
      return false;
    }

    // 3rd argument must be an input parameter.
    if (func_params_.find(Downcast<Var>(call->args[2])) == func_params_.end()) {
      return false;
    }

    return true;
  }

  /*! \brief The scope stack of the let list. */
  std::vector<std::unique_ptr<LetList>> scopes_;
  /*! \brief All function parameters. */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> func_params_;
  /*! \brief Mapping from the removed var (bind to add) to the new target var. */
  Map<Var, Var> mutated_vars_;
};

}  // namespace inplace_update

Pass InplaceUpdate() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    auto body = inplace_update::InplaceUpdateMutator().Run(f->body);
    auto func = Function(f->params, body, f->ret_type, f->type_params, f->attrs);
    return inplace_update::InplaceSimplifer().Run(func);
  };
  return CreateRAFFunctionPass(pass_func, 1, "InplaceUpdate", {"InferType"});
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
  return CreateRAFFunctionPass(pass_func, 1, "ValidateInplaceUpdate", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.InplaceUpdate").set_body_typed(InplaceUpdate);
RAF_REGISTER_GLOBAL("raf.pass_.ValidateInplaceUpdate").set_body_typed(ValidateInplaceUpdate);

}  // namespace pass
}  // namespace raf
