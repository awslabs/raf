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

#define RAF_OP_TRAIN_VARIANT(op_name, variant_op_name) \
  RELAY_REGISTER_OP(op_name).set_attr<std::string>("TrainOp", variant_op_name);

#define RAF_OP_INFER_VARIANT(op_name, variant_op_name) \
  RELAY_REGISTER_OP(op_name).set_attr<std::string>("InferOp", variant_op_name);

// Register the op pairs for inference/training variants.
RAF_OP_TRAIN_VARIANT("raf.op.layer_norm", "raf.op.layer_norm_train");
RAF_OP_INFER_VARIANT("raf.op.layer_norm_train", "raf.op.layer_norm");

class OpReplacer : public ExprMutator {
 public:
  OpReplacer(const Function& func, bool to_train_op) : func_(func), to_train_op_(to_train_op) {
    scopes_.emplace_back(new LetList);
  }

  Function Replace() {
    Array<Var> func_params{func_->params};
    auto new_body = this->Mutate(func_->body);
    return Function(func_params, new_body, func_->ret_type, {}, func_->attrs);
  }

  Expr VisitExpr_(const LetNode* node) override {
    auto switch_op_map = Op::GetAttrMap<std::string>((to_train_op_) ? "TrainOp" : "InferOp");

    scopes_.emplace_back(new LetList);
    auto scope = scopes_.back().get();
    Expr body;
    do {
      auto curr_var = node->var;
      auto value = node->value;

      // Switch the op if needed.
      if (auto node = value.as<CallNode>()) {
        const auto op_node = node->op.as<OpNode>();
        if (op_node && switch_op_map.count(GetRef<Op>(op_node)) > 0) {
          const Op& target_op = Op::Get(switch_op_map[GetRef<Op>(op_node)]);
          auto new_call = Call(target_op, node->args);

          // We only target to normalization ops and their training variants return
          // a tuple of (out, mean, variance).
          // We could revisit and refactor this pass in the future if we have motivations to
          // decompose/recompose the ops for training/inference.
          if (to_train_op_) {
            auto new_var = scope->Push(new_call);
            value = TupleGetItem(new_var, 0);
          } else {
            no_tuple_vars_.insert(curr_var);
            value = new_call;
          }
        } else {
          value = VisitExpr(value);
        }
      } else {
        value = VisitExpr(value);
      }
      scope->Push(curr_var, value);

      body = node->body;
      node = body.as<LetNode>();
    } while (node);
    auto ret = scopes_.back()->Get(this->Mutate(body));
    scopes_.pop_back();
    return ret;
  }

  Expr VisitExpr_(const TupleGetItemNode* node) override {
    auto tuple_var = Downcast<Var>(node->tuple);
    if (no_tuple_vars_.find(tuple_var) != no_tuple_vars_.end()) {
      CHECK_EQ(node->index, 0U)
          << "InternalError: The TupleGetItem for a switable op does not take the first element";
      return tuple_var;
    }
    return TupleGetItem(tuple_var, node->index);
  }

 private:
  /*! \brief The scope stack of the let list. */
  std::vector<std::unique_ptr<LetList>> scopes_;
  /*! \brief The let-binding vars that are no longer tuples after transform. */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> no_tuple_vars_;
  /*! \brief The target function. */
  Function func_;
  /*! \brief Switch to training or inference op. */
  bool to_train_op_;
};

}  // namespace switch_train

Pass SwitchTrainOp(bool to_train_op) {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return switch_train::OpReplacer(f, to_train_op).Replace();
  };
  auto switch_train = CreateRAFFunctionPass(pass_func, 0, "SwitchTrainOp", {});
  return RAFSequential({switch_train}, "SwitchTrainOp");
}

RAF_REGISTER_GLOBAL("raf.pass_.SwitchTrainOp").set_body_typed(SwitchTrainOp);

}  // namespace pass
}  // namespace raf
