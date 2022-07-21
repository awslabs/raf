/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file partition_gradient.cc
 * \brief Given a model after AutoDiff and InlineBackward, this pass performs the following:
 * ZeRO-1: Partition the gradients outputed by the given model, so that the later wrapped
 *         optimizer can have a partitioend optimizer status. Note that optimizers must
 *         consider gradient partitioning if applied; otherwise the result will be incorrect.
 * ZeRO-2: Replace the allreduce inserted by AutoDataParallel with reduce_scatter to obtain only
 *         a partition of gradients.
 */
#include "raf/pass.h"

#include "./common.h"
#include "./let_list.h"
#include "../common/shape_utils.h"

namespace raf {
namespace pass {
namespace partition_gradient {

class GradientPartitioner : public ExprMutator {
 public:
  GradientPartitioner(int opt_level, int n_part, int64_t bucket_size, const Function& func)
      : opt_level_(opt_level), n_part_(n_part), bucket_size_(bucket_size), func_(func) {
    // Build the var to expr map for the ANF.
    Map<Var, Expr> var_to_expr;
    auto ell = ExplicitLetList::make(func->body);
    for (size_t i = 0; i < ell->vars.size(); ++i) {
      var_to_expr.Set(ell->vars[i], ell->exprs[i]);
    }

    // Assume output is a tuple of (forward out, (grads, ...))
    auto ret = ell->exprs.back().as<TupleNode>();
    CHECK(ret != nullptr) << "Expected a tuple output, but got " << ell->exprs.back()->GetTypeKey();
    CHECK_EQ(ret->fields.size(), 2U)
        << "Expected the output tuple to be (out, (grad, ...)) with 2 fields, but it has "
        << ret->fields.size() << " fields";

    // Traverse back to find the gradient tuple.
    grad_tuple_var_ = Downcast<Var>(ret->fields[1]);
    auto grads = var_to_expr[grad_tuple_var_];
    while (!grads->IsInstance<TupleNode>()) {
      auto tgi = grads.as<TupleGetItemNode>();
      CHECK(tgi != nullptr) << "Expected TupleGetItem, but got " << grads->GetTypeKey();
      auto tuple = Downcast<Tuple>(var_to_expr[Downcast<Var>(tgi->tuple)]);
      grad_tuple_var_ = Downcast<Var>(tuple->fields[tgi->index]);
      grads = var_to_expr[grad_tuple_var_];
    }
    auto grad_fields = Downcast<Tuple>(grads)->fields;
    for (auto field : grad_fields) {
      if (field->IsInstance<VarNode>()) {
        grads_.Set(Downcast<Var>(field), Expr());
        last_all_reduce_ = Downcast<Var>(field);
      }
    }

    scopes_.emplace_back(new LetList);
  }

  /*! \brief Partition the parameters according to the parameter group. */
  Function Partition(int rank) {
    if (grads_.empty()) {  // No gradients to be partitioned.
      return func_;
    }

    rank_ = rank;
    Array<Var> func_params{func_->params};
    auto new_body = this->Mutate(func_->body);
    return Function(func_params, new_body, func_->ret_type, {}, func_->attrs);
  }

  Expr VisitExpr_(const LetNode* node) {
    scopes_.emplace_back(new LetList);
    auto scope = scopes_.back().get();
    Expr body;
    do {
      auto curr_var = node->var;
      auto value = VisitExpr(node->value);
      var_to_expr_.Set(curr_var, value);

      if (grads_.count(curr_var) > 0) {
        // The curr_var is a complete gradient.
        CHECK(!grads_[curr_var].defined());
        SliceGrad(scope, curr_var, value, opt_level_);
      } else if (curr_var == grad_tuple_var_) {
        // Replace gradients with sliced ones.
        Array<Expr> fields;
        for (auto field : Downcast<Tuple>(value)->fields) {
          if (field->IsInstance<VarNode>()) {
            auto var_node = field.as<VarNode>();
            auto var = GetRef<Var>(var_node);
            if (grads_.count(var) > 0) {
              CHECK(grads_[var].defined())
                  << "Internal error: gradient " << var << " does not map to the sliced one";
              fields.push_back(grads_[var]);
              continue;
            }
          }
          fields.push_back(field);
        }
        scope->Push(curr_var, Tuple(fields));
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

  /*!\ brief Log identified parameters for debugging. */
  std::string DebugDumpOutParams() {
    std::stringstream ss;
    for (const auto& kv : grads_) {
      ss << kv.first << "(share: " << kv.second << ")" << std::endl;
    }
    return ss.str();
  }

 private:
  /*! \brief Check whether a given expression is a call expression with all_reduce. */
  inline bool IsAllReduceCall(const Expr& expr) {
    static const Op& allreduce_op = Op::Get("raf.op._allreduce");
    if (!expr->IsInstance<CallNode>()) {
      return false;
    }
    auto call = Downcast<Call>(expr);
    if (auto node = call->op.as<OpNode>()) {
      return GetRef<Op>(node) == allreduce_op;
    }
    return false;
  }

  /*! \brief Detect allreduce expr or allreduce followed by divide due to use NCCL version<2.10.
   *  return allreduce expr and divied expr if the pattern match otherwise NullValue*/
  inline std::tuple<Expr, Expr> GetAllReduceExpr(const Expr& expr) {
    Expr allreduce_expr = Expr();
    Expr divide_expr = Expr();

    if (expr->IsInstance<CallNode>()) {
      static const Op& divide_op = Op::Get("raf.op.divide");
      auto call_node = expr.as<CallNode>();
      auto node = call_node->op.as<OpNode>();
      if (IsAllReduceCall(expr)) {
        allreduce_expr = expr;
      } else if (node && GetRef<Op>(node) == divide_op) {
        Var allreduce_var = Downcast<Var>(call_node->args[0]);
        if (var_to_expr_.count(allreduce_var) && IsAllReduceCall(var_to_expr_[allreduce_var])) {
          allreduce_expr = var_to_expr_[allreduce_var];
          divide_expr = expr;
        }
      }
    }
    return std::make_tuple(allreduce_expr, divide_expr);
  }

  /*! \brief Return the n'th argument of the given call expr. */
  inline Expr GetNArg(const Expr& expr, int n) {
    CHECK(expr->IsInstance<CallNode>());
    auto call = Downcast<Call>(expr);
    CHECK_GE(call->args.size(), n)
        << "Expected at least " << n << " argument, but got " << raf::ir::AsText(expr);
    return call->args[n];
  }

  /*! \brief Since we always partition the first dimension of gradient tensors, we have to
   * make sure the length of the first dimension is dividable to the total device number.
   * This helper function analyzes the tensor size and generates the pad call if needed.
   * */
  inline Var GenPadCall(LetList* scope, const Var& var) {
    static const Op& pad_op = Op::Get("raf.op.pad");

    // Extract the length of the first dimension.
    auto ttype = var->checked_type().as<TensorTypeNode>();
    CHECK(ttype != nullptr) << "Expected a tesnor, but got " << var->checked_type();
    auto shape_expr = ttype->shape[0];
    if (!shape_expr->IsInstance<IntImmNode>()) {
      LOG(FATAL) << "Do not support dynamic shape yet";
      throw;
    }
    auto dim0_length = tvm::tir::as_const_int(shape_expr)[0];

    // Check if we need to pad the gradient or not.
    auto ret_var = var;
    if (dim0_length % n_part_ != 0) {
      Array<Value> pad_width;
      auto part_dim0_length = dim0_length / n_part_ + 1;
      auto target_dim0_length = part_dim0_length * n_part_;
      pad_width = Array<Value>(ttype->shape.size() * 2, ScalarValue::make(0));
      // Always pad 0s to the end of the first axis.
      pad_width.Set(1, ScalarValue::make(target_dim0_length - dim0_length));

      ret_var = scope->Push(Call(pad_op, {var, MakeConstant(TupleValue::make(pad_width)),
                                          MakeConstant(ScalarValue::make(0)),
                                          MakeConstant(StringValue::make("constant"))}));
    }
    return ret_var;
  }

  /*!
   * \brief Slice gradient based on ZeRO-1 and ZeRO-2.
   * The desired IR for ZeRO-1 is:
   * let %1 = all_reduce(%0); // Could also be a backward op if data_parallel is disabled
   * let %2 = pad(%1, ...);   // %1 is the complete global gradient
   * let %3 = split(%2, ...);
   * let %4 = TupleGetItem(%3, rank);
   * TODO(comaniac): Add %rank to the function argument if rank_ is unknown.
   *
   * The desired IR for ZeRO-2 is if bucket_size_ < 2:
   * // if NCCL version is >= 2.10
   * let %1 = op(%0);       // A backward op to generate gradient
   * let %2 = pad(%1, ...); // %1 is the complete local gradient
   * let %3 = reduce_scatter(%2, avg);
   * // else NCCL version is < 2.10
   * let %1 = op(%0);       // A backward op to generate gradient
   * let %2 = pad(%1, ...); // %1 is the complete local gradient
   * let %3 = reduce_scatter(%2, sum);
   * let %4 = divide(%3, ...)
   * The desired IR for ZeRO-2 is if bucket_size_ > 2, which means group reduce_scatter:
   * // if NCCL version is >= 2.10
   * let %1 = op(%0);       // A backward op to generate gradient
   * let %2 = pad(%1, ...); // %1 is the complete local gradient
   * let %3 = Tuple(%2);
   * let %4 = group_reduce_scatter(%3, avg);
   * // else NCCL version is < 2.10
   * let %1 = op(%0);       // A backward op to generate gradient
   * let %2 = pad(%1, ...); // %1 is the complete local gradient
   * let %3 = Tuple(%2);
   * let %4 = group_reduce_scatter(%3, sum);
   * let %5 = %4.0
   * let %6 = divide(%5, ...)

*/
  void SliceGrad(LetList* scope, const Var& var, const Expr& value, int opt_level) {
    static const Op& split_op = Op::Get("raf.op.split");
    static const Op& reduce_scatter_op = Op::Get("raf.op._reduce_scatter");
    Expr allreduce_expr, divide_expr;
    std::tie(allreduce_expr, divide_expr) = GetAllReduceExpr(value);
    if (opt_level_ > 1 && !allreduce_expr.defined()) {
      // If this is not an AllReduce, then the gradient was generated locally and
      // no need to apply ZeRO-2.
      opt_level = 1;
    }
    Var grad_var;
    if (opt_level > 1) {
      // ZeRO-2: Replace the AllReduce with ReduceScatter.
      auto first_arg = Downcast<Var>(GetNArg(allreduce_expr, 0));
      // The 1st arg of allreduce is a tuple of tensors.
      Constant compute = Downcast<Constant>(GetNArg(allreduce_expr, 1));
      auto arg_tuple = Downcast<Tuple>(var_to_expr_[first_arg]);
      CHECK_EQ(arg_tuple->fields.size(), 1U) << "Not supported yet";

      // FIXME(comaniac): This happens when gradients are zeros and are folded.
      // However, we should eliminate zero gradients to reduce communication overheads.
      if (arg_tuple->fields[0]->IsInstance<ConstantNode>()) {
        grad_var = scope->Push(TupleGetItem(first_arg, 0));
        grad_var->checked_type_ = arg_tuple->fields[0]->checked_type();
      } else {
        grad_var = Downcast<Var>(arg_tuple->fields[0]);
      }

      int64_t size = common::shape_utils::GetElementNum(grad_var);
      grad_var = GenPadCall(scope, grad_var);
      if (bucket_size_ < 2) {
        // Do not group redcue_scatter
        auto reduce_scatter_var = scope->Push(Call(reduce_scatter_op, {grad_var, compute}));
        if (divide_expr.defined()) {
          // update the divide op args
          auto divide_call = divide_expr.as<CallNode>();
          reduce_scatter_var =
              scope->Push(Call(divide_call->op, {reduce_scatter_var, divide_call->args[1]}));
        }
        grads_.Set(var, reduce_scatter_var);
      } else {
        if (var == last_all_reduce_) {
          scatter_input_.push_back(grad_var);
          scatter_var_.push_back(var);
          divide_expr_.push_back(divide_expr);
          IssueGroupScatter(scope, compute);
          return;
        }
        if (curr_size_ + size < bucket_size_) {
          scatter_input_.push_back(grad_var);
          scatter_var_.push_back(var);
          divide_expr_.push_back(divide_expr);
          curr_size_ += size;
        } else {
          IssueGroupScatter(scope, compute);
          scatter_input_.push_back(grad_var);
          scatter_var_.push_back(var);
          divide_expr_.push_back(divide_expr);
          curr_size_ = size;
        }
      }
    } else {
      // ZeRO-1: Keep AllReduce (or the backward op if data parallel is disabled).
      scope->Push(var, value);
      grad_var = GenPadCall(scope, var);
      grad_var = scope->Push(Call(split_op, {grad_var, MakeConstant(ScalarValue::make(n_part_)),
                                             MakeConstant(ScalarValue::make(0))}));
      auto replace_var = scope->Push(TupleGetItem(grad_var, rank_));
      grads_.Set(var, replace_var);
    }
  }

  void IssueGroupScatter(LetList* scope, Constant compute) {
    static const Op& group_reduce_scatter = Op::Get("raf.op._group_reduce_scatter");
    auto inputs = scope->Push(Tuple(scatter_input_));

    auto scatter_out = scope->Push(Call(group_reduce_scatter, {inputs, compute}));
    for (int i = 0; i < scatter_var_.size(); ++i) {
      auto update_var = scope->Push(TupleGetItem(scatter_out, i));
      auto divide_expr = divide_expr_[i];
      if (divide_expr.defined()) {
        // update the divide op args
        auto divide_call = divide_expr.as<CallNode>();
        update_var = scope->Push(Call(divide_call->op, {update_var, divide_call->args[1]}));
      }
      grads_.Set(scatter_var_[i], update_var);
    }
    scatter_input_ = {};
    scatter_var_ = {};
  }
  /*! \brief The scope stack of the let list. */
  std::vector<std::unique_ptr<LetList>> scopes_;
  /*! \brief The optimization level (ZeRO-n). */
  int opt_level_;
  /*! \brief The expected number of partitions. */
  int n_part_;
  /*! \brief The target function. */
  Function func_;
  /*! \brief The rank of the current running device. */
  int rank_;
  /*! \brief Mapping from a gradient to a sliced gradient for this rank. */
  Map<Var, Expr> grads_;
  /*! \brief The var binding to the gradient tuple. */
  Var grad_tuple_var_;
  /*! \brief Mapping from let-binding var to the expression. */
  Map<Var, Expr> var_to_expr_;
  /*! \brief The bucket size for group scatter. */
  int64_t bucket_size_;
  /*! \brief The current bucket size for group scatter. */
  int64_t curr_size_ = 0;
  /*! \brief The last all reduce in graph. */
  Var last_all_reduce_;
  /*! \brief The inputs for group scatter. */
  std::vector<Expr> scatter_input_;
  /*! \brief The group scatter var. */
  std::vector<Var> scatter_var_;
  /*! \brief Divide expr after allreduce for NCCL version < 2.10. */
  std::vector<Expr> divide_expr_;
};

}  // namespace partition_gradient

Pass PartitionGradient(int opt_level, int n_part, int rank, int64_t bucket_size) {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return partition_gradient::GradientPartitioner(opt_level, n_part, bucket_size, f)
        .Partition(rank);
  };
  auto partition_gradient = CreateRAFFunctionPass(pass_func, 0, "PartitionGradientFunc", {});
  return RAFSequential({partition_gradient, EraseType(), DeadCodeElimination()},
                       "PartitionGradient");
}

RAF_REGISTER_GLOBAL("raf.pass_.PartitionGradient").set_body_typed(PartitionGradient);

}  // namespace pass
}  // namespace raf
