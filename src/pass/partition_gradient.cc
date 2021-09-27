/*!
 * Copyright (c) 2021 by Contributors
 * \file partition_gradient.cc
 * \brief Given a model after AutoDiff and InlineBackward, this pass performs the following:
 * ZeRO-1: Partition the gradients outputed by the given model, so that the later wrapped
 *         optimizer can have a partitioend optimizer status. Note that optimizers must
 *         consider gradient partitioning if applied; otherwise the result will be incorrect.
 * ZeRO-2: (TODO) Replace the reduce inserted by AutoDataParallel to obtain only
 *         a partition of gradients.
 */
#include "mnm/pass.h"

#include "./common.h"
#include "./let_list.h"
#include "../common/shape_utils.h"

namespace mnm {
namespace pass {
namespace partition_gradient {

class GradientPartitioner : public ExprMutator {
 public:
  GradientPartitioner(int n_part, const Function& func) : n_part_(n_part), func_(func) {
    // Build the var to expr map for the ANF.
    auto ell = ExplicitLetList::make(func->body);
    for (size_t i = 0; i < ell->vars.size(); ++i) {
      var_to_expr_.Set(ell->vars[i], ell->exprs[i]);
    }

    // Assume output is a tuple of (forward out, (grads, ...))
    auto ret = ell->exprs.back().as<TupleNode>();
    CHECK(ret != nullptr) << "Expected a tuple output, but got " << ell->exprs.back()->GetTypeKey();
    CHECK_EQ(ret->fields.size(), 2U)
        << "Expected the output tuple to be (out, (grad, ...)) with 2 fields, but it has "
        << ret->fields.size() << " fields";

    // Traverse back to find the gradient tuple.
    grad_tuple_var_ = Downcast<Var>(ret->fields[1]);
    auto grads = var_to_expr_[grad_tuple_var_];
    while (!grads->IsInstance<TupleNode>()) {
      auto tgi = grads.as<TupleGetItemNode>();
      CHECK(tgi != nullptr) << "Expected TupleGetItem, but got " << grads->GetTypeKey();
      auto tuple = Downcast<Tuple>(var_to_expr_[Downcast<Var>(tgi->tuple)]);
      grad_tuple_var_ = Downcast<Var>(tuple->fields[tgi->index]);
      grads = var_to_expr_[grad_tuple_var_];
    }

    for (auto field : Downcast<Tuple>(grads)->fields) {
      CHECK(field->IsInstance<VarNode>())
          << "Expected a var in the gradient tuple, but got " << field->GetTypeKey();
      grads_.Set(Downcast<Var>(field), Expr());
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
      // TODO(comaniac): ZeRO-2.
      // 1. Check if node->value is an allreduce, and change it to reduce scatter if so.
      // 2. Remove the following split logic. Once the gradient is already a slice by
      //    reduce scatter, we do not need to split it anymore.
      auto curr_var = node->var;
      auto value = VisitExpr(node->value);

      if (grads_.count(curr_var) > 0) {
        // Add the gradient slicing.
        scope->Push(curr_var, value);
        CHECK(!grads_[curr_var].defined());
        static const Op& split_op = Op::Get("mnm.op.split");
        static const Op& pad_op = Op::Get("mnm.op.pad");

        auto ttype = curr_var->checked_type().as<TensorTypeNode>();
        auto shape_expr = ttype->shape[0];
        if (!shape_expr->IsInstance<IntImmNode>()) {
          LOG(FATAL) << "Do not support dynamic shape yet";
          throw;
        }
        auto shape = tvm::tir::as_const_int(shape_expr)[0];

        // Padding is required if the axis to be splitted is not dividable.
        auto grad_var = curr_var;
        if (shape % n_part_ != 0) {
          auto part_shape = shape / n_part_;
          if (shape % n_part_ != 0) {
            part_shape += 1;
          }
          auto target_shape = part_shape * n_part_;
          auto pad_width = Array<Value>(ttype->shape.size() * 2, ScalarValue::make(0));
          // Always pad 0s to the end of the first axis.
          pad_width.Set(1, ScalarValue::make(target_shape - shape));
          grad_var = scope->Push(Call(pad_op, {grad_var, MakeConstant(TupleValue::make(pad_width)),
                                               MakeConstant(ScalarValue::make(0)),
                                               MakeConstant(StringValue::make("constant"))}));
        }

        // TODO(comaniac): Construct if-branches if rank_ is unknown.
        auto split_grad =
            scope->Push(Call(split_op, {grad_var, MakeConstant(ScalarValue::make(n_part_)),
                                        MakeConstant(ScalarValue::make(0))}));
        auto slice_grad = scope->Push(TupleGetItem(split_grad, rank_));
        grads_.Set(curr_var, slice_grad);
      } else if (curr_var == grad_tuple_var_) {
        // Replace gradients with sliced ones.
        Array<Expr> fields;
        for (auto field : Downcast<Tuple>(value)->fields) {
          auto var_node = field.as<VarNode>();
          CHECK(var_node != nullptr);
          auto var = GetRef<Var>(var_node);
          if (grads_.count(var) > 0) {
            CHECK(grads_[var].defined())
                << "Internal error: gradient " << var << " does not map to the sliced one";
            fields.push_back(grads_[var]);
          } else {
            fields.push_back(field);
          }
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

 protected:
  /*! \brief The scope stack of the let list. */
  std::vector<std::unique_ptr<LetList>> scopes_;
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
};

}  // namespace partition_gradient

Pass PartitionGradient(size_t n_part, int rank) {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return partition_gradient::GradientPartitioner(n_part, f).Partition(rank);
  };
  return CreateMNMFunctionPass(pass_func, 0, "PartitionGradientFunc", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.PartitionGradient").set_body_typed(PartitionGradient);

}  // namespace pass
}  // namespace mnm
