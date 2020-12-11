/*!
 * Copyright (c) 2020 by Contributors
 * \file merge_compiler_regions.cc
 * \brief After operators have been annotated with the targets that support
 * them, this pass creates regions of the operators for each target. It
 * is guaranteed that the regions will have a topological ordering so that
 * no data dependency issues exist.
 *
 * This pass only introduces annotations to indicate the regions.
 * partition_graph must subsequently be called to lift these regions out
 * as external functions.
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "./common.h"
#include "../op/schema/annotation.h"

namespace mnm {
namespace pass {
namespace merge_compiler_regions {

using namespace mnm::ir;
using namespace tvm;
using namespace tvm::relay;
using mnm::op::schema::CompilerArgs;

static const Op& begin_op = CompilerBeginOp();
static const Op& end_op = CompilerEndOp();

class MergeAnnotations : public ExprRewriter {
 public:
  explicit MergeAnnotations() {
  }

  /*!
   * \brief Remove the compiler_begin annotation inside the CallNode.
   * \param expr The input expression to remove annotations from.
   * \param ann_op The specific annotation to remove.
   * \return The expression after remove annotation.
   */
  Expr RemoveAnnotation(const Expr& expr, const Op& ann_op) {
    if (ann_op == begin_op) {
      if (expr.as<CallNode>()) {
        const CallNode* call = expr.as<CallNode>();

        // If the CallNode is annotated by compiler_end, then get
        // the CallNode inside the compiler_end.
        if (call->op == CompilerEndOp()) {
          CHECK_EQ(call->args.size(), 1U);
          auto input_expr = call->args[0];

          // Remove the compiler_begin annotation of this input_call,
          // and return the expr after annotate it with compiler_end.
          auto new_expr = RemoveAnnotation(input_expr, begin_op);
          Expr ret_expr = Call(call->op, {new_expr}, call->attrs);
          ret_expr->checked_type_ = expr->checked_type_;

          return ret_expr;
        } else {
          // Check if the compiler_begin has removed. If not return after
          // remove the annotation.
          if (call->args[0].as<CallNode>()->op == CompilerBeginOp()) {
            Array<Expr> new_args;
            for (auto& arg : call->args) {
              const CallNode* arg_call = arg.as<CallNode>();
              CHECK_EQ(arg_call->op, CompilerBeginOp()) << "ValueError: compiler_begin not found";
              CHECK_EQ(arg_call->args.size(), 1U);
              new_args.push_back(arg_call->args[0]);
            }

            Expr new_expr = {Call(call->op, new_args, call->attrs)};
            new_expr->checked_type_ = call->checked_type_;

            return new_expr;
          } else {
            // This expr is not annotated with compiler_begin, return it directly.
            return expr;
          }
        }
      } else if (expr.as<TupleNode>()) {
        // Remove the annotation for TupleNode.
        const TupleNode* tuple = expr.as<TupleNode>();

        // If the fields of the TupleNode is annotated, then remove
        // the annotation, else return this TupleNode directly.
        if (tuple->fields[0].as<CallNode>()->op == CompilerBeginOp()) {
          Array<Expr> new_fields;
          for (auto field : tuple->fields) {
            auto field_call = field.as<CallNode>();
            CHECK_EQ(field_call->op, CompilerBeginOp()) << "ValueError: compiler_begin not found";
            CHECK_EQ(field_call->args.size(), 1U);
            new_fields.push_back(field_call->args[0]);
          }

          Expr new_tuple = {Tuple(new_fields)};
          new_tuple->checked_type_ = expr->checked_type_;

          return new_tuple;
        } else {
          return expr;
        }
      }
    }
    // Remove the compiler_end annotation inside the CallNode.
    else if (ann_op == end_op) {
      const CallNode* call = expr.as<CallNode>();
      if (call->op == CompilerEndOp()) {
        // Remove compiler_begin annotations of the input call's arguments.
        return call->args[0];
      } else {
        // If the compiler_end annotation is already removed, then do nothing.
        return expr;
      }
    } else {
      LOG(FATAL) << "ValueError: unknown op";
    }
  }

  Expr Rewrite_(const FunctionNode* func, const Expr& post) final {
    std::unique_ptr<ExplicitLetList> ell_ = ExplicitLetList::make(func->body);
    const std::unique_ptr<ExplicitLetList> ref_ell_ = ExplicitLetList::make(func->body);

    size_t ell_n = ell_->vars.size();
    CHECK_GT(ell_n, 0U);

    // Multiple ops inside the FunctionNode. Explore the possible of merging.
    if (ell_n > 1) {
      for (size_t i = 1; i < ell_n; ++i) {
        const CallNode* prev_call = ell_->exprs[i - 1].as<CallNode>();
        const CallNode* call = ell_->exprs[i].as<CallNode>();

        // Merge the CallNodes inside two LetNodes if they have the same target.
        if (ref_ell_->exprs[i - 1].as<CallNode>()->attrs.as<CompilerArgs>()->compiler ==
            ref_ell_->exprs[i].as<CallNode>()->attrs.as<CompilerArgs>()->compiler) {
          // Remove the compiler_begin annotations of the previous CallNode.
          Expr prev_expr = RemoveAnnotation(GetRef<Expr>(prev_call), end_op);
          ell_->exprs[i - 1] = prev_expr;
          // Remove the compiler_end annotations of the present CallNode.
          Expr expr = RemoveAnnotation(GetRef<Expr>(call), begin_op);
          ell_->exprs[i] = expr;
        }
      }

      // Get the new function body from the modified ExplicitLetList.
      Expr new_body = ell_->AsExpr();

      return Function(func->params, new_body, func->ret_type, func->type_params, func->attrs);
    } else {
      return post;
    }
  }

 private:
  ExplicitLetList ell_;
};

Expr MergeCompilerRegions(const Expr& expr) {
  MergeAnnotations merge_anno = MergeAnnotations();
  return PostOrderRewrite(expr, &merge_anno);
}

}  // namespace merge_compiler_regions

ir::Expr MergeCompilerRegions(ir::Expr expr) {
  return merge_compiler_regions::MergeCompilerRegions(expr);
}

MNM_REGISTER_GLOBAL("mnm.pass_.MergeCompilerRegions").set_body_typed(MergeCompilerRegions);

}  // namespace pass
}  // namespace mnm
