/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
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
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "./common.h"
#include "../op/dialect/tvm/tvm_attrs.h"

namespace raf {
namespace pass {
namespace merge_compiler_regions {

using namespace raf::ir;
using raf::op::tvm_dialect::CompilerAttrs;

static const Op& begin_op = CompilerBeginOp();
static const Op& end_op = CompilerEndOp();

class MergeAnnotations : public ExprRewriter {
 public:
  explicit MergeAnnotations() {
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
        if (ref_ell_->exprs[i - 1].as<CallNode>()->attrs.as<CompilerAttrs>()->compiler ==
            ref_ell_->exprs[i].as<CallNode>()->attrs.as<CompilerAttrs>()->compiler) {
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

Pass MergeCompilerRegions() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(merge_compiler_regions::MergeCompilerRegions(f));
  };
  return CreateRAFFunctionPass(pass_func, 0, "MergeCompilerRegions", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.MergeCompilerRegions").set_body_typed(MergeCompilerRegions);

}  // namespace pass
}  // namespace raf
