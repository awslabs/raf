/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file annotate_target.cc
 * \brief Annotate Target
 */
#include <tvm/ir/transform.h>
#include <tvm/relay/transform.h>

#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "./common.h"
#include "../op/dialect/tvm/tvm_attrs.h"

namespace raf {
namespace pass {
namespace annotate_target {

using namespace raf::ir;
using raf::op::FRAFAnnotateTarget;
using raf::op::tvm_dialect::CompilerAttrs;

static const Op& begin_op = CompilerBeginOp();
static const Op& end_op = CompilerEndOp();

// A helper class to insert annotation boundaries for a program region that will
// be handled by a specific compiler.
class AnnotateTargetRewriter : public ExprRewriter {
 public:
  explicit AnnotateTargetRewriter(Array<ir::String> targets) : targets_(std::move(targets)) {
  }

  Expr InsertAnnotation(const Expr& expr, const std::string& target, const Op& ann_op) {
    auto attrs = make_object<CompilerAttrs>();
    attrs->compiler = target;
    Expr new_op = {Call(ann_op, {expr}, Attrs(attrs), {})};
    new_op->checked_type_ = expr->checked_type_;
    return new_op;
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    // Supported targets for this node. The order implies the priority.
    std::vector<std::string> supported_targets;

    auto op_node = pre->op.as<OpNode>();

    // Make sure the ir has not been annotated.
    CHECK_NE(pre->op, CompilerBeginOp()) << "ValueError: this ir has been annotated already";
    CHECK_NE(pre->op, CompilerEndOp()) << "ValueError: this ir has been annotated already";

    // Rewrite the RAF op CallNode.
    // Check which targets this op can be offloaded
    if (op_node) {
      // Check target specific op and add to supported_targets if it is supported.
      Op op = Downcast<Op>(pre->op);
      CHECK(op.defined());
      for (const auto& target : this->targets_) {
        if (!Op::HasAttrMap("target." + std::string(target))) {
          continue;
        }
        auto fannotate = Op::GetAttrMap<FRAFAnnotateTarget>("target." + std::string(target));
        if (fannotate.count(op) && fannotate[op](pre->attrs, pre->args)) {
          supported_targets.push_back(target);
        }
      }
    } else if (pre->op->IsInstance<FunctionNode>()) {
      // Composite Function Not Supported in RAF.
      Function func = Downcast<Function>(pre->op);
      CHECK(func.defined());

      CHECK(func->GetAttr<String>(attr::kComposite))
          << "NotImplementedError: Composite Function Not Supported in RAF";
    }
    supported_targets.push_back("default");  // Make default as the last option.

    // TODO(@XIAO-XIA): Now we simply assign this node to the target with
    // the highest priority, but we should preserve all supported targets so that
    // we can make a better decision.
    std::string target = supported_targets[0];

    // Visit and mutate arguments after the target of this op has been determined.
    Call post_call = Downcast<Call>(post);

    // Add compiler_begin to the arguments of the RAF op CallNode
    Array<Expr> new_args;
    for (auto& arg : post_call->args) {
      Expr new_arg = InsertAnnotation(arg, target, begin_op);
      new_args.push_back(new_arg);
    }
    Call new_call = Call(post_call->op, new_args, post_call->attrs);
    // TODO(@XIAO-XIA): To support multiple output, we could change the way
    // of adding compiler_end annotation.
    // Add compiler_end to this CallNode
    Expr ret_expr = InsertAnnotation(new_call, target, end_op);
    Call ret_call = Downcast<Call>(ret_expr);
    ret_call->checked_type_ = new_call->checked_type_;

    return std::move(ret_call);
  }

  Expr Rewrite_(const LetNode* let, const Expr& post) final {
    // Annotate TupleNode
    if (let->value.as<TupleNode>()) {
      auto ell_ = ExplicitLetList::make(let->body);
      auto tuple_var = let->var;

      // The target of TupleNode should be the same as the ops whose args
      // contain it. If there are several ops using it as args, push these
      // targets into supported_targets from near to far.
      std::vector<std::string> supported_targets;
      for (auto expr : ell_->exprs) {
        if (expr.as<CallNode>()) {
          // Check if the args of CallNode contain TupleVar.
          const CallNode* call = expr.as<CallNode>();
          // Make sure this LetNode has not been annotated before.
          CHECK_NE(call->op, CompilerBeginOp()) << "ValueError: this ir has been annotated already";
          CHECK_NE(call->op, CompilerEndOp()) << "ValueError: this ir has been annotated already";

          for (auto arg : call->args) {
            if (arg == tuple_var) {
              // Find out which target this op support and push it back to supported_targets.
              Op op = Downcast<Op>(call->op);
              CHECK(op.defined());
              for (const auto& target : this->targets_) {
                if (!Op::HasAttrMap("target." + std::string(target))) {
                  continue;
                }
                auto fannotate =
                    Op::GetAttrMap<FRAFAnnotateTarget>("target." + std::string(target));
                if (fannotate.count(op) && fannotate[op](call->attrs, call->args)) {
                  supported_targets.push_back(target);
                }
              }
              break;
            }
          }
        }
      }
      supported_targets.push_back("default");  // Make default as the last option.

      // Annotate this LetNode with the top priority target.
      std::string target = supported_targets[0];

      // Annotate the fields of TupleNode with compiler_begin CallNode.
      const TupleNode* tuple = let->value.as<TupleNode>();
      Array<Expr> new_fields;
      for (auto& field : tuple->fields) {
        Expr new_field = InsertAnnotation(field, target, begin_op);
        new_fields.push_back(new_field);
      }
      Tuple new_tuple = Tuple(new_fields);
      Expr ret_expr = InsertAnnotation(new_tuple, target, end_op);
      Call ret_call = Downcast<Call>(ret_expr);
      ret_call->checked_type_ = new_tuple->checked_type_;

      Let post_let = Downcast<Let>(post);

      return std::move(Let(post_let->var, ret_call, post_let->body));
    } else {
      return std::move(post);
    }
  }

 private:
  /*! \brief The target backends for annotation. */
  Array<ir::String> targets_;
};

Expr AnnotateTarget(const Expr& expr, const Array<ir::String>& targets) {
  auto rewriter = AnnotateTargetRewriter(targets);
  return PostOrderRewrite(expr, &rewriter);
}

}  // namespace annotate_target

Pass AnnotateTarget(Array<ir::String> targets) {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(annotate_target::AnnotateTarget(f, targets));
  };
  return CreateRAFFunctionPass(pass_func, 0, "AnnotateTargetFunc", {"raf.pass_.InferType"});
}

RAF_REGISTER_GLOBAL("raf.pass_.AnnotateTarget").set_body_typed(AnnotateTarget);

}  // namespace pass
}  // namespace raf
