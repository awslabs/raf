/*!
 * Copyright (c) 2020 by Contributors
 * \file annotate_target.cc
 * \brief Annotate Target
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "./common.h"
#include "../op/schema/annotation.h"

namespace mnm {
namespace pass {
namespace annotate_target {

using namespace mnm::ir;
using namespace tvm;
using namespace tvm::relay;
using mnm::op::FMNMAnnotateTarget;
using mnm::op::schema::CompilerArgs;

static const Op& begin_op = CompilerBeginOp();
static const Op& end_op = CompilerEndOp();

// A helper class to insert annotation boundaries for a program region that will
// be handled by a specific compiler.
class AnnotateTargetRewriter : public ExprRewriter {
 public:
  explicit AnnotateTargetRewriter(Array<ir::String> targets) : targets_(std::move(targets)) {
  }

  Expr InsertAnnotation(const Expr& expr, const std::string& target, const Op& ann_op) {
    auto attrs = make_object<CompilerArgs>();
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

    // Rewrite the Meta op CallNode.
    // Check which targets this op can be offloaded
    if (op_node) {
      // Check target specific op and add to supported_targets if it is supported.
      Op op = Downcast<Op>(pre->op);
      CHECK(op.defined());
      for (const auto& target : this->targets_) {
        if (!Op::HasAttrMap("target." + std::string(target))) {
          continue;
        }
        auto fannotate = Op::GetAttrMap<FMNMAnnotateTarget>("target." + std::string(target));
        if (fannotate.count(op) && fannotate[op](pre->attrs, pre->args)) {
          supported_targets.push_back(target);
        }
      }
    } else if (pre->op->IsInstance<FunctionNode>()) {
      // Composite Function Not Supported in Meta.
      Function func = Downcast<Function>(pre->op);
      CHECK(func.defined());

      CHECK(func->GetAttr<String>(tvm::relay::attr::kComposite))
          << "NotImplementedError: Composite Function Not Supported in Meta";
    }
    supported_targets.push_back("default");  // Make default as the last option.

    // TODO(@XIAO-XIA): Now we simply assign this node to the target with
    // the highest priority, but we should preserve all supported targets so that
    // we can make a better decision.
    std::string target = supported_targets[0];

    // Visit and mutate arguments after the target of this op has been determined.
    Call post_call = Downcast<Call>(post);

    // Add compiler_begin to the arguments of the Meta op CallNode
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

 private:
  /*! \brief The target backends for annotation. */
  Array<ir::String> targets_;
};

Expr AnnotateTarget(const Expr& expr, const Array<String>& targets) {
  auto rewriter = AnnotateTargetRewriter(targets);
  return PostOrderRewrite(expr, &rewriter);
}

}  // namespace annotate_target

ir::Expr AnnotateTarget(ir::Expr expr, ir::Array<ir::String> targets) {
  return annotate_target::AnnotateTarget(expr, targets);
}

MNM_REGISTER_GLOBAL("mnm.pass_.AnnotateTarget").set_body_typed(AnnotateTarget);

}  // namespace pass
}  // namespace mnm
