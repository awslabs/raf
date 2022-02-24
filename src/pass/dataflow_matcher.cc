/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file dataflow_matcher.cc
 * \brief The auxiliary data structure for dataflow matcher.
 */
#include <raf/registry.h>

#include "dataflow_matcher_impl.h"

namespace raf {
namespace pass {

// Pattern Matcher
Array<DFPattern> reverse(const Array<DFPattern>& args) {
  Array<DFPattern> new_args;
  for (auto it = args.rbegin(); it != args.rend(); ++it) {
    new_args.push_back(*it);
  }
  return new_args;
}

bool RAFDFPatternMatcher::VisitDFPattern_(const CallPatternNode* op, const Expr& expr) {
  // utilities
  auto get_op_node = [](const CallPatternNode* op) -> const tvm::OpNode* {
    if (op) {
      if (auto* expr_pattern = op->op.as<ExprPatternNode>()) {
        return expr_pattern->expr.as<OpNode>();
      }
    }
    return nullptr;
  };
  auto is_pattern_op = [&get_op_node](const CallPatternNode* op, std::string op_type) {
    if (const auto* op_node = get_op_node(op)) {
      auto base_op = GetRef<Op>(op_node);
      base_op = op::IsDialectOp(base_op) ? op::GetBaseOp(base_op) : base_op;
      if (base_op->name == op_type) {
        return true;
      }
    }
    return false;
  };
  auto is_expr_op = [](const Expr& expr, std::string op_type) {
    if (const auto* call_node = expr.as<CallNode>()) {
      if (const auto* op_node = call_node->op.as<OpNode>()) {
        auto base_op = GetRef<Op>(op_node);
        base_op = op::IsDialectOp(base_op) ? op::GetBaseOp(base_op) : base_op;
        if (base_op->name == op_type) {
          return true;
        }
      }
    }
    return false;
  };

  // logic
  auto watermark = matched_nodes_.size();
  if (const auto* call_node = expr.as<CallNode>()) {
    auto matches_op = VisitDFPattern(op->op, call_node->op);
    if (matches_op) {
      auto watermark2 = matched_nodes_.size();

      auto match_args = [this, &watermark2](const Array<DFPattern> pattern_args,
                                            const Array<Expr> expr_args) {
        bool matches = true;
        size_t i = 0;
        if (pattern_args.defined()) {
          if (pattern_args.size() == expr_args.size()) {
            while (matches && i < pattern_args.size()) {
              matches &= VisitDFPattern(pattern_args[i], expr_args[i]);
              ++i;
            }
          } else {
            matches = false;
          }
        }
        if (!matches) {
          ClearMap(watermark2);
        }
        return matches;
      };

      // Standard case
      if (match_args(op->args, call_node->args)) {
        return true;
      }
      // Commutative Matching
      if (const OpNode* op_node = get_op_node(op)) {
        auto base_op = GetRef<Op>(op_node);
        base_op = op::IsDialectOp(base_op) ? op::GetBaseOp(base_op) : base_op;
        if (base_op->name == "raf.op.multiply") {
          if (match_args(reverse(op->args), call_node->args)) {
            return true;
          }
        } else if (base_op->name == "raf.op.add") {
          Array<DFPattern> new_args{op->args[1], op->args[0], op->args[2], op->args[3]};
          if (match_args(new_args, call_node->args)) {
            return true;
          }
        }
      }
    } else {
      ClearMap(watermark);
      // associate divide/multiply, make pattern ((a * b) / c) matchs (a * (b / c)) or ((a / c) * b)
      if (is_pattern_op(op, "raf.op.divide")) {
        if (const auto* arg_node = op->args[0].as<CallPatternNode>()) {
          if (is_pattern_op(arg_node, "raf.op.multiply") && is_expr_op(expr, "raf.op.multiply") &&
              (is_expr_op(call_node->args[0], "raf.op.divide") ||
               is_expr_op(call_node->args[1], "raf.op.divide"))) {
            bool out = false;
            for (size_t arg_id = 0; arg_id < 2; ++arg_id) {
              auto div = CallPattern(op->op, {arg_node->args[arg_id], op->args[1]});
              auto mul = CallPattern(arg_node->op, {arg_node->args[(arg_id + 1) % 2], div});
              out = VisitDFPattern(mul, expr);
              if (out) {
                return true;
              } else {
                ClearMap(watermark);
              }
            }
            return out;
          }
        }
      }
      if (is_pattern_op(op, "raf.op.multiply")) {
        // associate multiply/divide, make pattern (a * (b / c)) or ((b / c) * a) matchs
        // ((a * b) / c)
        for (size_t arg_id = 0; arg_id < 2; ++arg_id) {
          if (auto* arg_node = op->args[arg_id].as<CallPatternNode>()) {
            if (is_pattern_op(arg_node, "raf.op.divide") && is_expr_op(expr, "raf.op.divide") &&
                (is_expr_op(call_node->args[0], "raf.op.multiply") ||
                 is_expr_op(call_node->args[1], "raf.op.multiply"))) {
              auto mul = CallPattern(op->op, {arg_node->args[0], op->args[(arg_id + 1) % 2]});
              auto div = CallPattern(arg_node->op, {mul, arg_node->args[1]});
              return VisitDFPattern(div, expr);
            }
          }
        }
      }
    }
  }
  return false;
}

bool RAFDFPatternMatcher::VisitDFPattern_(const RelayConstantPatternNode* op, const Expr& expr) {
  const ConstantPatternNode* node = static_cast<const ConstantPatternNode*>(op);
  auto konst = expr.as<ConstantNode>();
  return konst != nullptr && tvm::StructuralEqual()(konst->value, node->value);
}

bool RAFDFPatternMatcher::VisitDFPattern_(const TypePatternNode* op, const Expr& expr) {
  auto expr_type = InferType(expr).as<ExprNode>()->checked_type();
  return (tvm::StructuralEqual()(op->type, expr_type)) && VisitDFPattern(op->pattern, expr);
}

bool RAFDFPatternMatcher::VisitDFPattern_(const ShapePatternNode* op, const Expr& expr) {
  auto expr_type = InferType(expr).as<ExprNode>()->checked_type();
  if (const TensorTypeNode* tensor_type = expr_type.as<TensorTypeNode>()) {
    return (tvm::StructuralEqual()(op->shape, tensor_type->shape)) &&
           VisitDFPattern(op->pattern, expr);
  }
  return false;
}

bool RAFDFPatternMatcher::VisitDFPattern_(const DataTypePatternNode* op, const Expr& expr) {
  auto expr_type = InferType(expr).as<ExprNode>()->checked_type();
  if (const TensorTypeNode* tensor_type = expr_type.as<TensorTypeNode>()) {
    return (tvm::StructuralEqual()(op->dtype, tensor_type->dtype)) &&
           VisitDFPattern(op->pattern, expr);
  }
  return false;
}

// Pattern Grouper
const std::unordered_map<int, RAFPatternGrouper::Group>& RAFPatternGrouper::GroupMatches(
    const DFPattern& pattern, const Expr& pre) {
  groups_.clear();
  gid_assignments_.clear();

  pattern_ = pattern;
  pattern_graph_ = CreateIndexedGraph(pattern_);
  auto matcher = RAFDFPatternMatcher(pre);
  matcher_ = &matcher;
  this->VisitExprs();
  return this->groups_;
}

// Pattern Rewriter
Expr RAFPatternRewriter::Rewrite(const Array<DFPatternCallback>& callbacks, const Expr& pre) {
  auto post = pre;
  auto last = post;
  // rewrite the graph until it stops changing to make sure all rewrites are complete
  int count = 0;
  bool equal = true;
  static auto* structural_equal = tvm::runtime::Registry::Get("node.StructuralEqual");
  ICHECK(structural_equal) << "node.StructuralEqual is not registered.";
  do {
    last = post;
    for (auto callback : callbacks) {
      callback_ = callback;
      if (callback_->require_type) {
        post = InferTypeWithModule(post, mod_);
      }
      auto grouper = RAFPatternGrouper();
      groups_ = grouper.GroupMatches(callback_->pattern, post);
      gid_assignments_ = grouper.GetGIDAssignments();
      memo_.clear();
      post = this->VisitExpr(post);
      count++;
    }
    equal = (*structural_equal)(last, post, false, true);
  } while (!equal && count < 100 && !callback_->rewrite_once);
  if (count >= 100) {
    LOG(FATAL) << "Observed 100 rewrite passes, possible conflicting passes?";
  }
  return post;
}

// Pattern Partitioner
class RAFPatternPartitioner : protected MixedModeMutator {
 public:
  Expr Partition(const DFPattern& pattern, const Expr& pre, const Map<String, ObjectRef>& attrs,
                 PackedFunc check) {
    if (pattern.as<FunctionPatternNode>()) {
      LOG(WARNING) << "Partioning a Function that isn't called doesn't make sense, skipping"
                   << pattern;
      return pre;
    }
    auto grouper = RAFPatternGrouper();
    groups_ = grouper.GroupMatches(pattern, pre);
    gid_assignments_ = grouper.GetGIDAssignments();
    attrs_ = attrs;
    check_ = check;
    return this->VisitExpr(pre);
  }

 protected:
  Expr RewritePartition(const RAFPatternGrouper::Group& group) {
    Array<Expr> args;
    for (size_t i = 0; i < group.args.size(); ++i) {
      args.push_back(memo_[group.args[i]]);
    }
    Function func = WithAttr(group.function, attr::kPartitionedFromPattern, String(group.name));
    if (!attrs_.empty()) {
      for (auto kv : attrs_) {
        func = WithAttr(std::move(func), kv.first, kv.second);
      }
    }
    return Call(func, args);
  }

  Expr DispatchVisitExpr(const Expr& pre) override {
    auto post = MixedModeMutator::DispatchVisitExpr(pre);
    if (gid_assignments_.count(pre) && pre == groups_[gid_assignments_[pre]].root_node &&
        static_cast<bool>(check_(pre))) {
      post = RewritePartition(groups_[gid_assignments_[pre]]);
    }
    return post;
  }

  Map<String, ObjectRef> attrs_;
  std::unordered_map<int, RAFPatternGrouper::Group> groups_;
  std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual> gid_assignments_;
  PackedFunc check_;
};

}  // namespace pass

namespace ir {

using namespace pass;

bool RAFMatchPattern(DFPattern pattern, Expr expr) {
  return RAFDFPatternMatcher(expr).Match(pattern, expr);
}

Expr RAFRewritePatterns(Array<DFPatternCallback> callbacks, Expr expr, IRModule mod) {
  return RAFPatternRewriter(mod).Rewrite(callbacks, expr);
}

Expr RAFPartitionPattern(DFPattern pattern, Expr expr, Map<String, ObjectRef> attrs,
                         PackedFunc check) {
  return RAFPatternPartitioner().Partition(pattern, expr, attrs, check);
}

RAF_REGISTER_GLOBAL("raf.pass_.dataflow_pattern_match").set_body_typed(RAFMatchPattern);
RAF_REGISTER_GLOBAL("raf.pass_.dataflow_pattern_rewrite").set_body_typed(RAFRewritePatterns);
RAF_REGISTER_GLOBAL("raf.pass_.dataflow_pattern_partition").set_body_typed(RAFPartitionPattern);

}  // namespace ir

}  // namespace raf
