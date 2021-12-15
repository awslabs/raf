/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2021 by Contributors
 * \file dataflow_matcher.cc
 * \brief The auxiliary data structure for dataflow matcher.
 */
#include <mnm/registry.h>

#include "dataflow_matcher_impl.h"

namespace mnm {
namespace pass {

// Pattern Matcher
Array<DFPattern> reverse(const Array<DFPattern>& args) {
  Array<DFPattern> new_args;
  for (auto it = args.rbegin(); it != args.rend(); ++it) {
    new_args.push_back(*it);
  }
  return new_args;
}

bool MNMDFPatternMatcher::VisitDFPattern_(const CallPatternNode* op, const Expr& expr) {
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
        if (base_op->name == "mnm.op.multiply") {
          if (match_args(reverse(op->args), call_node->args)) {
            return true;
          }
        } else if (base_op->name == "mnm.op.add") {
          Array<DFPattern> new_args{op->args[1], op->args[0], op->args[2], op->args[3]};
          if (match_args(new_args, call_node->args)) {
            return true;
          }
        }
      }
    } else {
      ClearMap(watermark);
      // associate divide/multiply, make pattern ((a * b) / c) matchs (a * (b / c)) or ((a / c) * b)
      if (is_pattern_op(op, "mnm.op.divide")) {
        if (const auto* arg_node = op->args[0].as<CallPatternNode>()) {
          if (is_pattern_op(arg_node, "mnm.op.multiply") && is_expr_op(expr, "mnm.op.multiply") &&
              (is_expr_op(call_node->args[0], "mnm.op.divide") ||
               is_expr_op(call_node->args[1], "mnm.op.divide"))) {
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
      if (is_pattern_op(op, "mnm.op.multiply")) {
        // associate multiply/divide, make pattern (a * (b / c)) or ((b / c) * a) matchs
        // ((a * b) / c)
        for (size_t arg_id = 0; arg_id < 2; ++arg_id) {
          if (auto* arg_node = op->args[arg_id].as<CallPatternNode>()) {
            if (is_pattern_op(arg_node, "mnm.op.divide") && is_expr_op(expr, "mnm.op.divide") &&
                (is_expr_op(call_node->args[0], "mnm.op.multiply") ||
                 is_expr_op(call_node->args[1], "mnm.op.multiply"))) {
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

bool MNMDFPatternMatcher::VisitDFPattern_(const RelayConstantPatternNode* op, const Expr& expr) {
  const ConstantPatternNode* node = static_cast<const ConstantPatternNode*>(op);
  auto konst = expr.as<ConstantNode>();
  return konst != nullptr && tvm::StructuralEqual()(konst->value, node->value);
}

bool MNMDFPatternMatcher::VisitDFPattern_(const TypePatternNode* op, const Expr& expr) {
  auto expr_type = InferType(expr).as<ExprNode>()->checked_type();
  return (tvm::StructuralEqual()(op->type, expr_type)) && VisitDFPattern(op->pattern, expr);
}

bool MNMDFPatternMatcher::VisitDFPattern_(const ShapePatternNode* op, const Expr& expr) {
  auto expr_type = InferType(expr).as<ExprNode>()->checked_type();
  if (const TensorTypeNode* tensor_type = expr_type.as<TensorTypeNode>()) {
    return (tvm::StructuralEqual()(op->shape, tensor_type->shape)) &&
           VisitDFPattern(op->pattern, expr);
  }
  return false;
}

bool MNMDFPatternMatcher::VisitDFPattern_(const DataTypePatternNode* op, const Expr& expr) {
  auto expr_type = InferType(expr).as<ExprNode>()->checked_type();
  if (const TensorTypeNode* tensor_type = expr_type.as<TensorTypeNode>()) {
    return (tvm::StructuralEqual()(op->dtype, tensor_type->dtype)) &&
           VisitDFPattern(op->pattern, expr);
  }
  return false;
}

// Pattern Grouper
const std::unordered_map<int, MNMPatternGrouper::Group>& MNMPatternGrouper::GroupMatches(
    const DFPattern& pattern, const Expr& pre) {
  groups_.clear();
  gid_assignments_.clear();

  pattern_ = pattern;
  pattern_graph_ = CreateIndexedGraph(pattern_);
  auto matcher = MNMDFPatternMatcher(pre);
  matcher_ = &matcher;
  this->VisitExprs();
  return this->groups_;
}

// Pattern Rewriter
Expr MNMPatternRewriter::Rewrite(const Array<DFPatternCallback>& callbacks, const Expr& pre) {
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
      auto grouper = MNMPatternGrouper();
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
class MNMPatternPartitioner : protected MixedModeMutator {
 public:
  Expr Partition(const DFPattern& pattern, const Expr& pre, const Map<String, ObjectRef>& attrs,
                 PackedFunc check) {
    if (pattern.as<FunctionPatternNode>()) {
      LOG(WARNING) << "Partioning a Function that isn't called doesn't make sense, skipping"
                   << pattern;
      return pre;
    }
    auto grouper = MNMPatternGrouper();
    groups_ = grouper.GroupMatches(pattern, pre);
    gid_assignments_ = grouper.GetGIDAssignments();
    attrs_ = attrs;
    check_ = check;
    return this->VisitExpr(pre);
  }

 protected:
  Expr RewritePartition(const MNMPatternGrouper::Group& group) {
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
  std::unordered_map<int, MNMPatternGrouper::Group> groups_;
  std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual> gid_assignments_;
  PackedFunc check_;
};

}  // namespace pass

namespace ir {

using namespace pass;

bool MNMMatchPattern(DFPattern pattern, Expr expr) {
  return MNMDFPatternMatcher(expr).Match(pattern, expr);
}

Expr MNMRewritePatterns(Array<DFPatternCallback> callbacks, Expr expr, IRModule mod) {
  return MNMPatternRewriter(mod).Rewrite(callbacks, expr);
}

Expr MNMPartitionPattern(DFPattern pattern, Expr expr, Map<String, ObjectRef> attrs,
                         PackedFunc check) {
  return MNMPatternPartitioner().Partition(pattern, expr, attrs, check);
}

MNM_REGISTER_GLOBAL("mnm.pass_.dataflow_pattern_match").set_body_typed(MNMMatchPattern);
MNM_REGISTER_GLOBAL("mnm.pass_.dataflow_pattern_rewrite").set_body_typed(MNMRewritePatterns);
MNM_REGISTER_GLOBAL("mnm.pass_.dataflow_pattern_partition").set_body_typed(MNMPartitionPattern);

}  // namespace ir

}  // namespace mnm
