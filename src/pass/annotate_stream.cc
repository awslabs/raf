/*!
 * Copyright (c) 2021 by Contributors
 * \file annotate_stream.cc
 * \brief AnnotateStream Pass for CUDA. Annotates a program with CUDA stream assignments,
 * to enable parallization on GPU.
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "./common.h"
#include "../op/schema/stream.h"
#include "./graph_utils.h"

namespace mnm {
namespace pass {
namespace annotate_stream {

using namespace mnm::ir;
using namespace mnm::value;

// Pass: Annotate CUDA Stream
// Description: This pass is used to realize parallelism on GPU. Specifically, it marks the region
// bounded by op stream_start and stream_end to be dispatched to a particular cuda stream, and add
// barriers with stream_wait to enforce synchronization

// Helper class to make stream assignments
class StreamAssigner {
 public:
  // Helper structure to track stream assignmeents of a node's immediate ancestors/children and
  // itself.
  struct IdLabel {
    int32_t stream_id;
    std::unordered_set<int32_t> parents_stream_id;
    std::unordered_set<int32_t> children_stream_id;

    // Print current node's assignment
    std::string DebugDump() {
      std::ostringstream os;
      os << "stream_id = " << stream_id << ", parents_stream_id = [";
      for (auto& it : parents_stream_id) {
        os << it << ", ";
      }
      os << "], children_stream_id =[";
      for (auto& it : children_stream_id) {
        os << it << ", ";
      }
      os << "]\n";
      return os.str();
    }
  };

  explicit StreamAssigner(Arena* arena) : arena_(arena) {
  }

  // TODO(@YY665): A better algorithm that takes load-balancing and stream_reuse (graph coloring)
  // into account
  std::unordered_map<IndexedForwardGraph::Node*, IdLabel*> AssignStreams(
      const IndexedForwardGraph& graph) {
    for (int i = 0; i < graph.post_dfs_order.size(); ++i) {
      IndexedForwardGraph::Node* curr_node = graph.post_dfs_order[i];
      if (!curr_node || !curr_node->ref) continue;
      // If a node has already been assigned with a stream, skip
      if (visited_.count(curr_node)) continue;
      auto* ref = curr_node->ref;

      // Only assign streams to CallNode
      if (GetRef<ObjectRef>(ref).as<CallNode>()) {
        AssignLeftmostPath(curr_node);
      }
    }

    // Populate labels with info from immediate ancestors and children.
    for (int i = 0; i < graph.post_dfs_order.size(); ++i) {
      IndexedForwardGraph::Node* curr_node = graph.post_dfs_order[i];
      if (!curr_node || !curr_node->ref) continue;
      auto* ref = curr_node->ref;
      if (GetRef<ObjectRef>(ref).as<CallNode>()) {
        PopulateLabels(curr_node);
      }
    }
    return std::move(stream_assignment_);
  }

  // Print stream assignment of each CallNode
  std::string DebugDump() {
    std::ostringstream os;
    os << this;
    for (auto& it : stream_assignment_) {
      IndexedForwardGraph::Node* node = it.first;
      IdLabel* label = it.second;
      os << "for node [" << node->index << "], " << label->DebugDump();
    }
    return os.str();
  }

 private:
  Arena* arena_;
  std::unordered_set<IndexedForwardGraph::Node*> visited_;
  std::unordered_map<IndexedForwardGraph::Node*, IdLabel*> stream_assignment_;
  int num_stream_assigned_{0};
  // DFS from leftmost path, assign the entire path with one stream
  // then effectively just remove it from the graph, and do the same on the rest of the graph
  // Since we are iterating in post-dfs order, we start from the src of the path instead of the
  // middle of the path
  void AssignLeftmostPath(IndexedForwardGraph::Node* node) {
    CHECK(node) << "Node cannot be nullptr";
    if (visited_.count(node)) {
      // Every node should only be marked once
      return;
    }
    // Otherwise mark every node down the path
    IdLabel* curr_label = arena_->make<IdLabel>();
    curr_label->stream_id = num_stream_assigned_;
    curr_label->parents_stream_id = std::unordered_set<int32_t>();
    curr_label->children_stream_id = std::unordered_set<int32_t>();
    stream_assignment_.emplace(node, curr_label);
    visited_.emplace(node);
    auto* left_link = node->outputs.head;
    if (!left_link) {
      num_stream_assigned_++;
      return;
    }
    AssignLeftmostPath(left_link->value.node);
  }

  void PopulateLabels(IndexedForwardGraph::Node* node) {
    CHECK(node) << "Node cannot be nullptr";
    // Populate the info of the node itself and its immediate children
    IdLabel* curr_label = stream_assignment_.at(node);
    for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
      IdLabel* child_label = stream_assignment_.at(link->value.node);
      CHECK(child_label != nullptr);
      curr_label->children_stream_id.emplace(child_label->stream_id);
      child_label->parents_stream_id.emplace(curr_label->stream_id);
    }
  }
};

/*
A CallNode would have four states:
State 1: be the source of the path assigned to one particular stream -> annotated with stream_start
State 2: be the sink of the path assigned to one particular stream -> annotated with stream_end
State 3: have prev node running on different stream other than itself -> need to wait ->
stream_wait
State 4: otherwise -> no need to annotate A node can either have a combination of
{State 1, 2, 3} or {State 4} only To check whether a CallNode falls into any of the above category,
we only need to know the stream assignmeents of its immediate ancestors/children and itself.
A set of operations is synchronized with respect to the same CUDA stream in issue order
and is asynchronized with respect to all other streams
Correctly annotated graph would have structure as such:
        stream_start(0)
        |
        f1
  (added event for stream 0)
      /     \
      |     stream_start(1)
      f2    |
      |     stream_wait(0)
      |     |
      |     f4
      |     |
      f3    f5
      |     |
      |     stream_end(1)
      | (added event for stream 1)
      |     |
      \     /
        stream_wait(1)
        |
        f6
        |
        stream_end(0)
*/

class AnnotateStreamRewriter : public ExprRewriter {
 public:
  explicit AnnotateStreamRewriter()
      : stream_tag_(std::move(stream_tag_)),
        start_op_(Op::Get("mnm.op.stream_start")),
        end_op_(Op::Get("mnm.op.stream_end")),
        wait_op_(Op::Get("mnm.op.stream_wait")) {
  }

  Expr AssignStreams(const Expr& body) {
    // Create an IndexedForwardGraph, which is used to capture the dataflow fragment
    graph_ = IndexedForwardGraph::Create(&arena_, body);
    stream_assignment_ = StreamAssigner(&arena_).AssignStreams(graph_);
    return body;
  }

  Expr AnnotateStreamStart(const Expr& ret_expr, int32_t stream_id) {
    // Annotate with stream_start op
    Call ret_call = Downcast<Call>(ret_expr);
    Array<Expr> new_args;
    // Get all args of the original op, append start ops to the end of each arg
    for (auto& arg : ret_call->args) {
      // Don't annotate constant args
      if (arg.as<ConstantNode>()) {
        new_args.push_back(arg);
        continue;
      }
      auto id_arg = MakeConstant(ScalarValue::make(stream_id));
      Expr new_arg = Call(start_op_, {arg, id_arg}, {}, {});
      new_arg->checked_type_ = arg->checked_type_;
      new_args.push_back(new_arg);
    }

    Call new_call = Call(ret_call->op, new_args, ret_call->attrs);
    new_call->checked_type_ = ret_call->checked_type_;
    return std::move(new_call);
  }

  Expr AnnotateStreamEnd(const Expr& ret_expr, int32_t stream_id) {
    // Annotate with stream_end op
    Call ret_call = Downcast<Call>(ret_expr);
    auto id_arg = MakeConstant(ScalarValue::make(stream_id));
    // Add stream_end op after the current op is called, make sure the checked_type is the same
    Call new_call = Call(end_op_, {ret_expr, id_arg}, {}, {});
    new_call->checked_type_ = ret_call->checked_type_;
    return std::move(new_call);
  }

  Expr AnnotateStreamWait(const Expr& ret_expr, std::unordered_set<int32_t> wait_for_id) {
    // Annotate with stream_wait op
    Call ret_call = Downcast<Call>(ret_expr);
    Array<Expr> new_args;
    // Get all args of the original op, append wait ops to the end of each arg
    // Each arg will be annotated with N wait ops, where N = num of streams to wait
    // If a node has already been annotated with stream_start, we put stream_wait after
    // stream_start
    for (auto& arg : ret_call->args) {
      // Don't annotate constant args
      if (arg.as<ConstantNode>()) {
        new_args.push_back(arg);
        continue;
      }

      if (arg.as<CallNode>()) {
        auto op_node = arg.as<CallNode>()->op.as<OpNode>();
        if (op_node != start_op_.operator->() && op_node != end_op_.operator->()) {
          new_args.push_back(arg);
          continue;
        }
      }
      Expr curr_arg_last_expr = arg;
      for (auto& id : wait_for_id) {
        auto id_arg = MakeConstant(ScalarValue::make(id));
        curr_arg_last_expr = Call(wait_op_, {curr_arg_last_expr, id_arg}, {}, {});
      }
      curr_arg_last_expr->checked_type_ = arg->checked_type_;
      new_args.push_back(curr_arg_last_expr);
    }
    Call new_call = Call(ret_call->op, new_args, ret_call->attrs);
    new_call->checked_type_ = ret_call->checked_type_;
    return std::move(new_call);
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    // Annotate CallNode
    IndexedForwardGraph::Node* node = graph_.node_map.at(pre);

    // Check if annotated already && valid op
    Op op = Downcast<Op>(pre->op);
    CHECK_NE(op, start_op_) << "Already annotated with stream_start";
    CHECK_NE(op, end_op_) << "Already annotated with stream_end";
    CHECK_NE(op, wait_op_) << "Already annotated with stream_wait";
    CHECK(op.defined());

    // Fetch stream_assignment of current op
    StreamAssigner::IdLabel* assignment = stream_assignment_.at(node);

    Expr ret_expr = post;

    // If stream id of current node is not annotated on prev nodes, we start a new stream of
    // stream_id
    if (!assignment->parents_stream_id.count(assignment->stream_id)) {
      ret_expr = AnnotateStreamStart(ret_expr, assignment->stream_id);
    }

    // Wait all other streams except for our own stream
    // We'd like to push the barrier as far as possible, so it has to be exactly before the {State
    // 3} node
    std::unordered_set<int32_t> wait_for_id = assignment->parents_stream_id;
    wait_for_id.erase(assignment->stream_id);

    if (!wait_for_id.empty()) {
      ret_expr = AnnotateStreamWait(ret_expr, wait_for_id);
    }

    // If stream id of current node doesn't present in future nodes, we end current stream
    if (!assignment->children_stream_id.count(assignment->stream_id)) {
      ret_expr = AnnotateStreamEnd(ret_expr, assignment->stream_id);
    }
    // TODO(@YY665): To handle multiple outputs
    return ret_expr;
  }

 private:
  /*! \brief The target backends for annotation. */
  ir::Integer stream_tag_;
  const Op& start_op_;
  const Op& end_op_;
  const Op& wait_op_;
  Arena arena_;
  IndexedForwardGraph graph_;
  std::unordered_map<IndexedForwardGraph::Node*, StreamAssigner::IdLabel*> stream_assignment_;
};

// TODO(@YY665): To support max stream limits
Expr AnnotateStream(const Expr& expr, const Integer& stream_tag_) {
  // Initialize rewriter and mark the graph with stream_id
  auto rewriter = AnnotateStreamRewriter();
  rewriter.AssignStreams(expr);
  // Do actual work to rewrite the ops
  return PostOrderRewrite(expr, &rewriter);
}

}  // namespace annotate_stream

Pass AnnotateStream() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(annotate_stream::AnnotateStream(f, -1));
  };
  return CreateMNMFunctionPass(pass_func, 0, "AnnotateStream", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.AnnotateStream").set_body_typed(AnnotateStream);

}  // namespace pass
}  // namespace mnm
