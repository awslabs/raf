/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/stream_schedule_asap.cc
 * \brief ASAP (As Soon As Possible) stream scheduler.
 */
#include <relay/transforms/pass_utils.h>
#include "raf/pass.h"
#include "raf/analysis.h"
#include "./stream_schedule.h"

namespace raf {
namespace pass {
namespace asap_stream_schedule {

using namespace raf::analysis;
using stream_schedule::StreamSchedulerBase;
using Node = DependencyGraph::Node;

class ASAPScheduler : public StreamSchedulerBase {
 public:
  /*!
   * \brief We implement a as-soon-as-possible(ASAP) schedule. We describe the ASAP schedule as
   * follows briefly:
   *
   * 1. Construct the dependency graph, pruning the atomic nodes and redundant edges. Refer
   *    raf::analysis::CreateDependencyGraph for the definition of atomic nodes and redundant edges.
   *
   * 2. Classify all dependency edges in the dependency graph into two categories: heavy edge and
   *    light edge. Each node can have at most one heavy edge points to it and at most one heavy
   *    edge points out from it. We call the node it depends on through a heavy edge heavy child,
   *    and the node that depends on it heavy parent. Other nodes it depends on are light children
   *    and other nodes that depend on it light parents. After the edge classification, we partition
   *    the dependency graph into multiple heavy chains that connected by heavy edges.
   *
   * 3. We would launch each heavy chain on a cuda stream. Let A and B be two heavy chains. If all
   *    nodes in A are before all nodes in B in dependency graph (i.e., for all a in A and b in B,
   *    there is a path from a to b), we may reuse the stream used to launch heavy chain A as the
   *    cuda stream to launch heavy chain B. For each node that has light parents, we will add an
   *    event for that node. Before we launch a node with light children, we will wait for the
   *    events of light children.
   */
  Expr Schedule(const Expr& e) {
    // Initialize dependency graph of e
    InitDependencyGraph(e);

    // Start to schedule
    int event_id_clock = 0;
    std::unordered_map<Node*, int> finish_event;
    std::unordered_map<Node*, int> node_stream;
    std::vector<Node*> schedule_order = GetScheduleOrder();
    for (size_t i = 0; i < schedule_order.size(); i++) {
      auto node = schedule_order[i];

      // Get the stream to launch this node
      int stream_id;
      if (info_[node].heavy_child) {
        stream_id = node_stream[info_[node].heavy_child];
      } else {
        stream_id = GetNewStream(node);
      }
      node_stream[node] = stream_id;

      // Annotate SetStream when it is the first node or last issued node is not its heavy child
      if (i == 0 || schedule_order[i - 1] != info_[node].heavy_child) {
        AnnotateSetStream(0, stream_id);
      }

      // Wait dependent events
      for (auto iit = node->children.head; iit; iit = iit->next) {
        Node* child = iit->value;
        if (child == info_[node].heavy_child) {
          continue;
        }
        CHECK_GT(finish_event.count(child), 0);
        AnnotateWaitEvent(finish_event[child]);
      }

      // Issue this node
      VisitExpr(info_[node].expr);

      // Annotate AddEvent if current node has light parents
      if (info_[node].num_parents - (info_[node].heavy_parent != nullptr) > 0) {
        finish_event[node] = event_id_clock++;
        AnnotateAddEvent(finish_event[node]);
      }

      // Mark this node as the last node in stream_id if it does not have heavy parent
      if (!info_[node].heavy_parent) {
        SetStreamLastNode(stream_id, node);
      }
    }

    return let_list_.Get(VisitExpr(e));
  }

 private:
  /*!
   * \brief Initialize the dependency graph and fill the node info.
   *
   * It determines the heavy edge. The nodes connected by heavy edge will be launched on the same
   * stream. If there is an edge a -> b in dependency graph where a depends on b. We call a is the
   * heavy parent of b and b is the heavy child of a. We call all other edges light edges. There is
   * a synchronization for each light edge.
   *
   * The strategy to determine the heavy edges used here is trying to assign the heavy parent for
   * node with larger length first. We define the length of each node as the maximum number of nodes
   * of each path starting from that node.
   */
  void InitDependencyGraph(Expr e) {
    dg_ = CreateDependencyGraph(&arena_, e, true, true);

    // Get the expression of each node
    for (auto& it : dg_.expr_node) {
      const Node* node;
      Expr expr;
      std::tie(expr, node) = it;
      info_[node].expr = expr;
    }

    // Get the depth and number of parents of each node.
    for (size_t i = dg_.post_dfs_order.size(); i != 0; i--) {
      Node* node = dg_.post_dfs_order[i - 1];
      info_[node].depth = 1;
      info_[node].num_parents = 0;
      for (auto iit = node->parents.head; iit; iit = iit->next) {
        auto parent = iit->value;
        info_[node].depth = std::max(info_[node].depth, info_[parent].depth + 1);
        info_[node].num_parents += 1;
      }
    }

    // Sort the nodes in descending order, taking node depth as comparison key
    std::vector<Node*> depth_order(dg_.post_dfs_order);
    auto node_compare_dec = [&](const Node* lhs, const Node* rhs) {
      return this->info_[lhs].depth > this->info_[rhs].depth;
    };
    std::sort(depth_order.begin(), depth_order.end(), node_compare_dec);

    // Determine the heavy parent and heavy child of each node, if exists
    for (auto node : depth_order) {
      for (auto iit = node->children.head; iit; iit = iit->next) {
        Node* child = iit->value;
        if (info_[child].heavy_parent == nullptr) {
          info_[child].heavy_parent = node;
          info_[node].heavy_child = child;
          break;
        }
      }
    }
  }

  /*!
   * \brief Get a schedule order. Any topological order of the dependency graph is a valid order.
   * Here we implement an order that tries to minimize the number of SetStream operators.
   */
  std::vector<Node*> GetScheduleOrder() {
    // Get all ready-to-execute nodes
    std::unordered_map<Node*, int> out_degree;
    for (auto node : dg_.post_dfs_order) {
      for (auto iit = node->children.head; iit; iit = iit->next) {
        out_degree[node]++;
      }
    }
    std::vector<Node*> stack;
    for (auto node : dg_.post_dfs_order) {
      if (out_degree[node] == 0) {
        stack.push_back(node);
      }
    }
    auto node_compare_inc = [&](const Node* lhs, const Node* rhs) {
      return this->info_[lhs].depth < this->info_[rhs].depth;
    };
    // We first launch operator with larger depth
    std::sort(stack.begin(), stack.end(), node_compare_inc);
    std::vector<Node*> schedule_order;

    // Push each node into schedule_order until all nodes have been pushed.
    while (!stack.empty()) {
      Node* node = stack.back();
      stack.pop_back();
      schedule_order.push_back(node);

      // Push new available nodes.
      for (auto iit = node->parents.head; iit; iit = iit->next) {
        Node* parent = iit->value;
        if (parent == info_[node].heavy_parent) {
          // process heavy parent outside the loop
          continue;
        }
        if (--out_degree[parent] == 0) {
          stack.push_back(parent);
        }
      }
      std::sort(stack.begin(), stack.end(), node_compare_inc);
      if (info_[node].heavy_parent) {
        if (--out_degree[info_[node].heavy_parent] == 0) {
          // If current node has heavy parent, and all its dependent nodes have been executed,
          // put it at the end of stack to avoid stream switch (i.e. calling SetStream).
          stack.push_back(info_[node].heavy_parent);
        }
      }
    }
    return schedule_order;
  }

  /*!
   * \brief Get a new stream to launch given node. We first try to find an existing stream such that
   * the last node of that stream can reach given node. This means we can reuse this stream without
   * sacrifice any performance degradation. If we can not find such a stream, we allocate a new
   * stream.
   */
  int GetNewStream(Node* node) {
    for (auto kv : stream_last_node_) {
      if (kv.second == nullptr) {
        // There are nodes on this stream not been issued.
        continue;
      }
      CHECK_GE(ancestors_.count(kv.second), 1);
      if (ancestors_[kv.second].count(node)) {
        stream_last_node_[kv.first] = nullptr;
        ancestors_[kv.second].clear();
        return kv.first;
      }
    }
    // Allocate a new stream
    int stream_id = static_cast<int>(stream_last_node_.size()) + 1;
    stream_last_node_[stream_id] = nullptr;
    return stream_id;
  }
  /*!
   * \brief Mark node as the last node in given stream. We compute all ancestors of given node,
   * which helps to determine whether we can reuse this stream when we need new stream.
   */
  void SetStreamLastNode(int stream_id, Node* node) {
    stream_last_node_[stream_id] = node;
    std::unordered_set<Node*>& ancestors = ancestors_[node];
    std::vector<Node*> qu;
    qu.push_back(node);
    ancestors.insert(node);
    while (!qu.empty()) {
      Node* nd = qu.back();
      qu.pop_back();
      for (auto iit = nd->parents.head; iit; iit = iit->next) {
        Node* parent = iit->value;
        if (!ancestors.count(parent)) {
          qu.push_back(parent);
          ancestors.insert(parent);
        }
      }
    }
  }

  struct NodeInfo {
    /*! \brief The expression corresponding to the node. */
    Expr expr;
    /*! \brief The maximum number of nodes of all paths starting from the node to sink node. */
    int depth{};
    /*! \brief The number of parents. */
    int num_parents{};
    /*! \brief The heavy parent of the node, may be nullptr if it does not exist. */
    Node* heavy_parent{};
    /*! \brief The heavy child of the node, may be nullptr if it does not exist. */
    Node* heavy_child{};
  };
  /*! \brief Arena used to allocate memory for dependency graph. */
  Arena arena_;
  /*! \brief Dependency graph of given expr. */
  DependencyGraph dg_;
  /*! \brief The info for each node. */
  std::unordered_map<const Node*, NodeInfo> info_;
  /*! \brief The last node of each stream. If we have not issues all nodes in the stream, the value
   * of the stream in this map is nullptr. Used for stream allocation and recycle. */
  std::unordered_map<int, Node*> stream_last_node_;
  /*! \brief All ancestors of a node in the dependency graph. Used for stream allocation and
   * recycle. */
  std::unordered_map<Node*, std::unordered_set<Node*>> ancestors_;
};

Expr ASAPStreamSchedule(const Expr& e) {
  return ASAPScheduler().Schedule(e);
}

}  // namespace asap_stream_schedule

Pass ASAPStreamSchedule() {
  pass::PassContext pass_ctx = pass::PassContext::Current();
  tvm::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto transform = asap_stream_schedule::ASAPStreamSchedule;
        return Downcast<Function>(tvm::relay::TransformF(transform, f));
      };
  return CreateRAFFunctionPass(pass_func, 1, "ASAPStreamSchedule", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.ASAPStreamSchedule").set_body_typed(ASAPStreamSchedule);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.stream_schedule.policy", tvm::String);

}  // namespace pass
}  // namespace raf
