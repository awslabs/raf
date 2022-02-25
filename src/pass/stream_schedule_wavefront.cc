/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/stream_schedule_wavefront.cc
 * \brief Wavefront stream scheduler.
 */
#include <relay/transforms/pass_utils.h>
#include "raf/pass.h"
#include "raf/analysis.h"
#include "./stream_schedule.h"
#include "../analysis/dependency_graph.h"

namespace raf {
namespace pass {
namespace wavefront_stream_schedule {

using namespace raf::analysis;
using stream_schedule::StreamSchedulerBase;
using Node = DependencyGraph::Node;
using NodeExprMap = std::unordered_map<const Node*, Expr>;
using analysis::dependency_graph::GetListSize;

/*! Chain, Wave, and Partition are used to describe a wavefront schedule. */
using Chain = std::vector<Node*>;
using Wave = std::vector<Chain>;
using Partition = std::vector<Wave>;

/*!
 * \brief Partition the dependency graph into waves of operator chains.
 * \param dg The dependency graph we want to partition.
 * \return The wavefront partition.
 */
Partition WavefrontPartition(DependencyGraph* dg) {
  std::vector<Node*>& nodes = dg->post_dfs_order;

  std::unordered_map<Node*, int> in_degree, out_degree;
  for (auto& node : nodes) {
    for (auto child = node->children.head; child; child = child->next) {
      Node* child_node = child->value;
      in_degree[child_node]++;
      out_degree[node]++;
    }
  }

  std::vector<Node*> free_nodes;
  for (auto& node : nodes) {
    if (out_degree[node] == 0) {
      free_nodes.push_back(node);
    }
  }

  Partition partition;
  while (!free_nodes.empty()) {
    Wave wave;

    for (auto node : free_nodes) {
      // Each free node corresponds to a chain
      // There are three cases of the number of the free node's parents
      // case 1. no parent
      // case 2. one parents
      // case 3. two or more parents
      Chain chain;
      if (GetListSize(node->parents) != 1) {
        // case 1 and case 3. There is only the free node in this chain
        chain.push_back(node);
      } else {
        // case 2. There are more than one nodes in this chain, starting from the free node
        chain.push_back(node);
        Node* next_node = node->parents.head->value;
        while (GetListSize(next_node->parents) == 1 && out_degree[next_node] == 1) {
          chain.push_back(next_node);
          node = next_node;
          next_node = node->parents.head->value;
          CHECK_GE(out_degree[next_node], 1);
        }
        // There are three sub cases to stop growing this chain:
        // sub case 1. There are two or more nodes next_node depends on after ignoring previous
        //             waves (out_degree[next_node] > 1).
        // sub case 2. The number of nodes that depends on next_node does not equal to 1
        //             (GetListSize(next_node->parents) != 1).
        // sub case 3. Both of sub case 1 and sub case 2.
        // For sub case 2, we should also take next_node into this chain.
        if (out_degree[next_node] == 1) {
          chain.push_back(next_node);
        }
      }
      wave.push_back(chain);
    }
    free_nodes.clear();
    for (auto& chain : wave) {
      Node* last_node = chain.back();
      for (auto parent = last_node->parents.head; parent; parent = parent->next) {
        Node* parent_node = parent->value;
        if (--out_degree[parent_node] == 0) {
          free_nodes.push_back(parent_node);
        }
      }
    }
    partition.push_back(wave);
  }
  return partition;
}

class WavefrontScheduler : public StreamSchedulerBase {
 public:
  /*!
   * Generate the wavefront stream schedule. The input expr e is a dataflow graph in GNF format and
   * the output is the scheduled e in ANF.
   *
   * There are two steps in the schedule function:
   *
   *  step 1. Create a dataflow graph of input expr. We can get the dataflow graph by removing the
   *          nodes in dependency graph whose corresponding expr is atomic. Here atomic
   *          expr is an expr that does not influence the data flow graph structure. After this
   * step, the remaining nodes in the dataflow graph are CallNode, TupleNode, TupleGetItemNode.
   *
   *  step 2. Use the dataflow graph got in step 1 to issue the operator call in a schedule-specific
   *          order. Meanwhile, it would inject raf.op.set_stream, raf.op.add_event, and
   *          raf.op.wait_event operators to manage the synchronization.
   *
   *  When we finish the above two steps, we get the ANF of the scheduled computation graph.
   *
   * \param e The expr that we want to schedule. It should be a pure dataflow graph expr and should
   *          not contains any node that introduces new scope (such as FunctionNode, LetNode, and
   *          IfNode).
   *
   * \return The schedule expr. Schedule-related operators have been injected.
   */
  Expr Schedule(const Expr& e) {
    Arena arena;
    DependencyGraph dg = CreateDependencyGraph(&arena, e, true, true);

    NodeExprMap node_expr;
    for (auto& it : dg.expr_node) {
      node_expr[it.second] = it.first;
    }

    Partition partition = WavefrontPartition(&dg);

    for (int i = 0; i < partition.size(); i++) {
      Wave& wave = partition.at(i);
      for (int j = 0; j < wave.size(); j++) {
        Chain& chain = wave[j];
        AnnotateSetStream(0, j);
        for (Node* node : chain) {
          Expr expr = node_expr.at(node);
          VisitExpr(expr);
        }
      }
      if (i + 1 < partition.size()) {
        AnnotateStreamBarrier();
      }
    }
    return let_list_.Get(VisitExpr(e));
  }
};

Expr WavefrontScheduleTransform(const Expr& e) {
  return WavefrontScheduler().Schedule(e);
}

}  // namespace wavefront_stream_schedule

Pass WavefrontStreamSchedule() {
  pass::PassContext pass_ctx = pass::PassContext::Current();
  tvm::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto transform = wavefront_stream_schedule::WavefrontScheduleTransform;
        return Downcast<Function>(tvm::relay::TransformF(transform, f));
      };
  return CreateRAFFunctionPass(pass_func, 1, "WavefrontStreamSchedule", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.WavefrontStreamSchedule").set_body_typed(WavefrontStreamSchedule);

}  // namespace pass
}  // namespace raf
