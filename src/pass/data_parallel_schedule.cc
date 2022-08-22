/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file data_parallel_schedule.cc
 * \brief Schedules ops during data parallel training.
 */
#include <queue>
#include <unordered_set>
#include <relay/transforms/pass_utils.h>
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/dist_config.h"
#include "raf/pass.h"
#include "raf/analysis.h"
#include "./common.h"
#include "let_list.h"
#include "stream_schedule.h"
#include "raf/stream_pool.h"

namespace raf {
namespace pass {
namespace data_parallel_schedule {

using namespace raf::ir;
using raf::distributed::DistConfig;
using namespace raf::analysis;
using op::IsCollectiveOp;
using stream_pool::StreamTagEnum;
using tvm::OpAttrMap;
using Node = DependencyGraph::Node;
using LinkedList = tvm::relay::LinkedList<Node*>;
using NodeExprMap = std::unordered_map<const Node*, Expr>;
using stream_schedule::StreamSchedulerBase;

class FIFOScheduler : public StreamSchedulerBase {
 public:
  /*! This scheduler schedules the execution order of ops so communication ops can better overlap
   * with computation ops. It works on BBNF/GNF and outputs the scheduled expression in ANF.
   *
   * Why do we need this pass:
   * Since we do not have a dynamic execution engine that only launches an op when all its inputs
   * becomes ready, explicit synchronization ops are introduced to preserve correct dependencies
   * between ops launched to different CUDA streams. However, a bad op launch order can prevent the
   * overlap between ops on different streams. Take the following data flow graph as an example,
   * where all allreduce ops are executed on a separate communication stream (stream b) and all
   * computation ops (atan and mul) are executed on a single compute stream (stream a):
   *
   *           -> allreduce --> atan3 -
   *         /                          \
   *   atan0 --->   atan1   --> atan2 ---> mul
   *
   * When transforming it into ANF, the following two op orders are both possible:
   *   Order 1: atan0 -> allreduce -> atan3 -> atan1 -> atan2 -> mul
   *     Execution timeline:
   *       Stream a:   atan0              -> atan3 -> atan1 -> atan2 -> mul
   *       Stream b:         -> allreduce
   *
   *   Order 2: atan0 -> allreduce -> atan1 -> atan2 -> atan3 -> mul
   *     Execution timeline:
   *       Stream a:   atan0 -> atan1 -> atan2 -> atan3 -> mul
   *       Stream b:         -> allreduce
   *
   * In order 1, since atan3 depends on the output of allreduce, it must wait for allreduce to
   * finish. However, the scheduled op order demands that atan1 and atan2 are launched after atan3,
   * thus they are also blocked. Order 2 schedules atan1 and atan2 before atan3 to avoid this
   * problem.
   *
   * Current ToANormalForm travels the data flow graph in post DFS order, in which chains of ops
   * (e.g. allreduce and the consumer of its output) will be put next to each other, resulting in a
   * immediate synchonization after the communication op. This can lead to unnecessary blocking
   * if no enough computation ops have been launched on the computation stream to overlap with
   * the communication op. To alleviate this problem, this pass implements a simple FIFO scheduling
   * strategy:
   *   1. Maintain a counter for each op that keeps track of the number of its unscheduled
   *      predecessors. Add all ops with no predecessors into a ready queue.
   *   2. While the queue is not empty, pop an op out of the queue and mark it as scheduled.
   *      Decrease the counter for all its successors. If any successor has no unscheduled
   *      predecessors (i.e. becomes ready), push it into the exection queue.
   *   3. The order of popping ops out of the ready queue is out final op launch order.
   *
   * This effectively performs a topological sort on the ops and issues ops in a "BFS" order.
   * In addition, this pass also tries to delay any node that depends on a communication node
   * as late as possible by using a separate queue for them. Consider the following graph:
   *
   *           -> allreduce --> atan3 -----------------------
   *         /                                               \
   *   atan0 --->   atan1   --> atan2 --> atan4 --> atan5 --> mul
   *
   * Even if FIFO scheduling is used, atan3 will be scheduled at a relatively early position,
   * which can still block computation if the allreduce is long, e.g.:
   *
   * ANF order: atan0 -> atan1 -> allreduce -> atan2 -> atan3 -> atan4 -> atan5 -> mul
   * Execution timeline:
   *   Stream a:   atan0 -> atan1 -> atan2             -> atan3 -> atan4 -> atan5 -> mul
   *   Stream b:         -> l o n g _ a l l r e d u c e
   *
   * By using a separate queue for ops that directly depends on a communication,
   * those ops are delayed until no other op is available, leaving more room for overlap.
   */
  Expr Schedule(Expr e) {
    // create the data flow graph
    Arena arena;
    DependencyGraph dfg = CreateDependencyGraph(&arena, e, /*prune_atomic_nodes=*/true);
    // map each node in the dependency graph to the expression it represents
    NodeExprMap node_expr;
    // ready queue for all ops that directly depends on a communication op
    std::queue<Node*> comm_successor_ready_queue;
    // ready queue for all other ops
    std::queue<Node*> ready_queue;
    // counter that keeps track of the number of each op's current unscheduled predecessors
    // the dependency graph in tvm is a data flow graph with edge direction reversed, so we
    // use out-degree here instead of in-degree.
    std::unordered_map<Node*, int> out_degree;
    // keeps track of whether an op directly depends on a communication op
    std::unordered_set<Node*> comm_successor_nodes;

    for (auto& it : dfg.expr_node) {
      node_expr[it.second] = it.first;
    }
    std::vector<Node*>& nodes = dfg.post_dfs_order;

    // calculate out-degree for each node and populate comm_successor_nodes map
    for (auto node_it = nodes.rbegin(); node_it != nodes.rend(); node_it++) {
      out_degree[(*node_it)] = 0;
      if (auto call_node = node_expr[*node_it].as<CallNode>()) {
        if (IsCollectiveOp(call_node->op)) {
          // record direct successor nodes of communication op
          for (auto parent = (*node_it)->parents.head; parent; parent = parent->next) {
            comm_successor_nodes.insert(parent->value);
          }
        }
      }

      for (auto child = (*node_it)->children.head; child; child = child->next) {
        out_degree[(*node_it)]++;
      }
    }
    // push nodes with zero predecessors into the queue
    for (auto& node : nodes) {
      if (out_degree[node] == 0) {
        ready_queue.push(node);
      }
    }

    Expr ret;
    // in each step, we pop an op out of the queue, add it to the ANF and
    // push all its ready successors into the corresponding ready queue
    auto process_queue_element = [&](std::queue<Node*>& q) {
      while (!q.empty()) {
        Node* node = q.front();
        ret = VisitExpr(node_expr.at(node));
        for (auto parent = node->parents.head; parent; parent = parent->next) {
          out_degree[parent->value]--;
          if (out_degree[parent->value] == 0) {
            if (comm_successor_nodes.count(parent->value)) {
              comm_successor_ready_queue.push(parent->value);
            } else {
              ready_queue.push(parent->value);
            }
          }
        }
        q.pop();
      }
    };

    while (!ready_queue.empty() || !comm_successor_ready_queue.empty()) {
      process_queue_element(ready_queue);
      process_queue_element(comm_successor_ready_queue);
    }

    return let_list_.Get(ret);
  }
};

Expr FIFOScheduleTransform(const Expr& e) {
  return FIFOScheduler().Schedule(e);
}

}  // namespace data_parallel_schedule

Pass DataParallelSchedule() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(
        tvm::relay::TransformF(data_parallel_schedule::FIFOScheduleTransform, f));
  };
  return CreateRAFFunctionPass(pass_func, 0, "DataParallelSchedule", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.DataParallelSchedule").set_body_typed(DataParallelSchedule);

}  // namespace pass
}  // namespace raf
