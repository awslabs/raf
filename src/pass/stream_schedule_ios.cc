/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/stream_schedule_ios.cc
 * \brief IOS (Inter-Operator Scheduler) stream scheduler.
 *  Reference: IOS: Inter Operator Scheduler for CNN Acceleration (MLSys 2021).
 */
#include <chrono>
#include <utility>
#include <thread>
#include <relay/transforms/pass_utils.h>
#include <tvm/runtime/device_api.h>
#include "raf/pass.h"
#include "raf/analysis.h"
#include "raf/op.h"
#include "raf/op_profiler.h"
#include "raf/op_utils.h"
#include "raf/profiler.h"
#include "./stream_schedule.h"
#include "../requests.h"
#include "../analysis/dependency_graph.h"

#ifdef RAF_USE_CUDA
#include "../common/cuda_utils.h"
#include "../op/dialect/cudnn/cudnn_utils.h"
#include "../op/dialect/cublas/cublas_utils.h"
#endif

#define LOG_PRINTF(format, args...)                \
  do {                                             \
    char buf[1024];                                \
    std::snprintf(buf, sizeof(buf), format, args); \
    LOG(INFO) << buf;                              \
  } while (0)

namespace raf {
namespace pass {
namespace ios_stream_schedule {

using namespace raf::analysis;
using namespace raf::op;
using namespace raf::memory_pool;
using requests::Requests;
using stream_schedule::StreamSchedulerBase;
using Node = DependencyGraph::Node;

/*!
 * \brief Get a integer with ones in the least significant bits and zeros in the other bits.
 * \param num_ones The number of ones.
 */
uint64_t GetLeastSignificantOnes(size_t num_ones) {
  uint64_t state = 0;
  for (int i = 0; i < num_ones; i++) {
    state |= 1ull << i;
  }
  return state;
}

/*!
 * \brief Count the number of one bits in an unsigned 64-bit integer.
 * \param v The given integer.
 * \return The number of bits whose value is one.
 */
inline int CountOneBits(uint64_t v) {
  int c = 0;
  while (v) {
    c++;
    v &= (v - 1);  // Remove the least significant bit with value 1.
  }
  return c;
}

#ifdef RAF_USE_CUDA
/*!
 * \brief The cost model of IOS scheduler. It profile the latency of IOS proposed stage on device.
 *
 * [Decision choice]
 * Because the purpose of this cost model it to get the latency of the proposed stage on raf
 * virtual machine, the ideal way of profiling is to convert the stage into a VM program and
 * measure the latency in VM. However, there is a major issue prevent us to do so. The VM caches
 * OpEnv for each program, and the cache would be released when we remove the VM for that program.
 * This makes it slow when we create and destroy OpEnv frequently.
 *
 * Thus, we implements the CostModel directly by launching the kernels in the cost model directly.
 * This gives us the flexibility to profile (we can control the times of warmup and repeat). It is
 * also the most efficient way to profile. This decision is a trade-off between the profiling
 * accuracy and the compilation time.
 */
class IOSCostModel {
  using OpEnvCache = MetaCache<std::shared_ptr<OpEnv>>;

 public:
  /*!
   * \brief The IOS cost model.
   * \param device The target device. Must be a cuda device.
   * \param num_stream The maximum number of streams to support.
   * \param warmup The number of warmups before real execution.
   * \param number The number of executions as a repeat.
   * \param repeat The number of repeat times.
   */
  IOSCostModel(Device device, int warmup, int number, int repeat) {
    CHECK_EQ(device.device_type(), DevType::kCUDA()) << "IOS cost model only supports CUDA.";
    this->device_ = device;
    this->warmup_ = warmup;
    this->number_ = number;
    this->repeat_ = repeat;
    this->profiler_ = op_profiler::OpProfiler::Get(device);
  }

  /*!
   * \brief Measure the latency of a stage (consists of multiple independent groups).
   * \param groups The independent groups.
   * \return The latency of the stage.
   */
  std::vector<float> StageLatency(const std::vector<std::vector<Expr>>& groups) {
    std::vector<Expr> flat_group;
    std::vector<int> stream_ids;
    for (size_t i = 0; i < groups.size(); i++) {
      auto group = groups[i];
      for (auto& expr : group) {
        flat_group.push_back(expr);
        stream_ids.push_back(i);  // The i-th group is assigned to the i-th stream.
      }
    }
    auto prof_res = profiler_->ProfileOpGroup(flat_group, stream_ids, warmup_, number_, repeat_);
    return prof_res.first;
  }

 private:
  /*The following data are configs. */

  /*! \brief Target device to profile on. */
  Device device_;
  /*! \brief Number of warmups. */
  int warmup_;
  /*! \brief The number of executions as a repeat.*/
  int number_;
  /*! \brief The number of repeat times. */
  int repeat_;
  /*! \brief The op profiler. */
  op_profiler::OpProfiler* profiler_;
};
#else
class IOSCostModel {
 public:
  IOSCostModel(Device device, int warmup, int number, int repeat) {
    LOG(FATAL) << "Please build with CUDA enabled to use IOS schedule.";
  }
  std::vector<float> StageLatency(const std::vector<std::vector<Expr>>& groups) {
    return {};
  }
};
#endif

class IOSScheduler : public StreamSchedulerBase {
  /*! \brief A group of nodes. */
  using Group = std::vector<Node*>;
  /*!
   * \brief A stage with multiple independent groups. The operators in a group will be executed
   * sequentially on a stream. The groups will be executed on different streams.
   */
  using Stage = std::vector<Group>;
  /*!
   * \brief The state in the dynamic programming, which represents the operators remaining to be
   * scheduled. We use a 64-bit unsigned integer to represent a state, which limits the maximum
   * number of operators we can schedule at the same time.
   */
  using State = uint64_t;
  /*!
   * \brief The decision for each state is an *ending* of the operators in the state. An ending is
   * a subset operators of the state, and there does not exist an edge u -> v such that
   * u \in ending but v \in state - ending.
   */
  using Decision = uint64_t;

 public:
  /*!
   * \brief The IOS scheduler.
   * \param device The target device to schedule for. We need this because IOS scheduler is a
   * profiling-based scheduler.
   * \param max_block_size The maximum number of operators in a block. If a natural block has more
   * operators, we will split it to satisfy this constraint.
   * \param max_stream_num The maximum number of streams to support. If there are more independent
   * groups, we will merge small groups to satisfy this constraint.
   * \param max_stage_ops The maximum number of operators in a stage.
   * \param search_group_combination Whether to search the group combination of independent group.
   * When this is turned on, we will try to merge the groups to find a better stage configuration,
   * even if they have already satisfied the stream constraint. This may slower the scheduling.
   * \param schedule_units The schedule units. A schedule unit is a sequence of operators. We will
   * schedule the model based on these units. This helps to reduce the search complexity.
   * \param warmup The number of warmups before real execution.
   * \param number The number of executions as a repeat.
   * \param repeat The number of repeat times.
   * \param verbose Whether print the verbose message during scheduling.
   */
  explicit IOSScheduler(Device device, int max_block_size = 20, int max_stream_num = 5,
                        int max_stage_ops = 10, bool search_group_combination = true,
                        Array<Array<Op>> schedule_units = {}, int warmup = 2, int number = 5,
                        int repeat = 5, bool verbose = false)
      : cost_model_(device, warmup, number, repeat), verbose_(this, verbose) {
    CHECK_GE(max_stream_num, 1) << "Stream number must be greater or equal to 1, but got "
                                << max_stream_num;
    CHECK_LE(max_block_size, 64) << "Only support maximum block size less or equal to 64, but got "
                                 << max_block_size;
    for (int i = 0; i < schedule_units.size(); i++) {
      CHECK_GE(schedule_units[i].size(), 2)
          << "Each schedule unit must have at least two operators, but got " << schedule_units[i];
    }

    config_.max_block_size = max_block_size;
    config_.max_stream_num = max_stream_num;
    config_.max_stage_ops = max_stage_ops;
    config_.search_group_combination = search_group_combination;
    config_.schedule_units = std::move(schedule_units);
  }

  /*!
   * \brief Schedule the expr by injecting stream-related operators.
   * \param e The expr to be scheduled. It must be in BBNF or GNF.
   * \return The scheduled expr. It is in ANF.
   */
  Expr Schedule(const Expr& e) {
    InitGraphInfo(e);

    BlockPartition();
    if (verbose_.on) {
      std::stringstream ss;
      for (auto& block : blocks_) {
        ss << block.nodes.size() << " ";
      }
      LOG_PRINTF("Found %zu schedule units in %zu blocks: %s", graph_.all_nodes.size(),
                 blocks_.size(), ss.str().c_str());
    }

    auto stages = ScheduleBlocks();

    for (int i = 0; i < stages.size(); i++) {
      Stage& stage = stages.at(i);
      for (int j = 0; j < stage.size(); j++) {
        Group& group = stage[j];
        AnnotateSetStream(0, j);
        for (Node* node : group) {
          for (Expr expr : graph_.node_unit[node]) {
            VisitExpr(expr);
          }
        }
      }
      if (i + 1 < stages.size()) {
        AnnotateStreamBarrier();
      }
    }
    return let_list_.Get(VisitExpr(e));
  }

  /*!
   * \brief Initialize the graph information. It takes the following steps:
   * 1. Create the dependency graph.
   * 2. Extract the scheduling units.
   * 3. Get the post-dfs-order of the simplified graph.
   */
  void InitGraphInfo(const Expr& e) {
    auto dg = CreateDependencyGraph(&graph_.arena, e, true, true);

    for (auto& it : dg.expr_node) {
      graph_.node_unit[it.second] = {it.first};
    }

    for (auto schedule_unit : config_.schedule_units) {
      while (true) {
        std::vector<Node*> matched_nodes;
        for (auto& pr : graph_.node_unit) {
          Node* node = pr.first;
          int i;
          for (i = 0; i < schedule_unit.size(); i++) {
            if (graph_.node_unit[node].size() != 1 ||
                (graph_.node_unit[node].front().as<CallNode>() == nullptr) ||
                (graph_.node_unit[node].front().as<CallNode>()->op.as<OpNode>() == nullptr)) {
              break;
            }
            const auto* op_node = graph_.node_unit[node].front().as<CallNode>()->op.as<OpNode>();
            if (op::IsDialectOp(GetRef<Op>(op_node))) {
              op_node = op::GetBaseOp(GetRef<Op>(op_node)).as<OpNode>();
            }
            if (op_node != schedule_unit[i].get()) {
              break;
            }

            if (i + 1 < schedule_unit.size() && dependency_graph::GetListSize(node->parents) != 1) {
              break;
            } else {
              node = node->parents.head->value;
            }
          }
          if (i == schedule_unit.size()) {
            node = pr.first;
            for (int j = 0; j < schedule_unit.size(); j++) {
              matched_nodes.push_back(node);
              node = node->parents.head->value;
            }
            break;
          }
        }
        if (matched_nodes.empty()) {
          break;
        }
        std::vector<Node*> children, parents;
        for (auto iit = matched_nodes.front()->children.head; iit; iit = iit->next) {
          children.push_back(iit->value);
        }
        for (auto iit = matched_nodes.back()->parents.head; iit; iit = iit->next) {
          parents.push_back(iit->value);
        }
        for (Node* child : children) {
          dependency_graph::RemoveGraphEdge(matched_nodes.front(), child);
        }
        for (Node* parent : parents) {
          dependency_graph::RemoveGraphEdge(parent, matched_nodes.back());
        }
        Node* unit_node = dependency_graph::CreateNewNode(&graph_.arena);
        for (Node* child : children) {
          dependency_graph::AddGraphEdge(&graph_.arena, unit_node, child);
        }
        for (Node* parent : parents) {
          dependency_graph::AddGraphEdge(&graph_.arena, parent, unit_node);
        }
        Array<Expr> matched_exprs;
        for (Node* matched_node : matched_nodes) {
          matched_exprs.push_back(graph_.node_unit[matched_node].front());
          graph_.node_unit.erase(matched_node);
        }
        graph_.node_unit[unit_node] = std::move(matched_exprs);
      }
    }

    std::vector<Node*> nodes;
    for (auto& pr : graph_.node_unit) {
      nodes.push_back(pr.first);
    }
    graph_.all_nodes = dependency_graph::GetTopologicalOrder(nodes);
  }

  /*!
   * \brief Partition the dataflow graph into sequential blocks. It takes the following steps:
   * 1. Build the dominator tree of the dataflow graph.
   * 2. Find the cut nodes in the dataflow graph through the dominator tree.
   *     Let P = {p | p is a path from the entry to sink node}.
   *     Then {cut nodes} = intersection of all paths in P.
   * 3. Let p1, p2, ..., pn be the cut nodes in topological order. We can get each block:
   *     Block i = {op | pi-1 is before op and op is before pi}.
   * 4. Split block if its size is larger than config_.max_block_size.
   */
  void BlockPartition() {
    // dominator tree
    std::vector<Node*> stack;
    std::unordered_map<const Node*, int> out_degree;
    for (auto node : graph_.all_nodes) {
      for (auto iit = node->children.head; iit; iit = iit->next) {
        out_degree[node]++;
      }
    }

    std::unordered_map<const Node*, const Node*> dom_parent;
    std::unordered_map<const Node*, int> depth;
    depth[nullptr] = 0;
    for (auto node : graph_.all_nodes) {
      if (out_degree[node] == 0) {
        stack.push_back(node);
      }
    }
    auto LCA = [&](const Node* lhs, const Node* rhs) {
      while (lhs != rhs) {
        if (depth[lhs] < depth[rhs])
          rhs = dom_parent[rhs];
        else
          lhs = dom_parent[lhs];
      }
      return lhs;
    };

    while (!stack.empty()) {
      Node* node = stack.back();
      stack.pop_back();
      if (node->children.head == nullptr) {
        dom_parent[node] = nullptr;
      } else {
        dom_parent[node] = node->children.head->value;
        for (auto iit = node->children.head->next; iit; iit = iit->next) {
          dom_parent[node] = LCA(dom_parent[node], iit->value);
        }
      }
      depth[node] = depth[dom_parent[node]] + 1;
      for (auto iit = node->parents.head; iit; iit = iit->next) {
        Node* parent = iit->value;
        if (--out_degree[parent] == 0) {
          stack.push_back(parent);
        }
      }
    }

    std::vector<const Node*> cut_points;
    for (const Node* node = graph_.all_nodes.back(); node; node = dom_parent[node]) {
      cut_points.push_back(node);
    }
    std::reverse(cut_points.begin(), cut_points.end());

    std::vector<std::vector<Node*>> original_blocks;
    for (int i = 0, j = 0; i < cut_points.size(); i++) {
      original_blocks.emplace_back();
      while (graph_.all_nodes[j] != cut_points[i]) {
        original_blocks.back().push_back(graph_.all_nodes[j]);
        j++;
      }
      original_blocks.back().push_back(graph_.all_nodes[j++]);
    }
    std::stringstream ss;
    for (auto& block : original_blocks) {
      ss << block.size() << " ";
    }

    std::vector<std::vector<Node*>> blocks;
    for (auto& block : original_blocks) {
      if (block.size() <= config_.max_block_size) {
        blocks.push_back(std::move(block));
      } else {
        size_t block_size = block.size();
        size_t num_split_blocks =
            (block_size + config_.max_block_size - 1) / config_.max_block_size;
        size_t i = blocks.size();
        for (size_t j = 0; j < num_split_blocks; j++) {
          blocks.emplace_back();
        }
        for (size_t k = 0; k < block.size(); k++) {
          size_t j = k / config_.max_block_size;
          blocks[i + j].push_back(block[k]);
        }
        if (verbose_.on) {
          std::stringstream ss;
          for (size_t j = 0; j < num_split_blocks; j++) {
            ss << (j == 0 ? "" : ", ") << blocks[i + j].size();
          }
          LOG_PRINTF("Split a block with %zu operators into small blocks with %s operators.",
                     block_size, ss.str().c_str());
        }
      }
    }

    blocks_.resize(blocks.size());
    for (int i = 0; i < blocks.size(); i++) {
      blocks_[i].nodes = std::move(blocks[i]);
      for (int j = 0; j < blocks_[i].nodes.size(); j++) {
        blocks_[i].node_index[blocks_[i].nodes[j]] = j;
      }
    }
  }

  /*!
   * \brief Schedule all blocks.
   * \return The stages of the whole model.
   */
  std::vector<Stage> ScheduleBlocks() {
    int num_blocks = static_cast<int>(blocks_.size());

    // Collect dynamic programming statistics.
    std::vector<int> block_order;
    for (int i = 0; i < num_blocks; i++) {
      block_order.push_back(i);
    }
    std::sort(block_order.begin(), block_order.end(), [&](int lhs, int rhs) {
      return blocks_[lhs].nodes.size() < blocks_[rhs].nodes.size();
    });
    std::vector<std::thread> threads;
    int num_workers = static_cast<int>(std::thread::hardware_concurrency());
    auto workload = [&](int block_id) { CollectDynamicProgrammingStatistics(block_id); };
    for (int i = 0; i < num_blocks; i++) {
      if (i - num_workers >= 0) {
        threads[i - num_workers].join();
      }
      threads.emplace_back(workload, block_order[i]);
    }
    for (int i = 0; i < num_blocks; i++) {
      if (threads[i].joinable()) {
        threads[i].join();
      }
    }

    // Print the progress information when verbose on
    verbose_.InitProgress();

    // Schedule each block
    std::vector<Stage> stages;
    for (int i = 0; i < num_blocks; i++) {
      ScheduleBlock(i);
      for (auto& stage : blocks_[i].stages) {
        stages.push_back(stage);
      }
    }

    // Merge stages with only one group
    std::vector<Stage> merged_stages;
    bool mergeable = false;
    for (auto& stage : stages) {
      if (mergeable && stage.size() == 1) {
        auto& group = merged_stages.back()[0];
        for (auto node : stage[0]) {
          group.push_back(node);
        }
      } else {
        merged_stages.push_back(stage);
        mergeable = (stage.size() == 1);
      }
    }

    return merged_stages;
  }

  /*!
   * \brief Collect the dynamic programming statistics. This would calculate the states and the
   * decisions for each state. The decisions for each state are cached and can be directly used
   * in our later dynamic programming. The state information can be used to print the progress
   * messages.
   * \param block_id The index of block to collect the dynamic programming statistics.
   */
  void CollectDynamicProgrammingStatistics(int block_id) {
    BlockInfo& block = blocks_[block_id];
    std::function<void(State)> DP = [&](State state) {
      if (state == 0) {
        return;
      }
      if (block.state_decision_candidates.count(state)) {
        return;
      }
      for (auto candidate : GetStateDecisionCandidates(block_id, state)) {
        DP(state - candidate);
      }
    };
    State full_state = GetLeastSignificantOnes(block.nodes.size());
    DP(full_state);
  }

  /*!
   * \brief Schedule a block. This function would generate the scheduled stages for given block.
   * \param block_id The index of the block to be scheduled.
   *
   * \return This function returns nothing. It stores the scheduled stages for this block in
   * blocks_[block_id].stages.
   */
  void ScheduleBlock(int block_id) {
    auto& state_latency = blocks_[block_id].state_latency;
    auto& state_decision = blocks_[block_id].state_decision;

    std::function<float(State)> DP = [&](State state) {
      if (state == 0) {
        // empty state means we have scheduled all operators in the block.
        return 0.0f;
      }
      if (state_latency.count(state)) {
        // Return the memorized state latency.
        return state_latency[state];
      }
      float best_latency = 1e18;
      Decision best_decision = state;
      for (auto decision : GetStateDecisionCandidates(block_id, state)) {
        DCHECK_EQ((decision & state), decision);
        float decision_latency = GetDecisionStageLatency(block_id, decision);
        float total_latency = DP(state - decision) + decision_latency;
        if (best_latency > total_latency) {
          best_latency = total_latency;
          best_decision = decision;
        }
      }
      state_latency[state] = best_latency;
      state_decision[state] = best_decision;
      verbose_.UpdateProgress();
      return best_latency;
    };

    State state = GetLeastSignificantOnes(blocks_[block_id].nodes.size());

    DP(state);

    // construct the optimal schedule for this block
    auto& decision_stage = blocks_[block_id].decision_stage;
    auto& stages = blocks_[block_id].stages;
    while (state) {
      Decision decision = state_decision[state];
      stages.push_back(decision_stage[decision]);
      state -= decision;
    }
    std::reverse(stages.begin(), stages.end());
  }

  /*!
   * Get the decision candidates for given state in a block.
   * \param block_id The block id.
   * \param state A state in the block.
   * \return The decision candidates.
   */
  const std::vector<Decision>& GetStateDecisionCandidates(int block_id, State state) {
    if (blocks_[block_id].state_decision_candidates.count(state)) {
      // Return the cached result directly.
      return blocks_[block_id].state_decision_candidates[state];
    }

    std::unordered_set<Decision> found_decisions;
    std::vector<Decision> stack;
    auto& block_nodes = blocks_[block_id].nodes;
    auto& node_index = blocks_[block_id].node_index;

    // Expand the decision to get more valid decisions by adding a new node to current decision
    auto DecisionExpand = [&](Decision decision) {
      if (CountOneBits(decision) == config_.max_stage_ops) {
        // Return an empty set when the number of operators in current decision reached the limit.
        return std::vector<Decision>();
      }

      std::vector<Decision> new_decisions;
      for (size_t u = 0; u < block_nodes.size(); u++) {
        // Try to add block_nodes[u] to current decision. We need to make sure
        //  1. The node u is in state, but not in current decision
        //  2. There does not exist node v, such that
        //    2a. u -> v in the dataflow graph, and
        //    2b. v is in state but not in current decision.

        if ((((state ^ decision) >> u) & 1) == 0) {
          // Node u is not in {node | node in state} - {node | node in decision}, violate cond 1.
          continue;
        }

        Node* node = block_nodes[u];
        bool valid = true;
        for (auto iit = node->parents.head; iit; iit = iit->next) {
          Node* parent = iit->value;
          if (node_index.count(parent) == 0) {
            // Parent is in following blocks. Do not consider it.
            continue;
          }
          // There is an edge from u -> v in dataflow graph.
          int v = node_index[parent];
          if (((state >> v) & 1) != 0 && ((decision >> v) & 1) == 0) {
            // v is in state but not in decision, violate cond 2.
            valid = false;
            break;
          }
        }
        if (valid) {
          new_decisions.push_back(decision | (State(1) << u));
        }
      }
      return new_decisions;
    };
    stack.push_back(0);
    while (!stack.empty()) {
      Decision decision = stack.back();
      stack.pop_back();
      std::vector<Decision> new_decisions = DecisionExpand(decision);
      for (auto d : new_decisions) {
        if (found_decisions.count(d) == 0) {
          stack.push_back(d);
          found_decisions.insert(d);
        }
      }
    }

    // empty set is not a valid candidate
    found_decisions.erase(0);

    blocks_[block_id].state_decision_candidates[state] =
        std::vector<Decision>(found_decisions.begin(), found_decisions.end());
    return blocks_[block_id].state_decision_candidates[state];
  }

  /*!
   * \brief Get the stage for a decision. A decision is a set of operators. A stage is a set of
   * independent groups. Each group is a sequence of operators. All potential stages for a decision
   * have exactly the same set of operators as the decision.
   * \param block_id The block index.
   * \param decision The decision.
   * \return The stage of given decision.
   */
  float GetDecisionStageLatency(int block_id, Decision decision) {
    /* This function takes the following steps to find the stage for given decision.
     * 1. Use disjoint union set to find the independent groups of operators.
     * 2. Merge small groups if the number of independent groups exceeds config_.max_stream_num.
     * 3. If the config_.search_group_combination is off, return the found independent groups,
     *    otherwise, goto step 4.
     * 4. Merge the two smallest groups until all there is only one group left. After each merging,
     *    measure the latency of the independent groups and use the best one as the stage for the
     *    given decision.
     */
    auto& nodes = blocks_[block_id].nodes;
    auto& decision_stage = blocks_[block_id].decision_stage;
    auto& decision_latency = blocks_[block_id].decision_latency;
    auto& node_index = blocks_[block_id].node_index;

    // Use the cached result if possible
    if (decision_stage.count(decision)) {
      return decision_latency[decision];
    }

    // Extract the nodes in decision
    std::vector<Node*> stage_nodes;
    for (int i = 0; i < nodes.size(); i++) {
      if ((decision >> i) & 1) {
        stage_nodes.push_back(nodes[i]);
      }
    }

    // Disjoint set functions
    std::vector<int> disjoint_set_father(nodes.size());
    std::function<int(int)> Find = [&](int node) {
      if (disjoint_set_father[node] == node) return node;
      return disjoint_set_father[node] = Find(disjoint_set_father[node]);
    };
    std::function<void(int, int)> Union = [&](int a, int b) {
      a = Find(a);
      b = Find(b);
      disjoint_set_father[a] = b;
    };

    // Use disjoint set to union all connected stage nodes
    for (int i = 0; i < nodes.size(); i++) disjoint_set_father[i] = i;
    for (Node* node : stage_nodes) {
      for (auto iit = node->parents.head; iit; iit = iit->next) {
        Node* parent = iit->value;
        if (node_index.count(parent) && ((decision >> node_index[parent]) & 1)) {
          Union(node_index[node], node_index[parent]);
        }
      }
    }

    // Get the default stage. Default stage contains all disjoint node groups.
    Stage stage;
    for (Node* node : stage_nodes) {
      int index = node_index[node];
      if (index == Find(index)) {
        stage.emplace_back();
        for (Node* nd : stage_nodes) {
          if (Find(node_index[nd]) == index) {
            stage.back().push_back(nd);
          }
        }
      }
    }

    // Merge groups to satisfy the stream number constraint
    auto& node_unit = graph_.node_unit;
    std::vector<std::pair<Group, int>> group_ops;
    for (Group& group : stage) {
      int ops = 0;
      for (Node* node : group) {
        ops += static_cast<int>(node_unit[node].size());
      }
      group_ops.emplace_back(group, ops);
    }
    auto compare = [&](const std::pair<Group, int64_t>& lhs, const std::pair<Group, int64_t>& rhs) {
      return lhs.second > rhs.second;
    };
    auto MergeOnce = [&]() {
      std::sort(group_ops.begin(), group_ops.end(), compare);
      auto pr = group_ops.back();
      group_ops.pop_back();
      group_ops.back().second += pr.second;
      for (Node* node : pr.first) {
        group_ops.back().first.push_back(node);
      }
    };
    while (group_ops.size() > config_.max_stream_num) {
      MergeOnce();
    }

    Stage best_stage;
    for (auto& pr : group_ops) {
      best_stage.push_back(pr.first);
    }
    float best_latency = MeasureStageLatency(best_stage);
    if (config_.search_group_combination && group_ops.size() != 1) {
      // tries to merge the least two lightweight groups each time
      do {
        MergeOnce();
        Stage candidate_stage;
        for (auto& pr : group_ops) {
          candidate_stage.push_back(pr.first);
        }
        float candidate_latency = MeasureStageLatency(candidate_stage);
        if (best_latency > candidate_latency) {
          best_latency = candidate_latency;
          best_stage = candidate_stage;
        }
      } while (group_ops.size() > 1);
    }
    decision_stage[decision] = best_stage;
    decision_latency[decision] = best_latency;
    return decision_latency[decision];
  }

  /*!
   * Measure the latency of a stage through the IOS cost model.
   * \param stage The stage to be profiled.
   * \return The latency of the stage.
   */
  float MeasureStageLatency(const Stage& stage) {
    std::vector<std::vector<Expr>> expr_groups;
    for (const auto& group : stage) {
      expr_groups.emplace_back();
      for (Node* node : group) {
        for (Expr expr : graph_.node_unit[node]) {
          expr_groups.back().push_back(expr);
        }
      }
    }
    std::vector<float> results = cost_model_.StageLatency(expr_groups);
    std::sort(results.begin(), results.end());
    float sum = 0.0;
    // We only use the smallest 30% data
    int cnt = std::max(static_cast<int>(results.size() * 0.3), 1);
    for (int i = 0; i < cnt; i++) {
      sum += results[i];
    }
    return sum / cnt;
  }

  struct Config {
    /*! \brief The maximum block size. The natural block would be split if it exceeds this value. */
    int max_block_size;
    /*! \brief The maximum number of streams we can use. */
    int max_stream_num;
    /*! \brief The maximum number of operators per stage. */
    int max_stage_ops;
    /*! \brief Whether to search the group combination in a stage. */
    bool search_group_combination;
    /*!
     * \brief The schedule units. IOS can take a sequence of operators as a schedule unit (e.g.,
     * conv2d + bn + relu). IOS considers all such chain as a schedule unit and will not split them
     * into different streams. This helps to reduce the schedule space. All other operators that
     * does not match these pattern would be schedule unit individually.
     */
    Array<Array<Op>> schedule_units;
  };
  /*! \brief The config of IOS scheduler. */
  Config config_;

  /*! \brief The cost model that IOS uses to predict the stage performance. */
  IOSCostModel cost_model_;

  struct GraphInfo {
    /*! \brief The arena used to allocate memory for DependencyGraph. */
    Arena arena;
    /*! \brief All nodes in the dependency graph, in post dfs order. */
    std::vector<Node*> all_nodes;
    /*! \brief The mapping from node to schedule unit. */
    std::unordered_map<Node*, Array<Expr>> node_unit;
  };
  /*! \brief The dependency graph information of the entire function. */
  GraphInfo graph_;

  struct BlockInfo {
    /*! \brief All nodes in this block. */
    std::vector<Node*> nodes;
    /*! \brief The mapping from node to its index. */
    std::unordered_map<const Node*, int> node_index;
    /*! \brief The mapping from state to all of its decision candidates. */
    std::unordered_map<State, std::vector<Decision>> state_decision_candidates;
    /*! \brief The mapping from state to its optimal decision. */
    std::unordered_map<State, Decision> state_decision;
    /*! \brief The mapping from state to the latency of its optimal decision. */
    std::unordered_map<State, float> state_latency;
    /*! \brief The mapping from a decision to its optimal stage. */
    std::unordered_map<Decision, Stage> decision_stage;
    /*! \brief The mapping from a decision to the latency of its optimal stage. */
    std::unordered_map<Decision, float> decision_latency;
    /*! \brief The stages of this block after scheduling. */
    std::vector<Stage> stages;
  };
  /*! \brief The data of each block. */
  std::vector<BlockInfo> blocks_;

  struct Verbose {
    IOSScheduler* scheduler;
    /*! \brief Whether to print verbose message during scheduling. */
    bool on;
    /*! \brief The total number of states. */
    uint64_t total_states;
    /*! \brief The time stamp when start printing the progress message. */
    uint64_t start_time_stamp;
    /*! \brief The time stamp of last progress verbose message. */
    uint64_t prev_time_stamp;
    /*! \brief The time interval to print the progress message. Default: 2000 ms. */
    static const int msg_interval{1 * 1000};

    explicit Verbose(IOSScheduler* scheduler, bool on) : scheduler(scheduler), on(on) {
    }

    void InitProgress() {
      if (!on) {
        return;
      }
      std::stringstream ss;
      total_states = 0;
      for (auto& block : scheduler->blocks_) {
        total_states += block.state_decision_candidates.size();
        ss << block.state_decision_candidates.size() << " ";
      }
      start_time_stamp = prev_time_stamp = raf::profiler::ProfileStat::NowInMicrosec() / 1000;
      LOG_PRINTF("Total states: %zu", total_states);
    }
    inline void UpdateProgress() {
      if (!on) {
        return;
      }
      uint64_t finished_states = 0;
      for (auto& block : scheduler->blocks_) {
        finished_states += block.state_latency.size();
      }
      uint64_t time_stamp = raf::profiler::ProfileStat::NowInMicrosec() / 1000;
      if (finished_states == total_states || time_stamp - prev_time_stamp > msg_interval) {
        uint64_t percent = finished_states * 100 / total_states;
        double states_per_min =
            double(finished_states) * 1000 * 60 / double(time_stamp - start_time_stamp);
        double eta = double(total_states - finished_states) / states_per_min;
        const char* eta_unit = "minutes";
        if (eta < 1.0) {
          eta *= 60;
          eta_unit = "seconds";
        }
        LOG_PRINTF("[%2lu%%] States: %lu / %lu  Speed: %.1f states / min  ETA: %.1f %s", percent,
                   finished_states, total_states, states_per_min, eta, eta_unit);
        prev_time_stamp = time_stamp;
        if (percent == 100) {
          LOG_PRINTF("Finish in %.1f minutes.", (time_stamp - start_time_stamp) / 60.0 / 1000.0);
        }
      }
    }
  };
  /*! \brief The information used to print verbose messages. */
  Verbose verbose_;
};

/*!
 * Use IOS scheduler to schedule the given expression. Please refer to IOSScheduler for more
 * information about the meaning of each config parameter.
 * \return The scheduled expression.
 */
Expr IOSStreamSchedule(const Expr& e, Device device, int block_max_size = 20,
                       int max_stream_num = 5, int max_stage_ops = 10,
                       bool search_group_combination = true, Array<Array<Op>> schedule_units = {},
                       int warmup = 1, int number = 5, int repeat = 5, bool verbose = false) {
  IOSScheduler scheduler(device, block_max_size, max_stream_num, max_stage_ops,
                         search_group_combination, std::move(schedule_units), warmup, number,
                         repeat, verbose);
  return scheduler.Schedule(e);
}

}  // namespace ios_stream_schedule

Pass IOSStreamSchedule() {
  pass::PassContext ctx = pass::PassContext::Current();
  std::unordered_map<std::string, tvm::runtime::TVMArgValue> config;
  auto get_int_config = [&](const std::string& name, int default_value) {
    std::string key = "raf.stream_schedule.ios." + name;
    int64_t val = ctx->GetConfig<tvm::Integer>(key, tvm::Integer(default_value)).value()->value;
    return int(val);
  };
  auto get_bool_config = [&](const std::string& name, bool default_value) {
    std::string key = "raf.stream_schedule.ios." + name;
    return ctx->GetConfig<tvm::Bool>(key, tvm::Bool(default_value)).value()->value;
  };

  int block_max_size = get_int_config("block_max_size", 20);
  int max_stream_num = get_int_config("max_stream_num", 5);
  int max_stage_ops = get_int_config("max_stage_ops", 10);
  bool search_group_combination = get_bool_config("search_group_combination", false);
  int warmup = get_int_config("warmup", 2);
  int number = get_int_config("number", 1);
  int repeat = get_int_config("repeat", 8);
  bool verbose = get_bool_config("verbose", true);
  Array<Array<Op>> schedule_units =
      ctx->GetConfig<Array<Array<Op>>>("raf.stream_schedule.ios.schedule_units", Array<Array<Op>>())
          .value();
  // Add default schedule units at the end of user-provided units: Conv-Bn-Relu, Conv-Bn, Conv-Relu
  // Please add more here when needed.
  schedule_units.push_back(
      {Op::Get("raf.op.conv2d"), Op::Get("raf.op.batch_norm_infer"), Op::Get("raf.op.relu")});
  schedule_units.push_back({Op::Get("raf.op.conv2d"), Op::Get("raf.op.batch_norm_infer")});
  schedule_units.push_back({Op::Get("raf.op.conv2d"), Op::Get("raf.op.relu")});

  tvm::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto transform = [=](Expr e) {
          return ios_stream_schedule::IOSStreamSchedule(
              e, Device(DevType::kCUDA(), 0), block_max_size, max_stream_num, max_stage_ops,
              search_group_combination, schedule_units, warmup, number, repeat, verbose);
        };
        return Downcast<Function>(tvm::relay::TransformF(transform, f));
      };
  return CreateRAFFunctionPass(pass_func, 1, "IOSStreamSchedule", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.IOSStreamSchedule").set_body_typed(IOSStreamSchedule);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.stream_schedule.ios.block_max_size", tvm::Integer);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.stream_schedule.ios.max_stream_num", tvm::Integer);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.stream_schedule.ios.max_stage_ops", tvm::Integer);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.stream_schedule.ios.search_group_combination", tvm::Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.stream_schedule.ios.warmup", tvm::Integer);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.stream_schedule.ios.number", tvm::Integer);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.stream_schedule.ios.repeat", tvm::Integer);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.stream_schedule.ios.verbose", tvm::Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.stream_schedule.ios.schedule_units", Array<Array<Op>>);
}  // namespace pass
}  // namespace raf
