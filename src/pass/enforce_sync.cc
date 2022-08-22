/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file enforce_sync.cc
 * \brief Enforce synchronization between ops in multiple streams.
 */
#include "raf/device.h"
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/dist_config.h"
#include "raf/pass.h"
#include "raf/analysis.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "./common.h"
#include "raf/stream_pool.h"

namespace raf {
namespace pass {
namespace enforce_sync {
using namespace raf::ir;
using namespace raf::op;
using namespace raf::value;
using namespace raf::distributed::communicator;
using namespace raf::analysis;
using stream_pool::StreamTagEnum;

using OpSet = std::unordered_set<Op, ObjectPtrHash, ObjectPtrEqual>;
using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;
template <typename T>
using ExprMap = std::unordered_map<Expr, T, ObjectPtrHash, ObjectPtrEqual>;

struct pair_hash {
  std::size_t operator()(const std::pair<int, int>& v) const {
    return std::hash<std::string>{}(std::to_string(v.first) + "," + std::to_string(v.second));
  }
};

struct pair_equal {
  bool operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const {
    return lhs.first == rhs.first && lhs.second == rhs.second;
  }
};

using DepSet = std::unordered_set<std::pair<int, int>, pair_hash, pair_equal>;

static int64_t compute_stream_idx = StreamTagEnum::CudaCompute();
static int64_t communication_stream_idx = StreamTagEnum::CudaCommunicate();
static int64_t fuse_tensor_stream_idx = StreamTagEnum::MemCudaToCuda1();
static int64_t defuse_tensor_stream_idx = StreamTagEnum::MemCudaToCuda2();
static int64_t unknown_stream_idx = StreamTagEnum::Unknown();

static std::unordered_map<int64_t, std::string> stream_name_hint = {
    {compute_stream_idx, "comp"},
    {communication_stream_idx, "comm"},
    {fuse_tensor_stream_idx, "fuse"},
    {defuse_tensor_stream_idx, "defuse"},
};

int IdentifyStream(const Expr& op) {
  int stream_idx = compute_stream_idx;
  if (op->IsInstance<CallNode>() && IsCollectiveOp(op.as<CallNode>()->op)) {
    stream_idx = communication_stream_idx;
    // if need fuse_tensor before collective op, previous stream should be memory copy stream
  } else if (op->IsInstance<CallNode>() && IsFuseTensorOp(op.as<CallNode>()->op)) {
    // if already have memcpy ops in the graph.
    stream_idx = fuse_tensor_stream_idx;
  } else if (op->IsInstance<CallNode>() && IsDefuseTensorOp(op.as<CallNode>()->op)) {
    stream_idx = defuse_tensor_stream_idx;
  }
  return stream_idx;
}

class SyncAnalyzer : ExprVisitor {
 public:
  /*! \brief Analyse the needed dependency edges between ops. It stores the analysis result in
   * add_event_after_op, wait_event_before_op and set_stream_before_op.
   *
   * \returns true if we need to add any dependency edge (i.e. if collective communication ops
   * are found in expr).
   */
  bool Analyse(const Expr& expr) {
    VisitExpr(expr);

    DepSet dep_set;
    GetDepSet_(dep_set);
    CreateEventsUsingDepSet_(dep_set);

    return !dep_set.empty();
  }

  void VisitExpr_(const VarNode* var) {
    var_idx_map_[GetRef<Var>(var)] = current_idx_;
  }

  void VisitExpr_(const CallNode* call) {
    auto call_expr = GetRef<Expr>(call);
    UpdateStreamInfo_(call_expr);
    UpdateDependencyInfo_(call_expr, call->args);
  }

  void VisitExpr_(const TupleNode* tuple) {
    auto tuple_expr = GetRef<Expr>(tuple);
    UpdateStreamInfo_(tuple_expr);
    UpdateDependencyInfo_(tuple_expr, tuple->fields);
  }

  void VisitExpr_(const TupleGetItemNode* tuple_get_item) {
    auto tuple_get_item_expr = GetRef<Expr>(tuple_get_item);
    UpdateStreamInfo_(tuple_get_item_expr);
    UpdateDependencyInfo_(tuple_get_item_expr, {tuple_get_item->tuple});
  }

  void VisitExpr_(const LetNode* op) {
    auto pre_visit = [this](const LetNode* op) {
      Var var = op->var;
      Expr value = op->value;
      max_idx_ = std::max(current_idx_, max_idx_);
      expr_idx_map[value] = current_idx_;
      var_idx_map_[var] = current_idx_;
      this->VisitExpr(var);
      this->VisitExpr(value);
      current_idx_++;
    };
    auto post_visit = [this](const LetNode* op) {
      Expr body = op->body;
      max_idx_ = std::max(current_idx_, max_idx_);
      this->VisitExpr(body);
      current_idx_--;
    };
    ExpandANormalForm(op, pre_visit, post_visit);
  }

  int GetUniqueEventId() {
    return next_unique_event_id_++;
  }

  // maps expr to id
  ExprMap<int> expr_idx_map;

  std::unordered_map<int, int> idx_stream_map;

  // maps expr id to added event_id
  std::unordered_map<int, std::vector<int>> add_event_after_op;
  // maps expr id to event_id to wait for
  std::unordered_map<int, std::vector<int>> wait_event_before_op;
  // maps expr id to whether it needs an set stream op before it
  std::unordered_map<int, bool> set_stream_before_op;

 private:
  // Determines if a set_stream op should be added before the op.
  // a set_stream op is needed if the executing stream of the previous op
  // (previous_op_stream_idx_) is different from the executing stream
  // of the current op. It also updates previous_op_stream_idx_.
  void UpdateStreamInfo_(const Expr& op) {
    idx_stream_map[current_idx_] = IdentifyStream(op);
    int expected_stream_idx = idx_stream_map[current_idx_];
    if (previous_op_stream_idx_ == -1 || previous_op_stream_idx_ != expected_stream_idx) {
      // if the current op is the first op or last op's stream is not the stream for current op
      set_stream_before_op[current_idx_] = true;
      previous_op_stream_idx_ = expected_stream_idx;
    }
  }

  // This function is called once for every op during ANF expansion. It updates the
  // last input producer and first output consumer in each stream for each op.
  // The last input producer map is updated as we iterate through the input
  // arguments. The last input producer map will contain the correct value after
  // we iterate through every op, since we must have called the function on
  // the true "first consumer" once.
  void UpdateDependencyInfo_(const Expr& op, const Array<Expr>& input_args) {
    for (auto arg : input_args) {
      if (var_idx_map_.count(arg)) {
        // update producer map and consumer map
        int arg_idx = var_idx_map_[arg];
        int expr_idx = current_idx_;
        UpdateLastProducerMap_(expr_idx, arg_idx);
        UpdateFirstConsumerMap_(arg_idx, expr_idx);
      }
    }
  }

  void UpdateLastProducerMap_(int idx, int producer) {
    int producer_stream = idx_stream_map[producer];
    if (producer_stream == idx_stream_map[idx]) {
      // neglect producers in the same stream
      return;
    }
    if (!last_producer_idx_map_.count(idx)) {
      last_producer_idx_map_.emplace(idx, std::unordered_map<int, int>{});
    }
    if (!last_producer_idx_map_[idx].count(producer_stream)) {
      last_producer_idx_map_[idx].emplace(producer_stream, producer);
    }
    if (producer > last_producer_idx_map_[idx][producer_stream]) {
      last_producer_idx_map_[idx][producer_stream] = producer;
    }
  }

  void UpdateFirstConsumerMap_(int idx, int consumer) {
    int consumer_stream = idx_stream_map[consumer];
    if (consumer_stream == idx_stream_map[idx]) {
      // neglect consumers in the same stream
      return;
    }
    if (!first_consumer_idx_map_.count(idx)) {
      first_consumer_idx_map_.emplace(idx, std::unordered_map<int, int>{});
    }
    if (!first_consumer_idx_map_[idx].count(consumer_stream)) {
      first_consumer_idx_map_[idx].emplace(consumer_stream, consumer);
    }
    if (consumer < first_consumer_idx_map_[idx][consumer_stream]) {
      first_consumer_idx_map_[idx][consumer_stream] = consumer;
    }
  }

  void GetDepSet_(DepSet& dep_set) {
    dep_set.clear();
    for (int call = 0; call < max_idx_ + 1; ++call) {
      for (auto stream_and_producer : last_producer_idx_map_[call]) {
        CHECK(stream_and_producer.first != unknown_stream_idx) << "Still have unknown stream.";
        dep_set.insert(std::make_pair(stream_and_producer.second, call));
      }
      for (auto stream_and_consumer : first_consumer_idx_map_[call]) {
        CHECK(stream_and_consumer.first != unknown_stream_idx) << "Still have unknown stream.";
        dep_set.insert(std::make_pair(call, stream_and_consumer.second));
      }
    }
    PruneReduntantDependency_(dep_set);
  }

  void PruneReduntantDependency_(DepSet& dep_set) {
    if (dep_set.empty()) {
      return;
    }

    // In multi-stream dependency analysis, we consider these two kinds of dependencies:
    // (1) dependencies between ops in different streams
    // (2) dependencies between ops in the same stream
    // Before pruning, we add back dependencies of the second kind.
    std::unordered_map<int, std::vector<int>> stream_id_map;
    for (int i = 0; i < max_idx_; ++i) {
      int stream = idx_stream_map[i];
      if (!stream_id_map.count(stream)) {
        stream_id_map.emplace(stream, std::vector<int>{});
      }
      stream_id_map[stream].push_back(i);
    }
    for (auto& stream_and_ids : stream_id_map) {
      auto& ids = stream_and_ids.second;
      std::sort(ids.begin(), ids.end());
      for (int i = 0; i < ids.size() - 1; ++i) {
        dep_set.insert(std::make_pair(ids[i], ids[i + 1]));
      }
    }

    std::unordered_map<int, std::unordered_set<int>> direct_children_map = {};
    std::unordered_map<int, std::unordered_set<int>> direct_parent_map = {};
    std::unordered_set<int> roots = {};
    for (auto& dep_pair : dep_set) {
      roots.insert(dep_pair.first);
      direct_children_map[dep_pair.first].insert(dep_pair.second);
      direct_parent_map[dep_pair.second].insert(dep_pair.first);
    }
    for (auto& dep_pair : dep_set) {
      auto it = roots.find(dep_pair.second);
      if (it != roots.end()) {
        roots.erase(it);
      }
    }

    std::vector<int> post_dfs_order = {};
    std::unordered_map<int, bool> visited = {};
    for (auto root : roots) {
      DFS_(direct_children_map, visited, root, post_dfs_order);
    }

    // Prune redundant dependencies. One dependency (u, v) is redundant if and only if
    // there exists a path from u to v that does not go through the dependency (u, v).
    // Therefore, if dependency (u, v) is redundant, node v should be both direct and
    // indirect child of node u. When pruning, we firstly iterate through the dependency
    // graph in post DFS order to collect direct and indirect children of each node,
    // and then remove redundant edges from the graph.
    std::unordered_map<int, std::unordered_set<int>> indirect_children_map = {};
    for (auto node : post_dfs_order) {
      for (auto parent : direct_parent_map[node]) {
        indirect_children_map[parent].insert(direct_children_map[node].begin(),
                                             direct_children_map[node].end());
        indirect_children_map[parent].insert(indirect_children_map[node].begin(),
                                             indirect_children_map[node].end());
      }
    }
    DepSet deps_to_remove = {};
    for (auto node : post_dfs_order) {
      for (auto direct_child : direct_children_map[node]) {
        if (indirect_children_map[node].count(direct_child)) {
          deps_to_remove.insert(std::make_pair(node, direct_child));
        }
      }
    }

    for (auto& dep_to_remove : deps_to_remove) {
      if (dep_set.count(dep_to_remove)) {
        dep_set.erase(dep_set.find(dep_to_remove));
      }
    }

    // The pruned graph should only contains dependencies of the first kind.
    // Here we remove dependencies of the second kind.
    for (auto it = dep_set.begin(); it != dep_set.end();) {
      if (InSameStream_(it->first, it->second)) {
        it = dep_set.erase(it);
      } else {
        ++it;
      }
    }
  }

  void DFS_(std::unordered_map<int, std::unordered_set<int>>& direct_children_map,
            std::unordered_map<int, bool>& visited, int node, std::vector<int>& post_dfs_order) {
    if (!visited[node]) {
      for (auto child : direct_children_map[node]) {
        DFS_(direct_children_map, visited, child, post_dfs_order);
      }
      post_dfs_order.push_back(node);
      visited[node] = true;
    }
  }

  bool InSameStream_(int i, int j) {
    return idx_stream_map[i] == idx_stream_map[j];
  }

  void CreateEventsUsingDepSet_(DepSet& dep_set) {
    for (auto& dep_pair : dep_set) {
      add_event_after_op[dep_pair.first].push_back(GetUniqueEventId());
      wait_event_before_op[dep_pair.second].push_back(add_event_after_op[dep_pair.first].back());
    }
  }

  // current_idx_ keeps track of current "depth" (or the index of the op we are visiting in the ANF
  // order) when expanding the nested let exprs
  int current_idx_ = 0;
  // max "depth" of expr
  int max_idx_ = 0;
  // stream index of the previous op
  int previous_op_stream_idx_ = -1;
  // maps each var & expr to its index
  ExprMap<int> var_idx_map_;

  // counter to generate unique event ids.
  int next_unique_event_id_ = 1;

  // index refers to the order of an op in the input ANF expression.
  // map each Call, Tuple or TupleGetItem op to the index of the last producer of the op's inputs
  std::unordered_map<int, std::unordered_map<int, int>> last_producer_idx_map_;
  // map each Call, Tuple or TupleGetItem op to the index (in the input ANF expr) of the first
  // consumer of the op's outputs
  std::unordered_map<int, std::unordered_map<int, int>> first_consumer_idx_map_;
};

class SyncEnforcer : ExprVisitor {
 public:
  /*!
   * This pass works in ANF and adds necessary synchronization ops (i.e., set_stream, add_event,
   * wait_event) between communication ops and computation ops to ensure correctness. It does not
   * alter the execution order of ops and assumes single stream computation execution (i.e.
   * sequential stream schedule)
   *
   * Specifically,
   *    1. It inserts a set_stream(device_id, stream_idx) if the op requires the stream be switched
   *       before its execution.
   *
   *    2. It inserts an add_event(unique_event_id, stream_idx) after an op if the following op
   * depends on it executes on a different stream.
   *
   *    3. It inserts a wait_event(unique_event_id, stream_idx) before an op if the previous op it
   * depends on executes on a different stream.
   *
   * Example 1:
   *    Input data flow graph (number represent order in ANF):
   *
   *      atan (1) -> allreduce (2) -> atan(3)
   *
   *    Corresponding IR:
   *
   *      fn (%x: Tensor[(64, 128), float32]) {
   *        let %a1 = raf.op.atan(%x);
   *        let %a2 = (%a1,);
   *        let %a3 = raf.op._allreduce(%a2, str"sum");
   *        let %a4 = raf.op.atan(%a3);
   *        %a4
   *      }
   *
   *    After Transformation:
   *
   *      fn (%x: Tensor[(64, 128), float32]) {
   *        let %set_stream_comp = raf.op.set_stream(int64(0), int64(1));
   *        let %a1 = raf.op.atan(%x);
   *        let %a2 = (%a1,);
   *        let %add_event_comp = raf.op.add_event(int64(1), int64(1));
   *        let %set_stream_comm = raf.op.set_stream(int64(0), int64(5));
   *        let %wait_for_comp = raf.op.wait_event(int64(1), int64(5));
   *        let %a3 = raf.op._allreduce(%a2, str"sum");
   *        let %add_event_comm = raf.op.add_event(int64(2), int64(5));
   *        let %set_stream_comp = raf.op.set_stream(int64(0), int64(1));
   *        let %wait_for_comm = raf.op.wait_event(int64(2), int64(1));
   *        let %a4 = raf.op.atan(%a3);
   *        %a4
   *      }
   *
   * Example 2:
   *    Input data flow graph (number represent order in ANF):
   *
   *     atan (1) -> atan (2) -> allreduce (3) -> mul (5) -> concat (6)
   *        \                                           /
   *         -------------->  allreduce (4) ----------->
   *
   *    Here since allreduce(3) is executed before allreduce(4), and
   *    allreduce(3) depends on atan(2) which is guarenteed to executed
   *    after atan(1). Therefore the dependency atan(1) -> allreduce(4)
   *    is not necessary and we will not add the corresponding events.
   *
   *    Corresponding IR:
   *
   *      fn (%x: Tensor[(64, 128), float32]) {
   *        let %v = raf.op.atan(%x);
   *        let %v1 = (%v,);
   *        let %v2 = raf.op.atan(%v);
   *        let %v3 = (%v2,);
   *        let %v4 = raf.op._allreduce(%v3, str"sum");
   *        let %v5 = raf.op._allreduce(%v1, str"sum");
   *        let %v6 = raf.op.atan(%v4);
   *        let %v7 = (%v6, %v5);
   *        let %v8 = raf.op.concatenate(%v7, int64(0));
   *        %v8
   *      }
   *
   *    After Transformation:
   *
   *      fn (%x: Tensor[(64, 128), float32]) {
   *        let %set_stream_comp = raf.op.set_stream(int64(0), int64(1));
   *        let %v = raf.op.atan(%x);
   *        let %v1 = (%v,);
   *        let %v2 = raf.op.atan(%v);
   *        let %v3 = (%v2,);
   *        let %add_event_comp = raf.op.add_event(int64(1), int64(1));
   *        let %set_stream_comm = raf.op.set_stream(int64(0), int64(5));
   *        let %wait_for_comp = raf.op.wait_event(int64(1), int64(5));
   *        let %v4 = raf.op._allreduce(%v3, str"sum");
   *        let %add_event_comm = raf.op.add_event(int64(2), int64(5));
   *        let %v5 = raf.op._allreduce(%v1, str"sum");
   *        let %add_event_comm1 = raf.op.add_event(int64(3), int64(5));
   *        let %set_stream_comp = raf.op.set_stream(int64(0), int64(1));
   *        let %wait_for_comm = raf.op.wait_event(int64(2), int64(1));
   *        let %v6 = raf.op.atan(%v4);
   *        let %wait_for_comm1 = raf.op.wait_event(int64(3), int64(1));
   *        let %v7 = (%v6, %v5);
   *        let %v8 = raf.op.concatenate(%v7, int64(0));
   *        %v8
   *      }
   */
  explicit SyncEnforcer(const FunctionNode* func) : func_(func) {
  }

  void VisitExpr_(const LetNode* op) {
    Var orig_var = op->var;
    Expr orig_value = op->value;
    int orig_value_idx = analyzer_.expr_idx_map[orig_value];

    SetStreamAndWaitEvent(orig_value_idx);
    ell_->Push(orig_var, orig_value);
    AddEvent(orig_value_idx);

    ell_->ret = ell_->vars.back();
    VisitExpr(op->body);
  }

  Function Run() {
    auto device = Device::Current(/*allow_default=*/false);
    CHECK_NE(device.device_type(), DevType::kUnknown()) << "Encountered unknown device type.";
    device_id_ = device.device_id();
    CHECK_EQ(device_id_, GetGlobalCommunicator()->local_rank) << "Current device id != local rank.";

    if (!analyzer_.Analyse(func_->body)) {
      // no collectives found in expr. do nothing.
      return GetRef<Function>(func_);
    }

    ell_ = std::make_unique<ExplicitLetList>();
    VisitExpr(func_->body);

    return Function(func_->params, ell_->AsExpr(), {}, {});
  }

 protected:
  Expr CreateSetStreamOp(int64_t device_id, int64_t stream_id) {
    static Op set_stream_op = Op::Get("raf.op.set_stream");
    return CreateSetStreamOrEventOp_(set_stream_op, device_id, stream_id);
  }

  Expr CreateAddEventOp(int64_t event_id, int64_t stream_id) {
    static Op add_event_op = Op::Get("raf.op.add_event");
    return CreateSetStreamOrEventOp_(add_event_op, event_id, stream_id);
  }

  Expr CreateWaitEventOp(int64_t event_id, int64_t stream_id) {
    static Op wait_event_op = Op::Get("raf.op.wait_event");
    return CreateSetStreamOrEventOp_(wait_event_op, event_id, stream_id);
  }

  inline void AddEvent(int idx) {
    if (analyzer_.add_event_after_op.count(idx)) {
      int stream = analyzer_.idx_stream_map[idx];
      std::string add_event_var_name = stream_name_hint[stream] + "_add_event";
      for (auto event_id : analyzer_.add_event_after_op[idx]) {
        Var add_event_var = raf::ir::MakeVar(add_event_var_name, {});
        Expr add_event_value = CreateAddEventOp(event_id, stream);
        ell_->Push(add_event_var, add_event_value);
      }
    }
  }

  inline void SetStreamAndWaitEvent(int idx) {
    int stream = analyzer_.idx_stream_map[idx];
    std::string var_name_hint = stream_name_hint[stream];
    if (analyzer_.set_stream_before_op[idx]) {
      std::string set_stream_var_name = var_name_hint + "_set_stream";
      Var set_stream_var = raf::ir::MakeVar(set_stream_var_name, {});
      Expr set_stream_value = CreateSetStreamOp(device_id_, stream);
      ell_->Push(set_stream_var, set_stream_value);
    }
    if (analyzer_.wait_event_before_op.count(idx)) {
      std::string wait_event_var_name = var_name_hint + "_wait_event";
      for (auto event_id : analyzer_.wait_event_before_op[idx]) {
        Var wait_event_var = raf::ir::MakeVar(wait_event_var_name, {});
        Expr wait_event_value = CreateWaitEventOp(event_id, stream);
        ell_->Push(wait_event_var, wait_event_value);
      }
    }
  }

 private:
  Expr CreateSetStreamOrEventOp_(Op& op, int64_t first_arg, int64_t second_arg) {
    Expr first_arg_expr = MakeConstant(value::ScalarValue::make(first_arg));
    Expr second_arg_expr = MakeConstant(value::ScalarValue::make(second_arg));
    Array<Expr> args({first_arg_expr, second_arg_expr});
    return Call(op, args);
  }

  int device_id_ = -1;
  const FunctionNode* func_;
  SyncAnalyzer analyzer_;
  std::unique_ptr<ExplicitLetList> ell_;
};
}  // namespace enforce_sync

Pass EnforceSync() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return enforce_sync::SyncEnforcer(f.operator->()).Run();
  };
  auto func_pass = CreateRAFFunctionPass(pass_func, 0, "EnforceSync", {});
  PassInfo pass_info(0, "EnforceSync", {});
  return RAFSequential({InferType(), func_pass, EraseType()}, pass_info);
}

RAF_REGISTER_GLOBAL("raf.pass_.EnforceSync").set_body_typed(EnforceSync);

}  // namespace pass
}  // namespace raf
