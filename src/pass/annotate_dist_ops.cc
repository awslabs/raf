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
 * \file annotate_dist_ops.cc
 * \brief Add synchronization for communication ops.
 */
#include "mnm/device.h"
#include "mnm/ir.h"
#include "mnm/ir_ext.h"
#include "mnm/dist_context.h"
#include "mnm/pass.h"
#include "mnm/analysis.h"
#include "mnm/op.h"
#include "./common.h"
#include "mnm/stream_pool.h"

namespace mnm {
namespace pass {
namespace annotate_dist_ops {

using tvm::OpAttrMap;
using namespace mnm::ir;
using mnm::distributed::DistContext;
using namespace mnm::analysis;
using op::TMNMCollective;
using stream_pool::StreamTagEnum;

using OpSet = std::unordered_set<Op, ObjectPtrHash, ObjectPtrEqual>;
using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;
template <typename T>
using ExprMap = std::unordered_map<Expr, T, ObjectPtrHash, ObjectPtrEqual>;
using DepList = std::vector<std::pair<int, int>>;

class SyncAnalyzer : ExprVisitor {
 public:
  /*!
   * \brief Analyse the needed dependency edges between ops. It stores the analysis result in
   * add_event_after_op and wait_event_after_op.
   *
   * \returns true if we need to add any dependency edge (i.e. if collective communication ops
   * are found in expr).
   */
  bool Analyse(const Expr& expr) {
    VisitExpr(expr);

    // check if any communication collective is found
    if (comm_calls_.empty()) {
      return false;
    }

    // all synchronization dependencies between computation and communication ops (possibly with
    // redundant ones)
    DepList comp_to_comm_deps;
    DepList comm_to_comp_deps;

    for (auto& comm_call : comm_calls_) {
      int last_producer_idx, first_consumer_idx;
      std::tie(last_producer_idx, first_consumer_idx) = GetProducerConsumerIndices_(comm_call);
      // depended computation -> communication
      if (last_producer_idx != -1) {
        comp_to_comm_deps.push_back(std::make_pair(last_producer_idx, expr_idx_map_[comm_call]));
      }
      // communication -> dependent computation
      if (first_consumer_idx != -1) {
        comm_to_comp_deps.push_back(std::make_pair(expr_idx_map_[comm_call], first_consumer_idx));
      }
    }

    // Now remove any redundant dependency pairs.
    // A dependency pair is redundant if the interval (i.e., [src_idx, dst_idx]) it represents
    // completely overlaps the interval of another dependency pair. We find such intervals by
    // sorting them by their src_idx and check whether a interval's dst_idx is larger or equal
    // to that of its successors. Comm -> comp and comp -> comm dependencies are considered
    // separately.

    auto required_comp_to_comm_deps = RemoveRedundantDependency_(comp_to_comm_deps);
    auto required_comm_to_comp_deps = RemoveRedundantDependency_(comm_to_comp_deps);

    // create events for each remaining dependency pair
    for (auto& dep_pair : required_comp_to_comm_deps) {
      // depended computation -> communication
      CHECK(!add_event_after_op.count(exprs_[dep_pair.first]))
          << "Found redundant add_event after a computation op.";
      add_event_after_op[exprs_[dep_pair.first]] = GetUniqueEventId();
      CHECK(!wait_event_before_op.count(exprs_[dep_pair.second]))
          << "Found redundant wait_event before a communication op.";
      wait_event_before_op[exprs_[dep_pair.second]] = add_event_after_op[exprs_[dep_pair.first]];
    }
    for (auto& dep_pair : required_comm_to_comp_deps) {
      // communication -> dependent computation
      CHECK(!add_event_after_op.count(exprs_[dep_pair.first]))
          << "Found redundant add_event after a communication op.";
      add_event_after_op[exprs_[dep_pair.first]] = GetUniqueEventId();
      CHECK(!wait_event_before_op.count(exprs_[dep_pair.second]))
          << "Found redundant wait_event before a computation op.";
      wait_event_before_op[exprs_[dep_pair.second]] = add_event_after_op[exprs_[dep_pair.first]];
    }
    return true;
  }

  void VisitExpr_(const VarNode* var) {
    var_idx_map_[GetRef<Var>(var)] = current_idx_;
  }

  void VisitExpr_(const CallNode* call) {
    if (fcollective_ops_.get(call->op, false)) {
      comm_calls_.insert(GetRef<Expr>(call));
    }
    UpdateDependencyInfo_(GetRef<Expr>(call), call->args);
  }

  void VisitExpr_(const TupleNode* tuple) {
    UpdateDependencyInfo_(GetRef<Expr>(tuple), tuple->fields);
  }

  void VisitExpr_(const TupleGetItemNode* tuple_get_item) {
    UpdateDependencyInfo_(GetRef<Expr>(tuple_get_item), {tuple_get_item->tuple});
  }

  void VisitExpr_(const FunctionNode* op) {
    // currently assumes closure does not contain collectives
    // TODO(@chenyu-jiang): move this pass after lambda lift and enable it
    // to track event ids used across differnt runs
    auto ell = ExplicitLetList::make(op->body);
    for (auto& expr : ell->exprs) {
      CHECK(!(expr.as<CallNode>() && fcollective_ops_.get(expr.as<CallNode>()->op, false)))
          << "Unimplemented: Collectives in closures are currently not supported.";
    }
  }

  void VisitExpr_(const LetNode* op) {
    auto pre_visit = [this](const LetNode* op) {
      Var var = op->var;
      Expr value = op->value;
      expr_idx_map_[value] = current_idx_;
      this->VisitExpr(var);
      this->VisitExpr(value);
      current_idx_++;
      exprs_.push_back(value);
    };
    auto post_visit = [this](const LetNode* op) {
      Expr body = op->body;
      this->VisitExpr(body);
      current_idx_--;
    };
    ExpandANormalForm(op, pre_visit, post_visit);
  }

  // maps Expr -> added event_id
  std::map<Expr, int> add_event_after_op;
  // maps Expr -> event_id to wait for
  std::map<Expr, int> wait_event_before_op;

 private:
  int GetUniqueEventId() {
    return next_unique_event_id_++;
  }

  void UpdateDependencyInfo_(const Expr& op, const Array<Expr>& input_args) {
    for (auto arg : input_args) {
      if (var_idx_map_.count(arg)) {
        // update producer map for communication ops
        if (comm_calls_.count(op)) {
          if (!last_producer_idx_map_.count(op)) {
            last_producer_idx_map_[op] = var_idx_map_[arg];
          } else {
            last_producer_idx_map_[op] = std::max(last_producer_idx_map_[op], var_idx_map_[arg]);
          }
        }

        auto arg_expr = exprs_[var_idx_map_[Downcast<Var>(arg)]];
        // update consumer map for communication ops
        if (comm_calls_.count(arg_expr)) {
          if (!first_consumer_idx_map_.count(arg_expr)) {
            first_consumer_idx_map_[arg_expr] = current_idx_;
          } else {
            first_consumer_idx_map_[arg_expr] =
                std::min(first_consumer_idx_map_[arg_expr], current_idx_);
          }
        }
      }
    }
  }

  // returns (index of the last input producer, index of the first output consumer) of a
  // communication op. the corresponding value is -1 if there is no such producer/consumer, which
  // can happen if the op takes only function inputs as args or produces function return values.
  std::pair<int, int> GetProducerConsumerIndices_(Expr op) {
    int last_producer_idx = -1, first_consumer_idx = -1;
    if (last_producer_idx_map_.count(op)) {
      last_producer_idx = last_producer_idx_map_.at(op);
    }
    if (first_consumer_idx_map_.count(op)) {
      first_consumer_idx = first_consumer_idx_map_.at(op);
    }
    return std::make_pair(last_producer_idx, first_consumer_idx);
  }

  DepList RemoveRedundantDependency_(DepList& dependencies) {
    if (dependencies.empty()) return {};
    std::vector<std::pair<int, int>> filtered_dependencies;
    std::sort(dependencies.begin(), dependencies.end());
    for (auto& dep : dependencies) {
      while (!filtered_dependencies.empty() && filtered_dependencies.back().second >= dep.second) {
        filtered_dependencies.pop_back();
      }
      if (filtered_dependencies.empty() || !(filtered_dependencies.back().first == dep.first &&
                                             filtered_dependencies.back().second < dep.second)) {
        filtered_dependencies.push_back(dep);
      }
    }
    return filtered_dependencies;
  }

  // current_idx_ keeps track of current "depth" (or the index of the op we are visiting in the ANF
  // order) when expanding the nested let exprs
  int current_idx_ = 0;
  // maps each var & expr to its index
  ExprMap<int> var_idx_map_;
  ExprMap<int> expr_idx_map_;
  // list of exprs in ANF order
  Array<Expr> exprs_;
  // set of Call exprs to communication ops
  ExprSet comm_calls_;

  // counter to generate unique event ids.
  int next_unique_event_id_ = 1;

  // index refers to the order of an op in the input ANF expression.
  // map each Call, Tuple or TupleGetItem op to the index of the last producer of the op's inputs
  std::map<Expr, int> last_producer_idx_map_;
  // map each Call, Tuple or TupleGetItem op to the index (in the input ANF expr) of the first
  // consumer of the op's outputs
  std::map<Expr, int> first_consumer_idx_map_;

  OpAttrMap<TMNMCollective> fcollective_ops_ = Op::GetAttrMap<TMNMCollective>("TMNMCollective");
};

class DistOpAnnotator : ExprMutator {
 public:
  /*!
   * This pass works in ANF and adds necessary synchronization ops (i.e., set_stream, add_event,
   * wait_event) between communication ops and computation ops to ensure correctness. It does not
   * alter the execution order of ops and assumes single stream computation execution (i.e.
   * sequential stream schedule)
   *
   * Specifically,
   *    1. It inserts set_stream(device_id, comm_stream_id) & set_stream(device_id,
   *       default_comp_stream_id) in the beginning of the program to make vm allocate the streams
   *       needed.
   *
   *    2. For each communication op, it
   *       (1) inserts an add_event(unique_event_id, default_comp_stream_id) after the last
   *           computation op it depends on
   *       (2) inserts a wait_event(unique_event_id, comm_stream_id) before the communication op
   *       (3) inserts an add_event(another_unique_event_id, comm_stream_id) after the communication
   *           op
   *       (4) inserts a wait_event(another_unique_event_id, default_comp_stream_id) before the
   *           first computation op that depends on the communication op
   *
   *    3. It removes any dependency / events that is not needed (pls. see example below for
   *       details).
   *
   * Example 1:
   *    Input data flow graph (number represent order in ANF):
   *
   *      atan (1) -> allreduce (2) -> atan(3)
   *
   *    Corresponding IR:
   *
   *      fn (%x: Tensor[(64, 128), float32]) {
   *        let %a1 = mnm.op.atan(%x);
   *        let %a2 = (%a1,);
   *        let %a3 = mnm.op._allreduce(%a2, str"sum");
   *        let %a4 = mnm.op.atan(%a3);
   *        %a4
   *      }
   *
   *    After Transformation:
   *
   *      fn (%x: Tensor[(64, 128), float32]) {
   *        let %set_stream_comm = mnm.op.set_stream(int64(0), int64(5));
   *        let %set_stream_comp = mnm.op.set_stream(int64(0), int64(1));
   *        let %a1 = mnm.op.atan(%x);
   *        let %a2 = (%a1,);
   *        let %add_event_comp = mnm.op.add_event(int64(1), int64(1));
   *        let %wait_for_comp = mnm.op.wait_event(int64(1), int64(5));
   *        let %a3 = mnm.op._allreduce(%a2, str"sum");
   *        let %add_event_comm = mnm.op.add_event(int64(2), int64(5));
   *        let %wait_for_comm = mnm.op.wait_event(int64(2), int64(1));
   *        let %a4 = mnm.op.atan(%a3);
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
   *        let %v = mnm.op.atan(%x);
   *        let %v1 = (%v,);
   *        let %v2 = mnm.op.atan(%v);
   *        let %v3 = (%v2,);
   *        let %v4 = mnm.op._allreduce(%v3, str"sum");
   *        let %v5 = mnm.op._allreduce(%v1, str"sum");
   *        let %v6 = mnm.op.atan(%v4);
   *        let %v7 = (%v6, %v5);
   *        let %v8 = mnm.op.concatenate(%v7, int64(0));
   *        %v8
   *      }
   *
   *    After Transformation:
   *
   *      fn (%x: Tensor[(64, 128), float32]) {
   *        let %set_stream_comm = mnm.op.set_stream(int64(0), int64(5));
   *        let %set_stream_comp = mnm.op.set_stream(int64(0), int64(1));
   *        let %v = mnm.op.atan(%x);
   *        let %v1 = (%v,);
   *        let %v2 = mnm.op.atan(%v);
   *        let %v3 = (%v2,);
   *        let %add_event_comp = mnm.op.add_event(int64(1), int64(1));
   *        let %wait_for_comp = mnm.op.wait_event(int64(1), int64(5));
   *        let %v4 = mnm.op._allreduce(%v3, str"sum");
   *        let %add_event_comm = mnm.op.add_event(int64(2), int64(5));
   *        let %v5 = mnm.op._allreduce(%v1, str"sum");
   *        let %add_event_comm1 = mnm.op.add_event(int64(3), int64(5));
   *        let %wait_for_comm = mnm.op.wait_event(int64(2), int64(1));
   *        let %v6 = mnm.op.atan(%v4);
   *        let %wait_for_comm1 = mnm.op.wait_event(int64(3), int64(1));
   *        let %v7 = (%v6, %v5);
   *        let %v8 = mnm.op.concatenate(%v7, int64(0));
   *        %v8
   *      }
   */
  explicit DistOpAnnotator(const FunctionNode* func) : func_(func) {
  }

  Expr VisitExpr_(const LetNode* op) {
    auto pre_visit = [this](const LetNode* op) {};
    auto post_visit = [this](const LetNode* op) {
      Var var = op->var;
      Expr value = op->value;
      Expr body = this->VisitExpr(op->body);
      // insert add_events after op
      if (this->analyzer_.add_event_after_op.count(value)) {
        if (value.as<CallNode>() && fcollective_ops_.get(value.as<CallNode>()->op, false)) {
          // this is a communication op
          Var event_var = mnm::ir::MakeVar("add_event_comm", {});
          Expr event_value = CreateAddEventOp(this->analyzer_.add_event_after_op.at(value),
                                              this->communication_stream_idx_);
          body = Let(event_var, event_value, body);
        } else {
          // this is a computation op
          Var event_var = mnm::ir::MakeVar("add_event_comp", {});
          Expr event_value = CreateAddEventOp(this->analyzer_.add_event_after_op.at(value),
                                              this->compute_stream_idx_);
          body = Let(event_var, event_value, body);
        }
      }
      Expr orig_op = GetRef<Expr>(op);
      // insert wait_events before op
      if (this->analyzer_.wait_event_before_op.count(value)) {
        if (value.as<CallNode>() && fcollective_ops_.get(value.as<CallNode>()->op, false)) {
          // this is a communication op
          body = Let(var, value, body);
          var = mnm::ir::MakeVar("wait_for_comp", {});
          value = CreateWaitEventOp(this->analyzer_.wait_event_before_op.at(value),
                                    this->communication_stream_idx_);
        } else {
          // this is a computation op
          body = Let(var, value, body);
          var = mnm::ir::MakeVar("wait_for_comm", {});
          value = CreateWaitEventOp(this->analyzer_.wait_event_before_op.at(value),
                                    this->compute_stream_idx_);
        }
      }
      this->memo_[orig_op] = Let(var, value, body);
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return this->memo_[GetRef<Expr>(op)];
  }

  Function Run() {
    auto device = Device::Current(/*allow_default=*/false);
    CHECK_NE(device.device_type(), DevType::kUnknown()) << "Encountered unknown device type.";
    auto device_id = device.device_id();
    CHECK_EQ(device_id, DistContext::Global()->local_rank) << "Current device id != local rank.";

    if (!analyzer_.Analyse(func_->body)) {
      // no collectives found in expr. do nothing.
      return GetRef<Function>(func_);
    }

    Expr new_body = Mutate(func_->body);

    // add the two set_stream ops
    new_body = Let(mnm::ir::MakeVar("set_stream_comp", {}),
                   CreateSetStreamOp(device_id, compute_stream_idx_), new_body);
    new_body = Let(mnm::ir::MakeVar("set_stream_comm", {}),
                   CreateSetStreamOp(device_id, communication_stream_idx_), new_body);

    return Function(func_->params, new_body, {}, {});
  }

 protected:
  Expr CreateSetStreamOp(int64_t device_id, int64_t stream_id) {
    static Op set_stream_op = Op::Get("mnm.op.set_stream");
    return CreateSetStreamOrEventOp_(set_stream_op, device_id, stream_id);
  }

  Expr CreateAddEventOp(int64_t event_id, int64_t stream_id) {
    static Op add_event_op = Op::Get("mnm.op.add_event");
    return CreateSetStreamOrEventOp_(add_event_op, event_id, stream_id);
  }

  Expr CreateWaitEventOp(int64_t event_id, int64_t stream_id) {
    static Op wait_event_op = Op::Get("mnm.op.wait_event");
    return CreateSetStreamOrEventOp_(wait_event_op, event_id, stream_id);
  }

 private:
  Expr CreateSetStreamOrEventOp_(Op& op, int64_t first_arg, int64_t second_arg) {
    Expr first_arg_expr = MakeConstant(value::ScalarValue::make(first_arg));
    Expr second_arg_expr = MakeConstant(value::ScalarValue::make(second_arg));
    Array<Expr> args({first_arg_expr, second_arg_expr});
    return Call(op, args);
  }

  const FunctionNode* func_;

  int64_t compute_stream_idx_ = StreamTagEnum::CudaCompute();
  int64_t communication_stream_idx_ = StreamTagEnum::CudaCommunicate();

  SyncAnalyzer analyzer_;
  OpAttrMap<TMNMCollective> fcollective_ops_ = Op::GetAttrMap<TMNMCollective>("TMNMCollective");
};

}  // namespace annotate_dist_ops

Pass AnnotateDistOps() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return annotate_dist_ops::DistOpAnnotator(f.operator->()).Run();
  };
  return CreateMNMFunctionPass(pass_func, 0, "AnnotateDistOps", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.AnnotateDistOps").set_body_typed(AnnotateDistOps);

}  // namespace pass
}  // namespace mnm
