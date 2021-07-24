/*!
 * Copyright (c) 2021 by Contributors
 * \file src/pass/stream_schedule.cc
 * \brief Generate stream schedule for inter-operator execution.
 */
#include <unordered_map>
#include <tvm/node/structural_equal.h>
#include <relay/transforms/pass_utils.h>
#include <relay/analysis/dependency_graph.h>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "mnm/value.h"
#include "mnm/binding.h"
#include "mnm/analysis.h"
#include "let_list.h"
#include "support/arena.h"

namespace mnm {
namespace pass {
namespace stream_schedule {

using namespace mnm::ir;
using namespace mnm::op;
using namespace tvm::relay;
using namespace mnm::analysis;

using Node = DependencyGraph::Node;
using Arena = tvm::support::Arena;
using LinkedList = tvm::relay::LinkedList<Node*>;
using NodeExprMap = std::unordered_map<const Node*, Expr>;

/*! Chain, Wave, and Partition are used to describe a wavefront schedule. */
using Chain = std::vector<Node*>;
using Wave = std::vector<Chain>;
using Partition = std::vector<Wave>;

/*! \brief Get the size (length) of a linked list. */
size_t GetListSize(const LinkedList& list) {
  if (list.head == nullptr) return 0;
  size_t size = 0;
  for (auto p = list.head; p; p = p->next) {
    size++;
  }
  return size;
}

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

/*!
 * The base class of all stream schedulers. A stream scheduler transform a expr in GNF format to
 * scheduled expr in ANF format, injecting stream-related operators (set_stream, add_event, and
 * wait_event).
 */
class StreamSchedulerBase : public ExprMutator {
 public:
  Expr VisitExpr_(const VarNode* var) override {
    return GetRef<Expr>(var);
  }

  Expr VisitExpr_(const GlobalVarNode* var) override {
    return GetRef<Expr>(var);
  }

  Expr VisitExpr_(const FunctionNode* func) override {
    CHECK(func->HasNonzeroAttr(attr::kPrimitive))
        << "Wavefront schedule do not support nested function call now";
    return GetRef<Expr>(func);
  }

  Expr VisitExpr_(const RelayConstantNode* op) override {
    return let_list_.Push(GetRef<Expr>(op));
  }

  Expr VisitExpr_(const OpNode* op) override {
    return GetRef<Expr>(op);
  }

  Expr VisitExpr_(const TupleNode* op) override {
    std::vector<Expr> fields;
    for (auto field : op->fields) {
      fields.push_back(VisitExpr(field));
    }
    return let_list_.Push(Tuple(fields));
  }

  Expr VisitExpr_(const CallNode* c) override {
    std::vector<Expr> args;
    for (const auto& a : c->args) {
      args.push_back(VisitExpr(a));
    }
    return let_list_.Push(Call(VisitExpr(c->op), args, c->attrs, c->type_args));
  }

  Expr VisitExpr_(const TupleGetItemNode* op) override {
    return let_list_.Push(TupleGetItem(VisitExpr(op->tuple), op->index));
  }

 protected:
  Expr AnnotateSetStream(int64_t device_id, int64_t stream_id) {
    static Op op = Op::Get("mnm.op.set_stream");
    Expr device_id_e = MakeConstant(value::ScalarValue::make(device_id));
    Expr stream_id_e = MakeConstant(value::ScalarValue::make(stream_id));
    Array<Expr> args({device_id_e, stream_id_e});
    return let_list_.Push(Call(op, args));
  }

  Expr AnnotateAddEvent(int64_t event_id) {
    static Op op = Op::Get("mnm.op.add_event");
    Expr event_id_e = MakeConstant(value::ScalarValue::make(event_id));
    Array<Expr> args({event_id_e});
    return let_list_.Push(Call(op, args));
  }

  Expr AnnotateWaitEvent(int64_t event_id) {
    static Op op = Op::Get("mnm.op.wait_event");
    Expr event_id_e = MakeConstant(value::ScalarValue::make(event_id));
    Array<Expr> args({event_id_e});
    return let_list_.Push(Call(op, args));
  }

  LetList let_list_;
};

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
   *          order. Meanwhile, it would inject mnm.op.set_stream, mnm.op.add_event, and
   *          mnm.op.wait_event operators to manage the synchronization.
   *
   *  When we finish the above two steps, we get the ANF of the scheduled computation graph.
   *
   * \param e The expr that we want to schedule. It should be a pure dataflow graph expr and should
   *          not contains any node that introduces new scope (such as FunctionNode, LetNode, and
   *          IfNode).
   *
   * \return The schedule expr. Schedule-related operators have been injected.
   */
  Expr Schedule(Expr e) {
    Arena arena;
    DependencyGraph dg = CreateDependencyGraph(&arena, e, true);

    NodeExprMap node_expr;
    for (auto& it : dg.expr_node) {
      node_expr[it.second] = it.first;
    }

    Partition partition = WavefrontPartition(&dg);

    int64_t event_idclock = 0;

    std::unordered_map<int64_t, std::unordered_map<int64_t, int64_t>> chain_event;
    std::unordered_map<int64_t, int64_t> wave_event;

    for (int i = 0; i < partition.size(); i++) {
      Wave& wave = partition.at(i);
      for (int j = 0; j < wave.size(); j++) {
        Chain& chain = wave[j];
        AnnotateSetStream(0, j + 1);
        if (i != 0) {
          // wait for the wave event of the i-1 wave
          AnnotateWaitEvent(wave_event[i - 1]);
        }
        for (Node* node : chain) {
          Expr expr = node_expr.at(node);
          VisitExpr(expr);
        }
        if (i + 1 == partition.size() && j + 1 == wave.size()) {
          // we do not need to add the chain event for the last op in the last wave
          continue;
        }
        chain_event[i][j] = event_idclock++;
        AnnotateAddEvent(chain_event[i][j]);
      }
      if (i + 1 == partition.size()) {
        // we do not need to add the wave event for the last wave
        continue;
      }
      wave_event[i] = event_idclock++;
      // we add an wave event to stream 0 to
      // wait for all chain events of this wave
      AnnotateSetStream(0, 0);
      for (int j = 0; j < wave.size(); j++) {
        AnnotateWaitEvent(chain_event[i][j]);
      }
      AnnotateAddEvent(wave_event[i]);
    }

    size_t num_chains_in_last_wave = partition.back().size();
    CHECK_EQ(num_chains_in_last_wave, 1U)
        << "The last wave can only have a single chain because there is only a single output";
    Expr output_expr = VisitExpr(node_expr.at(partition.back().back().back()));
    return let_list_.Get(output_expr);
  }
};

Expr WavefrontScheduleTransform(const Expr& e) {
  return WavefrontScheduler().Schedule(e);
}

Expr DynamicScheduleTransform(const Expr& e) {
  // TODO(yaoyaoding): finish dynamic schedule
  LOG(FATAL) << "Not implemented.";
  throw;
}
}  // namespace stream_schedule

Pass StreamSchedule() {
  pass::PassContext pass_ctx = pass::PassContext::Current();
  auto policy_name = pass_ctx->GetConfig<tvm::String>("mnm.stream_schedule.policy", "sequential");

  if (policy_name == "sequential") {
    return ToANormalForm();
  } else {
    std::function<Expr(const Expr&)> transform;
    if (policy_name == "wavefront") {
      transform = stream_schedule::WavefrontScheduleTransform;
    } else if (policy_name == "dynamic") {
      transform = stream_schedule::DynamicScheduleTransform;
    } else {
      LOG(FATAL) << "Can not recognize schedule policy: " << policy_name << ", candidates are \n"
                 << "  sequential, wavefront, and dynamic" << std::endl;
    }
    tvm::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
        [=](Function f, IRModule m, PassContext pc) {
          return Downcast<Function>(tvm::relay::TransformF(transform, f));
        };
    return CreateMNMFunctionPass(pass_func, 1, "Schedule", {});
  }
}

MNM_REGISTER_GLOBAL("mnm.pass_.StreamSchedule").set_body_typed(StreamSchedule);
TVM_REGISTER_PASS_CONFIG_OPTION("mnm.stream_schedule.policy", tvm::String);

}  // namespace pass
}  // namespace mnm
