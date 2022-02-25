/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file memory_schedule.cc
 * \brief Schedule ANF IR to reduce memory footprint.
 */
#include <tvm/ir/type_functor.h>
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"

#include "./common.h"
#include "./let_list.h"
#include "../common/shape_utils.h"

namespace raf {
namespace pass {
namespace memory_schedule {

using namespace raf::op;
using common::shape_utils::BytesCompactType;

template <typename T>
using StdMap = std::unordered_map<Var, T, ObjectPtrHash, ObjectPtrEqual>;
using VSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

// The threashold (in MBs) to move schedule nodes.
constexpr float kMoveThreshold = 1.0f;
constexpr float kMegaBytes = 1048576;

/*! \brief A schedule node that forms a doublely linked-list. */
struct ScheduleNode {
  /*! \brief The let-binding var. */
  const Var var;
  /*! \brief The corresponding expression. */
  const Expr expr;
  /*! \brief Size in MBs. */
  float size = 0;
  /*! \brief If not nullptr, this var shares the memory of another var. */
  std::shared_ptr<ScheduleNode> may_share = nullptr;
  /*! \brief A set of schedule nodes that are needed to generate this var. */
  std::unordered_set<std::shared_ptr<ScheduleNode>> def_set;
  /*! \brief A set of schedule nodes that use this var. */
  std::unordered_set<std::shared_ptr<ScheduleNode>> use_set;
  /*! \brief The cached last def, which is the last schedule node that defines
   * the argument of this note. */
  std::shared_ptr<ScheduleNode> last_def = nullptr;
  /*! \brief The cached first use, which is the first schedule node that uses this node. */
  std::shared_ptr<ScheduleNode> first_use = nullptr;
  /*! \brief The cached last use, which is the last schedule node that uses this node.
   * Note that last use is not applicable to the return node. */
  std::shared_ptr<ScheduleNode> last_use = nullptr;
  /*! \brief Pointers for doubly linked list. */
  std::shared_ptr<ScheduleNode> prev = nullptr, next = nullptr;

  ScheduleNode(const Var var, const Expr expr, std::shared_ptr<ScheduleNode> prev,
               const StdMap<std::shared_ptr<ScheduleNode>>& schedule_map)
      : var(var), expr(expr), prev(prev) {
    if (!var.defined()) {
      // Do nothing for dummy node.
      return;
    }
    if (auto ext_var = var.as<ExtendedVarNode>()) {
      if (ext_var->may_share.defined()) {
        CHECK_GT(schedule_map.count(ext_var->may_share), 0U);
        may_share = schedule_map.at(ext_var->may_share);
      }
    }
    size = BytesCompactType(var->checked_type()) / kMegaBytes;
  }
};

using ScheduleNodePtr = std::shared_ptr<ScheduleNode>;

/*!
 * \brief Schedule ANF IR to reduce memory footprint.
 * The basic algorithm is:
 * 1. Build a doubly linked-list of let-binding vars.
 * 2. Build a def-use map to represent the dependency between vars.
 * 3. Traverse the linked list from head to tail, and move the vars that reduces memory footprint
 *    to the earliest position it can be.
 * 4. Traverse the linked list from tail to head, and move the vars that increases memory footprint
 *    to the latest position it can be.
 * 5. Repeat step 3-4 for several times or no more changes.
 * 6. Construct a new ANF IR based on the manipulated linked-list.
 */
class ANFScheduler4Memory {
 public:
  explicit ANFScheduler4Memory(const Function& func)
      : func_(func), ell_(ExplicitLetList::make(func->body)) {
  }

  /*! \brief Dump the schedule nodes for debugging. */
  std::string DumpSchedule(bool show_details = false) {
    std::stringstream ss;
    auto curr = sch_head_->next;
    while (curr != sch_tail_) {
      ss << curr->var->name_hint();
      if (show_details) {
        ss << ": " << std::endl << "  Prev: ";
        if (curr->prev != nullptr && curr->prev->var.defined()) {
          ss << curr->prev->var->name_hint();
        } else {
          ss << "N/A";
        }
        ss << std::endl << "  Next: ";
        if ((curr->next != nullptr && curr->next->var.defined())) {
          ss << curr->next->var->name_hint();
        } else {
          ss << "N/A";
        }
        ss << std::endl << "  Def: ";
        for (const auto& dep_node : curr->def_set) {
          ss << dep_node->var->name_hint() << ", ";
        }
        ss << std::endl << "  Use: ";
        for (const auto& dep_node : curr->use_set) {
          ss << dep_node->var->name_hint() << ", ";
        }
      }
      ss << std::endl;
      curr = curr->next;
    }
    return ss.str();
  }

  Expr Run() {
    if (!Init()) {
      return func_;
    }

    const auto& vars = ell_->vars;
    const auto& exprs = ell_->exprs;
    auto size = vars.size();

    int changed = 1;
    for (size_t iter = 0; iter < 5 && changed > 0; ++iter) {
      changed = 0;
      ScheduleNodePtr curr_node = nullptr;

      // Forward pass that moves the let-binding vars to the left (up).
      curr_node = sch_head_->next;
      while (curr_node != sch_tail_) {
        const auto& var = curr_node->var;
        const auto& expr = curr_node->expr;

        auto next_node = curr_node->next;
        if (auto call = expr.as<CallNode>()) {
          auto inc_size = CalcIncMemSize(curr_node);
          std::string op_name =
              (call->op->IsInstance<OpNode>()) ? call->op.as<OpNode>()->name : "fn";
          DLOG(INFO) << "Forward: " << var->name_hint() << ", " << op_name << ", " << inc_size
                     << "..." << ((inc_size < -kMoveThreshold) ? "ASAP" : "Stay");
          if (inc_size < -kMoveThreshold) {  // ASAP
            // Move the current node to the right of the last def node.
            auto last_def_node = GetLastDef(curr_node);
            CHECK(last_def_node != nullptr) << "The node that decreases memory footprint must have "
                                               "one or more def node, but not found for "
                                            << var->name_hint();
            changed += MoveScheduleNode(curr_node, last_def_node, true);
          }
        }
        curr_node = next_node;
      }

      // Backward pass that moves the let-binding vars to the right (down).
      curr_node = sch_tail_->prev;
      while (curr_node != sch_head_) {
        const auto& var = curr_node->var;
        const auto& expr = curr_node->expr;

        auto prev_node = curr_node->prev;
        if (auto call = expr.as<CallNode>()) {
          auto inc_size = CalcIncMemSize(curr_node);
          std::string op_name =
              (call->op->IsInstance<OpNode>()) ? call->op.as<OpNode>()->name : "fn";
          DLOG(INFO) << "Backward: " << var->name_hint() << ", " << op_name << ", " << inc_size
                     << "..." << ((inc_size > kMoveThreshold) ? "ALAP" : "Stay");
          if (inc_size > kMoveThreshold) {  // ALAP
            auto first_use_node = GetFirstUse(curr_node);
            if (first_use_node) {  // Ingore dead node or return node.
              // Move the current node to the left of the first use node.
              changed += MoveScheduleNode(curr_node, first_use_node, false);
            }
          }
        }
        curr_node = prev_node;
      }
      DLOG(INFO) << "Iter " << iter << ": " << changed;
    }

    // Construct new ANF IR based on the schedule node list.
    Expr new_body = LetList::With([&](LetList* ll) {
      auto curr_node = sch_head_->next;
      Var curr_let;
      while (curr_node != sch_tail_) {
        if (curr_node->expr.defined()) {  // Parameter nodes do not have expressions.
          ll->Push(curr_node->var, curr_node->expr);
        }
        curr_let = curr_node->var;
        curr_node = curr_node->next;
      }
      return curr_let;
    });
    return Function(func_->params, new_body, func_->ret_type, func_->type_params, func_->attrs);
  }

 private:
  class DefUseAnalyzer;

  StdMap<std::pair<VSet, VSet>> BuildDefUseMap(const Function& func);

  bool MoveScheduleNode(ScheduleNodePtr node, ScheduleNodePtr target_node, bool to_right) {
    CHECK(node != nullptr && target_node != nullptr);
    if ((!to_right && node == target_node->prev) || (to_right && node == target_node->next)) {
      DLOG(INFO) << "  No need to move";
      return false;
    }
    DLOG(INFO) << "  Move " << node->var->name_hint() << " to " << (to_right ? "right" : "left")
               << " of " << target_node->var->name_hint();

    // Remove the node from the current position.
    node->prev->next = node->next;
    node->next->prev = node->prev;

    // Find the schedule node pointers of the target position and invalid corresponding
    // cached first-use and last-def.
    ScheduleNodePtr new_prev, new_next;
    if (to_right) {
      new_prev = target_node;
      new_next = target_node->next;
    } else {
      new_prev = target_node->prev;
      new_next = target_node;
    }

    // Move a node may change the first/last-def/use of all nodes using/defining this node,
    // so invalid them here to let them be re-evaluated when needed.
    for (auto dep_node : node->use_set) {
      dep_node->last_def = nullptr;
    }
    for (auto dep_node : node->def_set) {
      dep_node->first_use = nullptr;
      dep_node->last_use = nullptr;
    }

    // Insert the node to the target position.
    new_prev->next = node;
    node->prev = new_prev;
    node->next = new_next;
    new_next->prev = node;
    return true;
  }

  /*!
   * \brief Get the last schedule node that defines the given node.
   * \param node The node to be evaluated.
   * \return The last schedule node that defines the given node, or nullptr if the given node
   * only uses parameters.
   */
  ScheduleNodePtr GetLastDef(ScheduleNodePtr node) {
    if (node->last_def != nullptr) {
      // Cached last-def.
      return node->last_def;
    }
    if (node->def_set.empty()) {
      // This node only uses parameters so no def node.
      return nullptr;
    }

    // Start from an arbitrary def node and go right until the target node.
    // By that time, the last found def node is the last def of the target node.
    ScheduleNodePtr last_def = nullptr;
    auto curr_node = *(node->def_set.begin());
    last_def = curr_node;
    while (curr_node != sch_tail_ && curr_node != node) {
      if (node->def_set.find(curr_node) != node->def_set.end()) {
        last_def = curr_node;
      }
      curr_node = curr_node->next;
    }
    node->last_def = last_def;
    return last_def;
  }

  /*!
   *\brief Get the first schedule node that uses the given node.
   * \param node The node to be evaluated.
   * \return The first schedule node that uses the given node, or nullptr if the given node
   * is never used (dead code or output node).
   */
  ScheduleNodePtr GetFirstUse(ScheduleNodePtr node) {
    if (node->first_use != nullptr) {
      return node->first_use;
    }
    if (node->use_set.empty()) {
      // This node may be dead or the return node.
      return nullptr;
    }

    // Start from an arbitrary use node and go left until the target node.
    // By that time, the last found use node is the first use of the target node.
    ScheduleNodePtr first_use = nullptr;
    auto curr_node = *(node->use_set.begin());
    first_use = curr_node;
    while (curr_node != sch_head_ && curr_node != node) {
      if (node->use_set.find(curr_node) != node->use_set.end()) {
        first_use = curr_node;
      }
      curr_node = curr_node->prev;
    }
    node->first_use = first_use;
    return first_use;
  }

  /*!
   *\brief Get the last schedule node that uses the given node.
   * \param node The node to be evaluated.
   * \return The last schedule node that uses the given node, or nullptr if the given node
   * is never used (dead code or output node).
   */
  ScheduleNodePtr GetLastUse(ScheduleNodePtr node) {
    if (node->last_use != nullptr) {
      return node->last_use;
    }
    if (node->use_set.empty()) {
      // This node may be dead or the return node.
      return nullptr;
    }

    // Start from an arbitrary use node and go right until the end or all use nodes are visited.
    // By that time, the last found use node is the last use of the target node.
    ScheduleNodePtr last_use = nullptr;
    auto curr_node = *(node->use_set.begin());
    last_use = curr_node;
    int visited = 0;  // The arbitrary selected use node will be visited again so start from 0.
    while (curr_node != sch_tail_ && visited < node->use_set.size()) {
      if (node->use_set.find(curr_node) != node->use_set.end()) {
        last_use = curr_node;
        visited++;
      }
      curr_node = curr_node->next;
    }
    node->last_use = last_use;
    return last_use;
  }

  /*!
   * \brief Caclculate the increased memory size after executing the give node.
   * Note that if one or more arguments will be freed after this node, then
   * the increased memory size may be negative.
   * \param node The node to be evaluated.
   * \return The increased memory size after executing the given node.
   */
  float CalcIncMemSize(ScheduleNodePtr node) {
    CHECK_NE(node->size, 0);

    float out_size = (node->may_share != nullptr) ? 0 : node->size;
    float inc_size = 0;
    if (auto call = node->expr.as<CallNode>()) {
      float free_size = 0;
      for (const auto& arg : call->args) {
        if (!arg->IsInstance<VarNode>()) {
          // Skip constants.
          continue;
        }
        auto arg_var = Downcast<Var>(arg);
        if (var_sch_map_.count(arg_var) == 0) {
          // Skip parameters.
          continue;
        }
        auto arg_sch_node = var_sch_map_[arg_var];
        CHECK_NE(arg_sch_node->size, 0);
        if (GetLastUse(arg_sch_node) != node) {
          // Skip the argument if it will be used by others after this node.
          continue;
        }
        free_size += arg_sch_node->size;
      }
      inc_size = out_size - free_size;
    }
    return inc_size;
  }

  /*! \brief Initialize the schedule linked list and def-use map. */
  bool Init() {
    const auto& vars = ell_->vars;
    const auto& exprs = ell_->exprs;
    auto size = vars.size();

    // Sanity check to deal with the case that the ANF graph has no let-bindings
    // (only include a single return expression.)
    if (size == 0) {
      return false;
    }

    // Build the linked list of schedule nodes.
    sch_head_ = std::make_shared<ScheduleNode>(Var(), Expr(), nullptr, var_sch_map_);
    ScheduleNodePtr prev = sch_head_;

    // Add input parameters to the linked list. Note that their orders do not matter.
    for (auto param : func_->params) {
      auto curr = std::make_shared<ScheduleNode>(param, Expr(), prev, var_sch_map_);
      if (curr->size == 0) {
        LOG(WARNING) << "Cannot schedule " << param->name_hint() << " due to dynamic size";
        return false;
      }
      var_sch_map_[param] = curr;
      prev->next = curr;
      prev = curr;
    }

    // Add let-binding vars to the linked list in order.
    for (size_t i = 0; i < size; ++i) {
      auto var = vars[i];
      auto curr = std::make_shared<ScheduleNode>(var, exprs[i], prev, var_sch_map_);
      if (curr->size == 0) {
        LOG(WARNING) << "Cannot schedule " << var->name_hint() << " due to dynamic size";
        return false;
      }
      var_sch_map_[var] = curr;
      prev->next = curr;
      prev = curr;
    }
    sch_tail_ = std::make_shared<ScheduleNode>(Var(), Expr(), prev, var_sch_map_);
    prev->next = sch_tail_;

    // In the case of "let %a (share: %p) = ...", this map maps from
    // SchNode(%b) to {SchNode(%a), is_defined}. "is_defined" is used later to indicate
    // whether the var in Use(%p) should be added to the def or use set of SchNode(%b).
    std::unordered_map<ScheduleNodePtr, std::pair<ScheduleNodePtr, bool>> inplace_map;

    // Build the def-use map.
    auto def_use_map = BuildDefUseMap(func_);
    for (const auto& kv : def_use_map) {
      auto var = kv.first;
      auto sch_node = var_sch_map_[var];

      if (sch_node->may_share != nullptr) {
        inplace_map[sch_node->may_share] = {sch_node, false};
      }

      // Build def set.
      for (const auto& dep_var : kv.second.first) {
        CHECK_GT(var_sch_map_.count(dep_var), 0);
        sch_node->def_set.insert(var_sch_map_[dep_var]);
      }

      // Build use set.
      for (const auto& dep_var : kv.second.second) {
        CHECK_GT(var_sch_map_.count(dep_var), 0);
        sch_node->use_set.insert(var_sch_map_[dep_var]);
      }
    }

    // Consider in-place updates in def-use maps.
    auto curr_node = sch_head_->next;
    while (curr_node != sch_tail_) {
      for (auto def_node : curr_node->def_set) {
        if (inplace_map.count(def_node)) {
          // If the current node needs a param that is shared memory with the other var,
          // Taking the following case as an example:
          //   let %a10 = op(%in, %weight);
          //   ...
          //   let %a15(share: %weight) = op(%a1);
          //   ..
          //   let %a20 = op(..., %weight);
          //   let %a21 = op(%a15);
          // In this case, we need to limit the schedule range of %a15 to [%a10, %a20]:
          //   def(%a15)={%a1, %a10}, and use(%a15)={%a20,%a21}.
          // We also need to limit the schedule range of the one that uses %weight:
          //   use(%10)={%a15}, def(%a20)={%weight, %a15}.
          if (inplace_map[def_node].first == curr_node) {
            continue;
          }
          if (!inplace_map[def_node].second) {
            // curr_node is %a10, inplace_map[def_node] is %a15.
            inplace_map[def_node].first->def_set.insert(curr_node);
            curr_node->use_set.insert(inplace_map[def_node].first);
          } else {
            // curr_node is %a20, inplace_map[def_node] is %a15.
            inplace_map[def_node].first->use_set.insert(curr_node);
            curr_node->def_set.insert(inplace_map[def_node].first);
          }
        }
      }

      // curr_node is %a15 in the above example. Mark "is_defined" as true means
      // the rest use of its shared var should be added to its use set.
      if (curr_node->may_share != nullptr) {
        CHECK_GT(inplace_map.count(curr_node->may_share), 0U);
        inplace_map[curr_node->may_share].second = true;
      }

      curr_node = curr_node->next;
    }

    DLOG(INFO) << "Schedule Info:" << std::endl << DumpSchedule(true);
    return true;
  }

  /*! \brief the function to be muatated. */
  const Function& func_;
  /*! \brief The let list. */
  std::unique_ptr<ExplicitLetList> ell_{nullptr};
  /*! \brief The dummy head/tail node of the schedule linked list. */
  ScheduleNodePtr sch_head_ = nullptr, sch_tail_ = nullptr;
  /*! \brief Mapping from var to its schedule node pointer. */
  StdMap<ScheduleNodePtr> var_sch_map_;
};

/*!
 * \brief A simple visitor to analyze the def-use of each tensor.
 */
class ANFScheduler4Memory::DefUseAnalyzer : public ExprVisitor {
 public:
  DefUseAnalyzer(const Function& func) : ell_(ExplicitLetList::make(func)) {
  }

  StdMap<std::pair<VSet, VSet>> Run() {
    const auto& vars = ell_->vars;
    const auto& exprs = ell_->exprs;
    CHECK_EQ(vars.size(), exprs.size());

    size_t size = exprs.size();
    for (int i = 0; i < size; ++i) {
      curr_let_ = vars[i];
      ExprVisitor::VisitExpr(exprs[i]);
    }

    return def_use_map_;
  }

  void VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      this->VisitExpr(op->var);
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      this->VisitExpr(op->body);
      this->visit_counter_[op] += 1;
    };
    ExpandANormalForm(op, pre_visit, post_visit);
  }

  void VisitExpr_(const VarNode* node) override {
    AddDefUse(GetRef<Var>(node));
  }

  void VisitExpr_(const CallNode* node) override {
    for (auto arg : node->args) {
      if (auto var_node = arg.as<VarNode>()) {
        AddDefUse(GetRef<Var>(var_node));
      }
    }
  }

  void VisitExpr_(const TupleNode* node) override {
    for (auto field : node->fields) {
      if (auto var_node = field.as<VarNode>()) {
        AddDefUse(GetRef<Var>(var_node));
      }
    }
  }

 private:
  inline void AddDefUse(const Var& var) {
    def_use_map_[curr_let_].first.insert(var);
    def_use_map_[var].second.insert(curr_let_);
  }

  /*! \brief The current let-binding var. */
  Var curr_let_;
  /*! \brief The let list. */
  std::unique_ptr<ExplicitLetList> ell_{nullptr};
  /*! \brief Mapping from a let-binding var to the let-binding vars that
   * def and and use this var. */
  StdMap<std::pair<VSet, VSet>> def_use_map_;
};

StdMap<std::pair<VSet, VSet>> ANFScheduler4Memory::BuildDefUseMap(const Function& func) {
  return DefUseAnalyzer(func).Run();
}

}  // namespace memory_schedule

TVM_REGISTER_PASS_CONFIG_OPTION("raf.memory_schedule", Bool);

Pass MemorySchedule() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    PassContext pass_ctx = PassContext::Current();
    bool enable = pass_ctx->GetConfig("raf.memory_schedule", Bool(false)).value();
    if (enable) {
      return Downcast<Function>(memory_schedule::ANFScheduler4Memory(f).Run());
    }
    return f;
  };

  Pass func_pass = CreateRAFFunctionPass(pass_func, 2, "MemoryScheduleHelper", {});
  PassInfo pass_info(2, "MemorySchedule", {});
  return RAFSequential({InferType(), func_pass}, pass_info);
}

RAF_REGISTER_GLOBAL("raf.pass_.MemorySchedule").set_body_typed(MemorySchedule);

}  // namespace pass
}  // namespace raf
