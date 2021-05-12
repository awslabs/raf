/*!
 * Copyright (c) 2020 by Contributors
 * \file memory_plan.cc
 * \brief Optimized allocated memory in the IR.
 */
#include <algorithm>
#include <random>
#include <vector>

#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/ir_ext.h"
#include "mnm/value.h"
#include "mnm/pass.h"
#include "./let_list.h"
#include "./liveness_analysis.h"
#include "tvm/relay/attrs/memory.h"

namespace mnm {
namespace pass {

namespace memory_plan {

using namespace mnm::ir;
using namespace mnm::value;

template <typename T>
using StdMap = std::unordered_map<Var, T, ObjectPtrHash, ObjectPtrEqual>;
using VSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

/*! \brief A tensor group. The intermediate tensors which liveness dummy tensors in a group
 * can be allocated to the same memory buffer.
 */
struct TensorGroup {
  TensorGroup(Var storage, int64_t alignment) : storage(storage), alignment(alignment) {
  }

  /*! \brief The map from dummy tensors created by livensss analysis to its let binding var
   * and required storage size.
   */
  StdMap<std::pair<Var, int64_t>> members;

  /*! \brief The binded variable for the allocated storage. */
  Var storage;

  /*! \brief The buffer size that should be allocated for this group. -1 means the size is
   * dynamic. In this case, this group should only have one tensor.
   */
  int64_t size = 0;

  /*! \brief The alignment of this group. */
  int64_t alignment;
};

/*! \brief A list of tensor groups with manipulation utilities. */
class TensorGroups {
 public:
  /*! \brief A list of storage allocation groups. */
  std::vector<TensorGroup> groups;
  /*! \brief Map from let var to the candidate group IDs. */
  StdMap<std::vector<int>> map_let_to_valid_group_ids;

  TensorGroups(liveness_analysis::LivenessAnalyzer* analyzer) : analyzer_(analyzer) {
  }

  /*! \brief Get the dummy output tensor in liveness analyzer. */
  Var GetTensorVar(const Var& let_var) {
    auto target_vars = analyzer_->GetTensorVars(let_var);
    CHECK_EQ(target_vars.size(), 1U);
    return target_vars[0];
  }

  /*! \brief Find the group ID that the target var belongs to, or -1 if not found. */
  int FindGroupIdByMember(const Var& let_var) {
    const Var target_var = GetTensorVar(let_var);
    for (size_t j = 0; j < groups.size(); ++j) {
      if (groups[j].members.find(target_var) != groups[j].members.end()) {
        return j;
      }
    }
    return -1;
  }

  /*! \brief Find the group ID that the target storage var belongs to, or -1 if not found. */
  int FindGroupIdByStorageVar(const Var& storage_var) {
    for (size_t j = 0; j < groups.size(); ++j) {
      if (groups[j].storage == storage_var) {
        return j;
      }
    }
    return -1;
  }

  /*! \brief Find valid tensor groups of the given tensor and its alignment. */
  std::vector<int> FindValidGroups(const Var& let_var, const int64_t alignment) {
    // If the let var has been processed before, simply return the cached results
    // because the available groups will not change at all.
    if (map_let_to_valid_group_ids.count(let_var) > 0) {
      return map_let_to_valid_group_ids[let_var];
    }

    // Live in vars.
    auto live_in_vars = analyzer_->GetLiveVars(let_var);

    // Find valid tensor groups.
    std::vector<int> candidates;
    for (size_t i = 0; i < groups.size(); ++i) {
      bool valid = true;
      for (const auto& member : groups[i].members) {
        if (groups[i].size == -1 || live_in_vars.count(member.first) > 0 ||
            groups[i].alignment != alignment) {
          // This group is invalid for this tensor to join if:
          // 1. Its size is dynamic,
          // 2. One of its members appears at the live-in set, or
          // 3. Its alignment is not the same as the current tensor.
          valid = false;
          break;
        }
      }
      if (valid) {
        candidates.push_back(i);
      }
    }

    map_let_to_valid_group_ids[let_var] = candidates;
    return candidates;
  }

  /*! \brief Join the given tensor group. */
  void JoinGroup(size_t group_id, const Var& let_var, int64_t size) {
    const Var target_var = GetTensorVar(let_var);
    groups[group_id].members[target_var] = std::make_pair(let_var, size);
    groups[group_id].size = (groups[group_id].size > size) ? groups[group_id].size : size;
    CHECK(map_let_to_valid_group_ids.count(let_var) > 0);
    auto it = std::find(map_let_to_valid_group_ids[let_var].begin(),
                        map_let_to_valid_group_ids[let_var].end(), group_id);
    if (it == map_let_to_valid_group_ids[let_var].end()) {
      map_let_to_valid_group_ids[let_var].push_back(group_id);
    }
  }

  /*! \brief Create a new group and return its ID. */
  int CreateGroup(const Var& storage_var, int64_t alignment) {
    groups.emplace_back(TensorGroup(storage_var, alignment));
    return groups.size() - 1;
  }

  /*! \brief Remove the tensor from the group, update the storage size, and return the size of
   * the removed tensor.
   */
  int64_t RemoveFromGroup(size_t group_id, const Var& let_var) {
    const Var target_var = GetTensorVar(let_var);
    CHECK_GT(groups[group_id].members.count(target_var), 0U);
    auto storage_nbytes = groups[group_id].members[target_var].second;
    if (groups[group_id].size == storage_nbytes) {
      // The storage size of this group may be reduced due to the removal of this tensor.
      int64_t max_size = 0;
      for (auto kv : groups[group_id].members) {
        if (kv.first == target_var) {
          continue;
        }
        max_size = (kv.second.second > max_size) ? kv.second.second : max_size;
      }
      groups[group_id].size = max_size;
    }
    groups[group_id].members.erase(target_var);
    return storage_nbytes;
  }

  /*! \brief Calculate the total memory footprint in MBs. */
  float GetTotalMemoryMBs() {
    float total = 0.0;
    for (size_t j = 0; j < groups.size(); ++j) {
      total += groups[j].size;
    }
    return total / 1024.0 / 1024.0;
  }

  /*!\ brief Log tensor groups with details for debugging. */
  std::string DebugDumpGroups() {
    std::stringstream ss1;
    for (const auto& group : groups) {
      std::stringstream ss2;
      for (const auto member : group.members) {
        ss2 << member.second.first->name_hint() << "(" << member.second.second << "), ";
      }
      ss1 << "Storage " << group.storage->name_hint() << ", size " << group.size
          << ", members: " << ss2.str() << std::endl;
    }
    return ss1.str();
  }

 private:
  /*! \brief The liveness analyzer, including liveness analysis results. */
  liveness_analysis::LivenessAnalyzer* analyzer_;
};

/*! \brief A mutator to perform the following tasks:
 * 1. Run tensor grouper to group the tensors generated by alloc_tensor according to
 *    the liveness analysis. All tensors in a tensor group will use the same storage.
 * 2. Mutate alloc_tensor to use storage of the group it belongs to.
 * 3. Mutate alloc_storage to allocate sufficient memory, which is required by
 *    the largest tensor in the group.
 * 4. Remove alloc_storages that do not be used by any group.
 */
class MemoryPlanner : public MixedModeMutator {
 public:
  MemoryPlanner(const Function& func, liveness_analysis::LivenessAnalyzer* analyzer)
      : func_(func), analyzer_(analyzer), tensor_groups_(Group()) {
  }

  Expr Run() {
    DLOG(INFO) << "Tensor groups:";
    DLOG(INFO) << tensor_groups_.DebugDumpGroups();

    // Keep storage var that is being used by one or more tensors.
    for (const auto& group : tensor_groups_.groups) {
      if (group.members.size() > 0) {
        keep_storages_.insert(group.storage);
      }
    }

    // Mutate the function.
    return this->Mutate(func_);
  }

  Expr VisitExpr_(const LetNode* node) final {
    auto pre_visit = [this](const LetNode* node) {
      curr_let_ = node->var;
      this->Mutate(node->var);
      this->Mutate(node->value);
    };

    auto post_visit = [this](const LetNode* node) {
      static const Op& alloc_storage_op = Op::Get("mnm.op.vm.alloc_storage");
      Var var = Downcast<Var>(this->Mutate(node->var));
      Expr value = this->Mutate(node->value);
      Expr body = this->Mutate(node->body);
      auto expr = GetRef<Expr>(node);

      // Determine whether the alloc_storage is unused and should be dismissed.
      bool dismiss_storage = false;
      if (const auto& node = value.as<CallNode>()) {
        const auto* op_node = node->op.as<OpNode>();
        if (op_node && GetRef<Op>(op_node) == alloc_storage_op &&
            keep_storages_.find(var) == keep_storages_.end()) {
          dismiss_storage = true;
        }
      }

      if (var.same_as(node->var) && value.same_as(node->value) && body.same_as(node->body)) {
        this->memo_[expr] = expr;
      } else if (dismiss_storage) {
        // Dismiss the expression.
        this->memo_[expr] = body;
      } else {
        this->memo_[expr] = Let(var, value, body);
      }
    };
    ExpandANormalForm(node, pre_visit, post_visit);
    return memo_[GetRef<Expr>(node)];
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    static const Op& alloc_storage_op = Op::Get("mnm.op.vm.alloc_storage");
    static const Op& alloc_tensor_op = Op::Get("mnm.op.vm.alloc_tensor");
    const auto* op_node = pre->op.as<OpNode>();

    auto call = Downcast<Call>(post);
    if (op_node && GetRef<Op>(op_node) == alloc_storage_op) {
      // Mutate the size of alloc_storage indicated by the tensor group.
      auto group_id = tensor_groups_.FindGroupIdByStorageVar(curr_let_);
      if (group_id != -1 && tensor_groups_.groups[group_id].size > 0) {
        Array<Expr> new_args = call->args;
        new_args.Set(0, MakeConstant(ScalarValue::make(tensor_groups_.groups[group_id].size)));
        return Call(alloc_storage_op, new_args);
      }
    } else if (op_node && GetRef<Op>(op_node) == alloc_tensor_op) {
      // Reassign alloc_tensor to the alloc_storage indicated by the tensor group.

      // Find the tensor group of this alloc_tensor.
      auto group_id = tensor_groups_.FindGroupIdByMember(curr_let_);

      // Check whether this tensor needs to be re-assigned to another storage.
      CHECK(group_id != -1) << "Internal error: output tensor of " << curr_let_->name_hint()
                            << " does not belong to any tensor group";
      auto storage_var = Downcast<Var>(call->args[0]);
      if (tensor_groups_.groups[group_id].storage != storage_var) {
        DLOG(INFO) << "Assign " << curr_let_->name_hint() << " to "
                   << tensor_groups_.groups[group_id].storage->name_hint() << " from "
                   << storage_var->name_hint();
        Array<Expr> new_args = call->args;
        new_args.Set(0, tensor_groups_.groups[group_id].storage);
        return Call(alloc_tensor_op, new_args);
      }
    }
    return post;
  }

 private:
  class TensorGrouper;

  TensorGroups Group();

 private:
  /*! \brief The functio to be optimized. */
  const Function& func_;
  /*! \brief The current processing let var. */
  Var curr_let_;
  /*! \brief The let list. */
  std::unique_ptr<ExplicitLetList> ell_{nullptr};
  /*! \brief The liveness analyzer, including liveness analysis results. */
  liveness_analysis::LivenessAnalyzer* analyzer_;
  /*! \brief A list of storage allocation groups. */
  TensorGroups tensor_groups_;
  /*! \brief A set of storage vars that will be preserved. */
  VSet keep_storages_;
};

/*! \brief A visitor to group tensors generated by alloc_tensor according to
 * the liveness analysis. A tensor can join an existing group (i.e., can reuse the storage
 * of the group) if the group:
 * 1) has no other tensors in the live-in set of the current tensor,
 * 2) has the same the alignment, and
 * 3) has the closest storage size as the current tensor.
 */
class MemoryPlanner::TensorGrouper : public ExprVisitor {
 public:
  TensorGrouper(const Expr& body, liveness_analysis::LivenessAnalyzer* analyzer)
      : analyzer_(analyzer), tensor_groups_(analyzer), ell_(ExplicitLetList::make(body)) {
    CHECK(analyzer_->IsSuccess());
  }

  TensorGroups Run() {
    const auto& vars = ell_->vars;
    const auto& exprs = ell_->exprs;
    CHECK_EQ(vars.size(), exprs.size());
    size_t n = exprs.size();

    for (size_t i = 0; i < n; ++i) {
      expr_map_[vars[i]] = exprs[i];
    }

    for (int i = 0; i < n; ++i) {
      curr_let_ = vars[i];
      ExprVisitor::VisitExpr(exprs[i]);
    }

    // TODO(comaniac): Add an iteratiive optimization algorithm.
    return tensor_groups_;
  }

  void VisitExpr_(const CallNode* node) override {
    static const Op& alloc_storage_op = Op::Get("mnm.op.vm.alloc_storage");
    static const Op& alloc_tensor_op = Op::Get("mnm.op.vm.alloc_tensor");
    const auto* op_node = node->op.as<OpNode>();

    // Only interest in alloc_tensor
    if (GetRef<Op>(op_node) == alloc_tensor_op) {
      for (auto& arg : node->args) {
        VisitExpr(arg);
      }

      // Get the storage size in bytes and alignment of this tensor.
      auto storage_var = Downcast<Var>(node->args[0]);
      auto storage_node = expr_map_[storage_var].as<CallNode>();
      CHECK(storage_node && Downcast<Op>(storage_node->op) == alloc_storage_op)
          << "Expected alloc_storage as the first arg of alloc_tensor, but got "
          << Downcast<Op>(storage_node->op)->name;

      int cand_group_id = -1;

      // Alignment
      CHECK(storage_node->args[1].as<ConstantNode>())
          << "The alignment of alloc_storage is not a constant";
      auto align_val = storage_node->args[1].as<ConstantNode>()->value;
      CHECK(align_val->IsInstance<IntValueObj>());
      int64_t alignment = align_val.as<IntValueObj>()->value;

      // Use tensor size and alignment to determine whether it can join an existing tensor gruop.
      int64_t storage_nbytes = -1;
      if (storage_node->args[0].as<ConstantNode>()) {
        // Storage size
        auto size_val = storage_node->args[0].as<ConstantNode>()->value;
        CHECK(size_val->IsInstance<IntValueObj>());
        storage_nbytes = size_val.as<IntValueObj>()->value;

        // Find tensor group candidates to join.
        auto candidates = tensor_groups_.FindValidGroups(curr_let_, alignment);

        // Select the one with the closest storage size.
        auto curr_dis = INT64_MAX;
        for (auto group_id : candidates) {
          auto new_dis = abs(storage_nbytes - tensor_groups_.groups[group_id].size);
          if (new_dis <= curr_dis) {
            cand_group_id = group_id;
            curr_dis = new_dis;
          }
        }
      }

      if (cand_group_id == -1) {
        // Cannot allocate this tensor to any of existing groups so create a new one.
        cand_group_id = tensor_groups_.CreateGroup(storage_var, alignment);
        DLOG(INFO) << "Create a new group " << cand_group_id << " for " << storage_var->name_hint();
      }
      CHECK(cand_group_id != -1);

      // Allocate this tensor to an existing group and update the storage size if needed.
      DLOG(INFO) << curr_let_->name_hint() << " joins group " << cand_group_id;
      tensor_groups_.JoinGroup(cand_group_id, curr_let_, storage_nbytes);
    } else {
      ExprVisitor::VisitExpr_(node);
    }
  }

  /*! \brief The current processing let var. */
  Var curr_let_;
  /*! \brief The let list. */
  std::unique_ptr<ExplicitLetList> ell_{nullptr};
  /*! \brief A map from let varr to its expression. */
  std::unordered_map<Var, Expr, ObjectHash, ObjectEqual> expr_map_;
  /*! \brief The liveness analyzer, including liveness analysis results. */
  liveness_analysis::LivenessAnalyzer* analyzer_;
  /*! \brief A list of storage allocation groups. */
  TensorGroups tensor_groups_;
};

TensorGroups MemoryPlanner::Group() {
  return TensorGrouper(func_, analyzer_).Run();
}

}  // namespace memory_plan

Pass MemoryPlan() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto analyzer = liveness_analysis::LivenessAnalyzer(f);
        analyzer.Run();
        if (!analyzer.IsSuccess()) {
          LOG(WARNING) << "Memory planning is disabled because Liveness analysis was failed";
          return f;
        }
        return Downcast<ir::Function>(memory_plan::MemoryPlanner(f, &analyzer).Run());
      };
  auto func_pass = CreateMNMFunctionPass(pass_func, 3, "MemoryPlan", {});
  PassInfo pass_info(3, "MemoryPlan", {});
  return MNMSequential({InferType(), func_pass}, pass_info);
}

MNM_REGISTER_GLOBAL("mnm.pass_.MemoryPlan").set_body_typed(MemoryPlan);

}  // namespace pass
}  // namespace mnm
