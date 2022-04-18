/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file memory_plan.cc
 * \brief Optimized allocated memory in the IR.
 */
#include <algorithm>
#include <random>
#include <vector>

#include "raf/op.h"
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/value.h"
#include "raf/pass.h"
#include "./let_list.h"
#include "./liveness_analysis.h"
#include "tvm/relay/attrs/memory.h"

namespace raf {
namespace pass {

namespace memory_plan {

using namespace raf::ir;
using namespace raf::value;

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

  TensorGroups(liveness_analysis::LivenessAnalyzer* analyzer) : analyzer_(analyzer) {
    dummy_out_vars_ = analyzer_->GetOutputTensorVars();
  }

  /*! \brief Get the dummy output tensor in liveness analyzer. */
  Var GetTensorVar(const Var& let_var) {
    auto target_vars = analyzer_->GetTensorVars(let_var);
    CHECK_EQ(target_vars.size(), 1U);
    return target_vars[0];
  }

  /*! \brief Check whether the tensor is a final output. */
  bool IsOutputTensor(const Var& tensor_var) {
    auto target_var = GetTensorVar(tensor_var);
    return dummy_out_vars_.find(target_var) != dummy_out_vars_.end();
  }

  /*! \brief Check whether the group has a final output tensor. */
  bool HasOutputTensor(size_t group_id) {
    for (const auto& member : groups[group_id].members) {
      if (IsOutputTensor(member.second.first)) {
        return true;
      }
    }
    return false;
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
    return candidates;
  }

  /*! \brief Join the given tensor group. */
  void JoinGroup(size_t group_id, const Var& let_var, int64_t size = 0) {
    const Var target_var = GetTensorVar(let_var);
    groups[group_id].members[target_var] = std::make_pair(let_var, size);
    groups[group_id].size = (groups[group_id].size > size) ? groups[group_id].size : size;
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
  /*! \brief Dummy vars that output tensors map to. */
  VSet dummy_out_vars_;
};

/*! \brief A mutator to perform the following tasks:
 * 1. Run tensor grouper to group the tensors generated by alloc_tensor according to
 *    the liveness analysis. All tensors in a tensor group will use the same storage.
 * 2. Mutate alloc_tensor to use storage of the group it belongs to.
 * 3. Mutate alloc_storage to allocate sufficient memory, which is required by
 *    the largest tensor in the group.
 * 4. Remove alloc_storages that do not be used by any group.
 * 5. Insert free(%x) to free tensor/storage %x at the end of its life-cycle.
 */
class MemoryPlanner : public ExprMutator {
 public:
  MemoryPlanner(const Function& func, liveness_analysis::LivenessAnalyzer* analyzer)
      : func_(func), analyzer_(analyzer), tensor_groups_(Group()) {
    scopes_.emplace_back(new LetList);
  }

  Expr Run() {
    DLOG(INFO) << "Tensor groups:";
    DLOG(INFO) << tensor_groups_.DebugDumpGroups();

    // List storage vars that will be used by one or more tensors.
    for (const auto& group : tensor_groups_.groups) {
      if (group.members.size() > 0) {
        used_storages_.insert(group.storage);
      }
    }

    // Mutate the function.
    return this->Mutate(func_);
  }

  Expr VisitExpr_(const LetNode* node) {
    static const Op& alloc_storage_op = Op::Get("raf.op.vm.alloc_storage");
    scopes_.emplace_back(new LetList);
    auto scope = scopes_.back().get();
    Expr body;
    do {
      curr_let_ = node->var;

      // Free allocated tensors that will not be used anymore.
      auto live_in_vars = analyzer_->GetLiveVars(curr_let_);
      auto it = live_tensors_.begin();
      while (it != live_tensors_.end()) {
        const Var target_var = tensor_groups_.GetTensorVar(*it);
        if (live_in_vars.find(target_var) == live_in_vars.end()) {
          auto group_id = tensor_groups_.FindGroupIdByMember(*it);
          // If a tensor does not belong to any group, meaning that we its storage is not
          // allocated by us (i.e., parameters), so simply remove it from the live set.
          if (group_id != -1) {
            // Remove the freed tensor from the tensor group when its life is eneded.
            // Note that we do not need to insert vm.free for tensors as only the storage
            // should hold the memory pointer.
            tensor_groups_.RemoveFromGroup(group_id, *it);

            // Free allocated storages that will not be used anymore.
            auto group = tensor_groups_.groups[group_id];
            if (group.members.size() == 0) {
              scope->Push(MakeFreeMemory(group.storage));
            }
          }
          it = live_tensors_.erase(it);
        } else {
          it++;
        }
      }

      auto value = VisitExpr(node->value);

      // Determine whether the alloc_storage is never used and should be dismissed.
      bool dismiss_storage = false;
      if (const auto& node = value.as<CallNode>()) {
        const auto* op_node = node->op.as<OpNode>();
        if (op_node && GetRef<Op>(op_node) == alloc_storage_op &&
            used_storages_.find(curr_let_) == used_storages_.end()) {
          dismiss_storage = true;
        }
      }
      if (!dismiss_storage) {
        expr_map_.emplace(node->var, value);
        scope->Push(node->var, value);
      }

      body = node->body;
      node = body.as<LetNode>();
    } while (node);
    auto new_body = VisitExpr(body);
    auto ret = scopes_.back()->Get(new_body);
    scopes_.pop_back();
    return ret;
  }

  Expr VisitExpr_(const CallNode* node) final {
    static const Op& alloc_storage_op = Op::Get("raf.op.vm.alloc_storage");
    static const Op& alloc_tensor_op = Op::Get("raf.op.vm.alloc_tensor");
    static const Op& reshape_tensor_op = Op::Get("raf.op.vm.set_shape");
    const auto* op_node = node->op.as<OpNode>();

    auto call = GetRef<Call>(node);
    if (op_node && GetRef<Op>(op_node) == alloc_storage_op) {
      // Mutate the size of alloc_storage indicated by the tensor group.

      auto group_id = tensor_groups_.FindGroupIdByStorageVar(curr_let_);
      if (group_id != -1 && tensor_groups_.groups[group_id].size > 0) {
        bool alloc_async = !tensor_groups_.HasOutputTensor(group_id);

        Array<Expr> new_args;
        for (auto& arg : call->args) {
          new_args.push_back(VisitExpr(arg));
        }
        new_args.Set(0, MakeConstant(ScalarValue::make(tensor_groups_.groups[group_id].size)));
        if (new_args.size() == 5) {
          new_args.push_back(MakeConstant(BoolValue::make(alloc_async)));
        } else {
          CHECK_EQ(new_args.size(), 6U);
          new_args.Set(5, MakeConstant(BoolValue::make(alloc_async)));
        }
        return Call(alloc_storage_op, new_args);
      }
    } else if (op_node && GetRef<Op>(op_node) == alloc_tensor_op) {
      // Reassign alloc_tensor to the alloc_storage indicated by the tensor group.
      live_tensors_.insert(curr_let_);

      Array<Expr> new_args;
      for (auto& arg : call->args) {
        new_args.push_back(VisitExpr(arg));
      }

      // Find the tensor group of this alloc_tensor and check whether this tensor needs
      // to be re-assigned to another storage.
      auto group_id = tensor_groups_.FindGroupIdByMember(curr_let_);
      CHECK_NE(group_id, -1) << "Internal error: output tensor of " << curr_let_->name_hint()
                             << " does not belong to any tensor group";
      auto storage_var = Downcast<Var>(call->args[0]);
      if (tensor_groups_.groups[group_id].storage != storage_var) {
        DLOG(INFO) << "Assign " << curr_let_->name_hint() << " to "
                   << tensor_groups_.groups[group_id].storage->name_hint() << " from "
                   << storage_var->name_hint();
        new_args.Set(0, tensor_groups_.groups[group_id].storage);
      }

      // Override the own memory flag argument.
      auto own = MakeConstant(ScalarValue::make(tensor_groups_.IsOutputTensor(curr_let_)));
      if (call->args.size() == 4) {
        new_args.push_back(own);
      } else {
        new_args.Set(4, own);
      }

      return Call(alloc_tensor_op, new_args);
    } else if (op_node && GetRef<Op>(op_node) == reshape_tensor_op) {
      // Other ops that will also create a new tensor/view. We do not need to mutate them,
      // but have to trace their life-cycle to know the right place of inserting free storage.
      live_tensors_.insert(curr_let_);
    }
    return ExprMutator::VisitExpr_(node);
  }

 private:
  class TensorGrouper;

  TensorGroups Group();

  inline Expr MakeFreeMemory(const Var& memory_var) {
    static const Op& op = Op::Get("raf.op.vm.free");
    return Call(op, {memory_var});
  }

 private:
  /*! \brief The scope stack of the let list. */
  std::vector<std::unique_ptr<LetList>> scopes_;
  /*! \brief The functio to be optimized. */
  const Function& func_;
  /*! \brief The current processing let var. */
  Var curr_let_;
  /*! \brief A map from let varr to its expression. */
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> expr_map_;
  /*! \brief The liveness analyzer, including liveness analysis results. */
  liveness_analysis::LivenessAnalyzer* analyzer_;
  /*! \brief A list of storage allocation groups. */
  TensorGroups tensor_groups_;
  /*! \brief Used storage vars. */
  VSet used_storages_;
  /*! \brief Current live tensor vars. */
  VSet live_tensors_;
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

    return tensor_groups_;
  }

  void VisitExpr_(const CallNode* node) override {
    static const Op& alloc_storage_op = Op::Get("raf.op.vm.alloc_storage");
    static const Op& alloc_tensor_op = Op::Get("raf.op.vm.alloc_tensor");
    static const Op& reshape_tensor_op = Op::Get("raf.op.vm.set_shape");
    const auto* op_node = node->op.as<OpNode>();

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

      // Alignment.
      CHECK(storage_node->args[1].as<ConstantNode>())
          << "The alignment of alloc_storage is not a constant";
      auto align_val = storage_node->args[1].as<ConstantNode>()->value;
      CHECK(align_val->IsInstance<IntValueObj>());
      int64_t alignment = align_val.as<IntValueObj>()->value;

      // Storage size.
      int64_t storage_nbytes = -1;
      if (storage_node->args[0].as<ConstantNode>()) {
        auto size_val = storage_node->args[0].as<ConstantNode>()->value;
        CHECK(size_val->IsInstance<IntValueObj>());
        storage_nbytes = size_val.as<IntValueObj>()->value;
      }

      // Create a new tensor group.
      auto cand_group_id = tensor_groups_.CreateGroup(storage_var, alignment);
      DLOG(INFO) << "Create a new group " << cand_group_id << " for " << storage_var->name_hint();
      tensor_groups_.JoinGroup(cand_group_id, curr_let_, storage_nbytes);
    } else if (GetRef<Op>(op_node) == reshape_tensor_op) {
      // set_shape creates a new tensor view so it has to be considered as a new tensor too.
      for (auto& arg : node->args) {
        VisitExpr(arg);
      }
      auto tensor_var = Downcast<Var>(node->args[0]);
      auto group_id = tensor_groups_.FindGroupIdByMember(tensor_var);

      // If its original tensor is not in a tensor group, it means the tensor is a parameter,
      // which is not allocated by us so we do not need to track it.
      if (group_id != -1) {
        // Put this tensor to the samee group as its original tensor.
        DLOG(INFO) << curr_let_->name_hint() << " joins group " << group_id;
        tensor_groups_.JoinGroup(group_id, curr_let_);
      }
    } else {
      ExprVisitor::VisitExpr_(node);
    }
  }

  /*! \brief The current processing let var. */
  Var curr_let_;
  /*! \brief The let list. */
  std::unique_ptr<ExplicitLetList> ell_{nullptr};
  /*! \brief A map from let varr to its expression. */
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> expr_map_;
  /*! \brief The liveness analyzer, including liveness analysis results. */
  liveness_analysis::LivenessAnalyzer* analyzer_;
  /*! \brief A list of storage allocation groups. */
  TensorGroups tensor_groups_;
};

TensorGroups MemoryPlanner::Group() {
  return TensorGrouper(func_, analyzer_).Run();
}

}  // namespace memory_plan

TVM_REGISTER_PASS_CONFIG_OPTION("raf.memory_plan.dump_liveness_stat", Bool);

Pass MemoryPlan() {
  PassContext pass_ctx = PassContext::Current();
  Bool dump_stat = pass_ctx->GetConfig("raf.memory_plan.dump_liveness_stat", Bool(false)).value();
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    auto func = f;
    try {
      func = Downcast<Function>(pass::InferType(f));
    } catch (const dmlc::Error& e) {
      LOG(WARNING) << "Memory planning is disabled because infer type was failed";
      return f;
    }

    auto analyzer = liveness_analysis::LivenessAnalyzer(func);
    try {
      auto live_in = analyzer.Run();
      if (!analyzer.IsSuccess()) {
        throw;
      }
      if (dump_stat) {
        liveness_analysis::DumpLivenessStat(live_in);
      }
    } catch (const dmlc::Error& e) {
      LOG(WARNING) << "Memory planning is disabled because liveness analysis was failed";
      return func;
    }
    return Downcast<ir::Function>(memory_plan::MemoryPlanner(func, &analyzer).Run());
  };
  return CreateRAFFunctionPass(pass_func, 2, "MemoryPlan", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.MemoryPlan").set_body_typed(MemoryPlan);

}  // namespace pass
}  // namespace raf
