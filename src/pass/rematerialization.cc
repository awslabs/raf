/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file rematerialization.cc
 * \brief Perform rematerialization to reduce peak memory footrpint.
 */
#include <tvm/ir/type_functor.h>
#include "raf/op.h"
#include "raf/ir_ext.h"
#include "raf/pass.h"
#include "raf/op_profiler.h"
#include "./common.h"
#include "./estimate_flops.h"
#include "./let_list.h"
#include "./liveness_analysis.h"
#include "../op/dialect/tvm/tvm_utils.h"

namespace raf {
namespace pass {
namespace rematerialization {

using namespace raf::op;

template <typename T>
using StdMap = std::unordered_map<Var, T, ObjectPtrHash, ObjectPtrEqual>;
using VSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

constexpr float kMegaBytes = 1048576;
constexpr float kGigaBytes = 1073741824;

// Whether to display verbose logging.
#define SHOW_VERBOSE_LOG 0

// Whether to update tensor index when rematerialization. If defined, then
// the rematerialized tensors are less likely to be freed and rematerialized again.
#define UPDATE_TENSOR_INDEX

// The max number of ops is allowed to rematerialized a tensor.
#define MAX_REMAT_DEPTH 10

#if SHOW_VERBOSE_LOG == 1
#define VERBOSE_LOG LOG(INFO)
#else
#define VERBOSE_LOG DLOG(INFO)
#endif

/*! \brief A data structure of required information for a tensor. */
struct TensorInfo {
 public:
  TensorInfo(size_t idx, const Var& let_var, const Var& liveness_var, int64_t size,
             float compute_cost, int64_t workspace_size, size_t tuple_field_idx = -1)
      : index(idx),
        let_var(let_var),
        liveness_var(liveness_var),
        size(size),
        compute_cost(compute_cost),
        workspace_size(workspace_size),
        tuple_field_idx(tuple_field_idx) {
  }

  /*! \brief The index of this tensor, which indicates the created order. If this tensor is
   * rematerialized, than the index may be updated, depending on UPDATE_TENSOR_INDEX. */
  size_t index;
  /*! \brief The latest let binding var of this tensor. If this tensor is marked as dead and
   * rematerialized during the pass, then this var will be updated to the rematerialized one. */
  Var let_var;
  /*! \brief The size of this tensor in bytes. 0 means this tensor is dynamic. */
  int64_t size;
  /*! \brief The tuple field index of this tensor. -1 if this tensor does not belong to a tuple. */
  size_t tuple_field_idx = -1;
  /*! \brief The corresponding var used by liveness analyzer. Liveness var will not be change
   * even this tensor is rematerialized, because this var is only used as a symbol in live-in
   * set to indicate whether this tensor will be used in the future. */
  Var liveness_var;
  /*! \brief Whether this tensor is an input parameter. */
  bool is_param = false;
  /*! \brief A set of liveness vars that this tensor shares the storage with.
   * If this set is not empty, then this tensor cannot be remeterialized. */
  VSet share_storage;
  /*! \brief Computation cost of this tensor. -1 means recomputing this tensor is invalid. */
  float compute_cost = -1;
  /*! \brief Workspace memory size of this tensor in bytes. -1 means recomputing this tensor is
   * invalid. */
  int64_t workspace_size = -1;
  /*! \brief Only TensorInfos can change this status since TensorInfos has to maintain the live
   * tensor list. */
  bool IsDead() {
    return is_dead_;
  }
  // Function interface to protect use count from being arbitrarily set.
  // Can also perform some checking if necessary.

  /*! \brief Get the current use count of this tensor info. */
  inline int64_t GetUseCount() {
    return use_count_;
  }

  /*! \brief Set the use count of this tensor info. The input must be non-negative. */
  inline int64_t SetUseCount(int64_t value) {
    CHECK_GE(value, 0U) << "Use count of a tensor must be non-negative!";
    use_count_ = value;
    return use_count_;
  }

  /*! \brief Increment the use count of this tensor info by 1. */
  inline int64_t IncUseCount() {
    use_count_ += 1;
    return use_count_;
  }

  /*! \brief Decrement the use count of this tensor info by 1. */
  inline int64_t DecUseCount() {
    use_count_ -= 1;
    return use_count_;
  }

 private:
  /*! \brief Whether this tensor has been marked as dead and should be rematerialized later. */
  bool is_dead_ = false;
  /*! \brief The use count of this tensor. */
  int64_t use_count_ = 0;

  friend class TensorInfos;
};

/*! \brief A set of tensor infos with utilities. */
class TensorInfos {
 public:
  void CreateTensorInfo(const Var& let_var, const Array<Var>& liveness_vars, bool is_param = false,
                        float compute_cost = -1, int64_t workspace_size = -1) {
    auto sizes = liveness_analysis::CalcBytesCompactSizes(let_var->checked_type());
    CHECK_EQ(liveness_vars.size(), sizes.size());
    const auto* extended_var = static_cast<const ExtendedVarNode*>(let_var.operator->());

    size_t n_tensors = liveness_vars.size();
    bool is_tuple = n_tensors > 1;
    for (size_t i = 0; i < n_tensors; ++i) {
      auto liveness_var = liveness_vars[i];
      auto size = sizes[i];

      // If the tensor info has been created, then this let-binding var does not create
      // a new tensor (e.g., let %a = %b;).
      if (liveness_var_to_info_.count(liveness_var) == 0) {
        liveness_var_to_info_[liveness_var] =
            std::make_shared<TensorInfo>(tensor_idx_++, let_var, liveness_var, size, compute_cost,
                                         workspace_size, (is_tuple) ? i : -1);
      }
      auto tensor_info = liveness_var_to_info_[liveness_var];
      tensor_info->is_param = is_param;
      let_var_to_infos_[let_var].push_back(tensor_info);

      // If two vars are shared, then both of them cannot be rematerialized.
      if (extended_var->may_share.defined()) {
        auto shared_tensor_infos = GetTensorInfoFromLetVar(extended_var->may_share);
        CHECK(liveness_vars.size() == 1 && shared_tensor_infos.size() == 1)
            << "Does not support tuple " << let_var << " shares with " << extended_var->may_share;
        tensor_info->share_storage.insert(shared_tensor_infos[0]->liveness_var);
        shared_tensor_infos[0]->share_storage.insert(liveness_var);
      }
    }
  }

  /*! \brief Update the let-binding var to be the latest rematerialized one, and add the new
   * let-binding var to the map. */
  void UpdateLetVar(const Var& new_var, std::vector<std::shared_ptr<TensorInfo>>& tensor_infos) {
    for (auto tensor_info : tensor_infos) {
      tensor_info->let_var = new_var;
      MarkAsLive(tensor_info);
#ifdef UPDATE_TENSOR_INDEX
      tensor_info->index = tensor_idx_++;
#endif
    }
    let_var_to_infos_[new_var] = tensor_infos;
  }

  /*! \brief Add a new entry that maps a let-binding var to existing tensor infos. The alias var
   * is usually reshape ops. */
  void AddLetVar(const Var& alias_var,
                 const std::vector<std::shared_ptr<TensorInfo>>& tensor_infos) {
    let_var_to_infos_[alias_var] = tensor_infos;
  }

  /*! \brief Get a set of tensor infos of the currently live tensors. */
  std::unordered_set<std::shared_ptr<TensorInfo>> GetLiveTensorInfos() {
    return live_tensors_;
  }

  void MarkAsLive(std::shared_ptr<TensorInfo> tensor_info) {
    tensor_info->is_dead_ = false;
    live_tensors_.insert(tensor_info);
  }

  void MarkAsDead(std::shared_ptr<TensorInfo> tensor_info) {
    tensor_info->is_dead_ = true;
    live_tensors_.erase(tensor_info);
  }

  /*! \brief Get the tensor info given a let binding var. */
  std::vector<std::shared_ptr<TensorInfo>> GetTensorInfoFromLetVar(const Var& var) {
    CHECK_GT(let_var_to_infos_.count(var), 0U) << "Cannot find the tensor info of given " << var;
    return let_var_to_infos_[var];
  }

  /*! \brief Get the tesnor info given a liveness var. */
  std::shared_ptr<TensorInfo> GetTensorInfoFromLivenessVar(const Var& var) {
    CHECK_GT(liveness_var_to_info_.count(var), 0) << "Cannot find the tensor info of given " << var;
    return liveness_var_to_info_[var];
  }

  /*! \brief Dump all tensor infos for debugging. */
  std::string DebugDump() {
    std::ostringstream os;
    os << "If a tensor is binded multiple times (e.g., let b = a;), "
       << "then only the first let_var will be displayed" << std::endl;
    for (auto pair : liveness_var_to_info_) {
      auto tensor_info = pair.second;
      os << "LivenessVar: " << tensor_info->liveness_var->name_hint()
         << ((tensor_info->is_param) ? " (param)" : "")
         << ((!tensor_info->share_storage.empty()) ? " (shared)" : "")
         << ((tensor_info->tuple_field_idx != -1)
                 ? " (tuple." + std::to_string(tensor_info->tuple_field_idx) + ")"
                 : "")
         << ", GFLOPS: " << tensor_info->compute_cost
         << ", LetVar: " << tensor_info->let_var->name_hint()
         << ", Size(MBs): " << tensor_info->size / 1048576.0
         << ", Workspace Size(MBs): " << tensor_info->workspace_size / 1048576.0
         << ", UseCount: " << tensor_info->GetUseCount() << std::endl;
    }
    return os.str();
  }

  /*! \brief Verify that all use counts are maintained properly at the end. All tensors should have
   *  use counts of zero when this pass finishes.
   */
  void VerifyUseCountTracking() {
    LOG(INFO) << "Verifying the correctness of use count tracking... ";
    std::stringstream ss;
    for (auto pair : liveness_var_to_info_) {
      auto liveness_var = pair.first;
      auto tensor_info = pair.second;
      int64_t remaining_use_count = tensor_info->GetUseCount();
      if (remaining_use_count) {
        ss << "Tensor " << tensor_info->let_var->name_hint() << "(" << liveness_var->name_hint()
           << ") has a remaining use count of " << remaining_use_count << " at the end of the pass!"
           << std::endl;
      }
    }
    if (ss.rdbuf()->in_avail()) {
      LOG(FATAL) << "InternalError: One or more tensors have incorrect use count: " << std::endl
                 << ss.str();
      throw;
    }
  }

 private:
  /*! \brief The mapping from liveness vars to the analyzed tensor info. */
  StdMap<std::shared_ptr<TensorInfo>> liveness_var_to_info_;
  /*! \brief A reversed map of liveness vars to tensor info. */
  StdMap<std::vector<std::shared_ptr<TensorInfo>>> let_var_to_infos_;
  /*! \brief The current live tensors. */
  std::unordered_set<std::shared_ptr<TensorInfo>> live_tensors_;
  /*! \brief The current tensor index. */
  size_t tensor_idx_ = 0;
};

/*!
 * \brief Perform rematerialization algorithm to reduce the peak memory footprint. The algorithm
 * is briefly described as follows:
 * 1. Traverse the ANF graph to obtain the use count of each tensor.
 * 2. Traverse the ANF graph again in the execution order along with a static memory tracing
 *    based on liveness analysis.
 * 3. For each call node that produces a new tensor,
 *    3.1. Update the current memory consumption by adding new allocated tensor and releasing
 *         end-of-life tensors.
 *    3.2. Rematerialize the arguments that were marked as dead previouly and update the current
 *         memory consumption.
 *    3.3. If the total memory consumption exceeds the given budget, estimate the rematerialization
 *         cost of each live-in tensors, excluding input parameters, call node arguments, and
 *         tensors that are just rematerialized.
 *    3.4. Mark the tensor with the lowest cost as dead, meaning that later call nodes that use this
 *         tensor need to rematerialize it. The use counts of this tensor's arguments will be
 *         incremented so that the rematerialization cost estimator is aware of the additional uses.
 *    3.5. Repeat 3.3 - 3.4 until the total memory consumption is lower than the budget. If the
 *         memory still exceeds the budget but no more tensors can be marked as dead, then error out
 *         to let users adjust the budget.
 * Assumptions:
 * 1. Memory plan will be applied later to insert "free" properly to reflect the rematerialization.
 *    If memory plan is not applied, then rematerialization simply brings latency overheads.
 * 2. The output ANF will not be transformed to DF or BBNF. Otherwise, it is not guaranteed that
 *    the cloned call nodes for rematerializing tensors are executed as late as possible.
 */
class Rematerializer : public ExprMutator {
 public:
  explicit Rematerializer(liveness_analysis::LivenessAnalyzer* analyzer, const Device& device,
                          const Function& func, const IRModule& mod, const int64_t budget,
                          op_profiler::OpProfiler* profiler)
      : analyzer_(analyzer),
        func_(func),
        budget_(budget),
        profiler_(profiler),
        tensor_infos_(AnalyzeTensors(device, func, mod, analyzer, profiler)) {
    scopes_.emplace_back(new LetList);
    VERBOSE_LOG << "Tensor infos:\n" << tensor_infos_.DebugDump();
  }

  /*! \brief A debug function to dump a set of vars. */
  std::string DebugDump(VSet vset) {
    std::ostringstream os;
    for (const auto& v : vset) {
      os << v << ", ";
    }
    return os.str();
  }

  /*! \brief A debug function to dump a vector of rematerialization candidates. */
  std::string DebugDumpCandidates(
      std::vector<std::pair<std::shared_ptr<TensorInfo>, float>>& data) {
    std::ostringstream os;
    for (const auto& pair : data) {
      auto tensor_info = pair.first;
      os << "(" << tensor_info->liveness_var << ", " << pair.second << "); ";
    }
    return os.str();
  }

  /*! \brief Run the rematerialization algorithm and return the mutated function. */
  Expr Run() {
    // Initialize memory trace with parameter sizes.
    for (const auto& var : func_->params) {
      for (auto tensor_info : tensor_infos_.GetTensorInfoFromLetVar(var)) {
        // A parameter may have multiple tensor infos if it is a tuple.
        curr_mem_trace_ += tensor_info->size;
      }
    }
    auto ret = this->Mutate(func_);
    std::stringstream ss;
    ss << "Estimated peak memory after rematerialization is " << peak_memory_ / kMegaBytes
       << " MBs; while the budget is " << budget_ / kMegaBytes << " MBs. " << n_recompute_ops_
       << " more ops were inserted";
    if (profiler_) {
      ss << " with " << std::setw(2) << (total_recompute_cost_ / 1000.0) << " ms latency overhead";
    }
    LOG(INFO) << ss.str();
    return ret;
  }

  Expr VisitExpr_(const LetNode* node) {
    scopes_.emplace_back(new LetList);
    auto scope = scopes_.back().get();

    Expr body;
    do {
      curr_let_ = node->var;

      // Update memory consumption. Note that in-place tensors will not allocate a new buffer.
      auto tensor_infos = tensor_infos_.GetTensorInfoFromLetVar(curr_let_);
      const auto* extended_var = static_cast<const ExtendedVarNode*>(curr_let_.operator->());

      VERBOSE_LOG << "[@" << curr_let_->name_hint() << "]";
      if (!extended_var->may_share.defined()) {
        for (auto tensor_info : tensor_infos) {
          if (tensor_info->let_var.same_as(curr_let_)) {
            // If the let_var in the tensor info is not the same as the current let_var,
            // then the current let_var binds to an existing tensor (e.g., let %b = %a),
            // so we do not need to allocate it again.
            tensor_infos_.MarkAsLive(tensor_info);
            curr_mem_trace_ += tensor_info->size;
            // Also need to add in the workspace size
            curr_mem_trace_ += tensor_info->workspace_size;
            VERBOSE_LOG << "|-Alloc: " << tensor_info->liveness_var->name_hint() << " for "
                        << (tensor_info->size + tensor_info->workspace_size) / kMegaBytes << " MBs";
          }
        }
      } else {
        auto shared_tensor_infos = tensor_infos_.GetTensorInfoFromLetVar(extended_var->may_share);
        CHECK_EQ(shared_tensor_infos.size(), 1U);
        tensor_infos_.MarkAsLive(shared_tensor_infos[0]);
        VERBOSE_LOG << "|-Inplace update with "
                    << shared_tensor_infos[0]->liveness_var->name_hint();
      }

      auto new_value = VisitExpr(node->value);

      // Remove the workspace size from the current memory trace after visiting
      // this let node, since they will be freed immediately after execution
      if (!extended_var->may_share.defined()) {
        for (auto tensor_info : tensor_infos) {
          if (tensor_info->let_var.same_as(curr_let_)) {
            curr_mem_trace_ -= tensor_info->workspace_size;
          }
        }
      }

      // Proceed to the next node
      scope->Push(node->var, new_value);
      let_vars_.emplace(node->var, new_value);
      body = node->body;
      node = body.as<LetNode>();
    } while (node);

    auto new_body = VisitExpr(body);
    auto ret = scopes_.back()->Get(new_body);
    scopes_.pop_back();
    return ret;
  }

  Expr VisitExpr_(const CallNode* node) {
    auto scope = scopes_.back().get();
    auto live_in_vars = analyzer_->GetLiveVars(curr_let_);

    // Release end of life tensors. Note that input parameters cannot be released.
    if (!curr_live_in_vars_.empty()) {
      VSet vars_to_remove;
      for (auto liveness_var : curr_live_in_vars_) {
        auto tensor_info = tensor_infos_.GetTensorInfoFromLivenessVar(liveness_var);
        // Liveness analysis is unaware of our rematerialization decision before we actually
        // rematerialize the tensor. The use_count is maintained by this pass. A tensor
        // actually reaches its end of life only when its use count decreases to 0.
        if ((!tensor_info->is_param) && (live_in_vars.find(liveness_var) == live_in_vars.end()) &&
            (tensor_info->GetUseCount() <= 0) && (!tensor_info->IsDead())) {
          tensor_infos_.MarkAsDead(tensor_info);
          vars_to_remove.insert(liveness_var);

          // If this tensor shares storage with other tensors which are still alive, then
          // we mark this tensor as dead without removing its size from current memory usage.
          bool should_free = true;
          for (auto share_liveness_var : tensor_info->share_storage) {
            if (!tensor_infos_.GetTensorInfoFromLivenessVar(share_liveness_var)->IsDead()) {
              should_free = false;
              break;
            }
          }
          if (should_free) {
            curr_mem_trace_ -= tensor_info->size;
            VERBOSE_LOG << "|-Free: " << tensor_info->liveness_var->name_hint() << " for "
                        << tensor_info->size / kMegaBytes << " MBs";
          }
        }
      }
      // Remove the dead tensors from the current live set
      for (auto dead_var : vars_to_remove) {
        curr_live_in_vars_.erase(dead_var);
      }
    }
    // Add the new live vars from liveness analysis into the current live set
    for (auto live_var : live_in_vars) {
      curr_live_in_vars_.insert(live_var);
    }

    // Process arguments and rematerialize mark-as-dead tensors.
    newly_remat_tensors_.clear();
    Array<Expr> new_args;
    for (auto arg : node->args) {
      if (auto var_node = arg.as<VarNode>()) {
        auto arg_var = GetRef<Var>(var_node);

        // Remove the current node from the use count of its arguments.
        for (auto tensor_info : tensor_infos_.GetTensorInfoFromLetVar(arg_var)) {
          tensor_info->DecUseCount();
        }

        // Remeterialize mark-as-dead arguments.
        auto remat_var = Rematerialize(scope, arg_var);
        new_args.push_back(remat_var);
      } else {
        new_args.push_back(arg);
      }
    }
    VERBOSE_LOG << "|-CurrMem: " << curr_mem_trace_ / kMegaBytes << " MBs";

    // Need to kill other tensors to fit the given budget.
    if (curr_mem_trace_ > budget_) {
      // Find candidates to be rematerialized from the live tensors.
      std::vector<std::pair<std::shared_ptr<TensorInfo>, float>> candidate_n_scores;
      for (const auto tensor_info : tensor_infos_.GetLiveTensorInfos()) {
        // Skip argument and output tensors.
        // Conservatively, we choose not to free tensors that are just rematerialized, because they
        // are dead before rematerialization and we cannot free their memory twice.
        if (tensor_info->let_var == curr_let_ ||
            std::find(new_args.begin(), new_args.end(), tensor_info->let_var) != new_args.end() ||
            newly_remat_tensors_.find(tensor_info->let_var) != newly_remat_tensors_.end()) {
          continue;
        }

        auto cost = EstimateRematCost(tensor_info->liveness_var, node);
        // Skip the tensors that cannot be rematerialized.
        if (cost != -1) {
          candidate_n_scores.push_back({tensor_info, cost});
        }
      }

      VERBOSE_LOG << "|-Force free to fit into " << budget_ / kMegaBytes << " MBs: ";
      // Sort candidates by their costs from high to low. If two tensors have the same score,
      // then the one created earlier will be marked as dead.
      std::sort(candidate_n_scores.begin(), candidate_n_scores.end(),
                [](std::pair<std::shared_ptr<TensorInfo>, float>& a,
                   std::pair<std::shared_ptr<TensorInfo>, float>& b) {
                  return (a.second == b.second) ? a.first->index > b.first->index
                                                : a.second > b.second;
                });
      VERBOSE_LOG << "| |-Cands: " << DebugDumpCandidates(candidate_n_scores);
      while (curr_mem_trace_ > budget_) {
        // Mark a var (tensor) to be dead and remove its size from memory trace. This tensor
        // will be rematerialized later when necessary.
        if (candidate_n_scores.empty()) {
          LOG(FATAL)
              << "Memory consumption at " << curr_let_ << " is " << curr_mem_trace_ / kMegaBytes
              << "MBs. Cannot rematerialize more tensors to meet the memory budget requirement ("
              << budget_ / kMegaBytes << " MBs). Please try a higher memory budget";
          throw;
        }
        auto cand_tensor_info = candidate_n_scores.back().first;
        candidate_n_scores.pop_back();
        tensor_infos_.MarkAsDead(cand_tensor_info);
        curr_mem_trace_ -= cand_tensor_info->size;
        VERBOSE_LOG << "| |-" << cand_tensor_info->liveness_var->name_hint() << " with "
                    << cand_tensor_info->size / kMegaBytes << " MBs";
        // Remove this tensor from the current live set
        auto liveness_var = cand_tensor_info->liveness_var;
        curr_live_in_vars_.erase(liveness_var);

        // When deciding to rematerialize a tensor, increment the use count of its direct
        // producers if they are still live. In this case, these tensors won't be considered
        // "dead" before the rematerialization takes place. This helps in the following case:
        /*
          x = expensive_compute()
          y = cheap_compute(x)
          ...
          # memory budget exceeded, decide to rematerialize y because it is cheap to compute
          # notice that x is still live at this point, so the cost function won't include
          # the cost of computing x
          free(y)
          ...
          # lifetime of x ends here, x gets killed
          free(x)
          ...
          # y is rematerialized here, but because x is already killed, it must be rematerialized too
          x' = expensive_compute()
          y' = cheap_compute(x')
          ...
        */
        // TODO: to properly track the use count of all tensors over here, we actually need to
        // recursively increment the use count of all affected tensors. Left for future work.
        auto cand_let_var = cand_tensor_info->let_var;
        auto cand_call_node = let_vars_[cand_let_var].as<CallNode>();
        CHECK(cand_call_node != nullptr)
            << "Tensor " << cand_let_var
            << " is not generated by a call node: " << raf::ir::AsText(cand_let_var);
        for (auto arg : cand_call_node->args) {
          if (auto var_node = arg.as<VarNode>()) {
            auto arg_var = GetRef<Var>(var_node);
            for (auto tensor_info : tensor_infos_.GetTensorInfoFromLetVar(arg_var)) {
              int64_t new_use_count = tensor_info->IncUseCount();
              VERBOSE_LOG << "Increment use count of " << arg_var->name_hint() << "("
                          << tensor_info->liveness_var->name_hint() << ")"
                          << " to reflect remat decision, now use count is " << new_use_count;
            }
          }
        }
      }
      VERBOSE_LOG << "|-CurrMem: " << curr_mem_trace_ / kMegaBytes << " MBs";
    }
    VERBOSE_LOG << "|-TraceMem: "
                << ((node->op->IsInstance<OpNode>()) ? node->op.as<OpNode>()->name : "closure")
                << "\t" << curr_mem_trace_ / kMegaBytes << " MBs";
    peak_memory_ = (curr_mem_trace_ > peak_memory_) ? curr_mem_trace_ : peak_memory_;

    // Gnereate a new call node with rematerialized arguments.
    auto call = GetRef<Call>(node);
    auto new_call = Call(call->op, new_args, call->attrs, call->type_args);
    new_call->checked_type_ = call->checked_type();
    return new_call;
  }

 private:
  class TensorAnalyzer;

  TensorInfos AnalyzeTensors(const Device& device, const Function& func, const IRModule& mod,
                             liveness_analysis::LivenessAnalyzer* analyzer,
                             op_profiler::OpProfiler* profiler);

  /*!
   * \brief Generate TupleGetItem or reshape to match the given type.
   * \param scope The current let list scope.
   * \param target_let_var The let-binding var of the target tensor.
   * \return The tensor with the correct type.
   */
  Var CorrectType(LetList* scope, const Var& target_let_var) {
    static const auto reshape_op = Op::Get("raf.op.reshape");

    auto tensor_infos = tensor_infos_.GetTensorInfoFromLetVar(target_let_var);
    if (tensor_infos.size() > 1) {
      // TODO: Handle tuple. This happens for ops taking a tuple as an argument (e.g., concat).
      // Ideally we should check each element in the tuple and correct their types.
      return target_let_var;
    }
    auto curr_let_var = tensor_infos[0]->let_var;

    // Do nothing if the types are already match.
    if (ObjectPtrHash()(curr_let_var->checked_type()) ==
        ObjectPtrHash()(target_let_var->checked_type())) {
      return curr_let_var;
    }

    // If not match, then the target let_var must be in a tensor type.
    auto target_type = target_let_var->checked_type().as<TensorTypeNode>();
    CHECK(target_type != nullptr);

    // Need to generate a TupleGetItem node.
    if (auto tuple_type_node = curr_let_var->checked_type().as<TupleTypeNode>()) {
      CHECK_NE(tensor_infos[0]->tuple_field_idx, -1);
      auto tgi = TupleGetItem(curr_let_var, tensor_infos[0]->tuple_field_idx);
      curr_let_var = scope->Push(tgi);
      tgi->checked_type_ = tuple_type_node->fields[tensor_infos[0]->tuple_field_idx];
      curr_let_var->checked_type_ = tgi->checked_type_;
      let_vars_.emplace(curr_let_var, tgi);
      tensor_infos_.AddLetVar(curr_let_var, tensor_infos);
    }

    auto ori_type = curr_let_var->checked_type().as<TensorTypeNode>();
    CHECK(ori_type != nullptr);

    // Need to generate a reshape call.
    bool need_reshape = target_type->shape.size() != ori_type->shape.size();
    if (!need_reshape) {
      for (size_t i = 0; i < ori_type->shape.size(); ++i) {
        if (ori_type->shape[i].as<AnyNode>() || target_type->shape[i].as<AnyNode>()) {
          need_reshape = true;
          break;
        }
        auto lhs = ori_type->shape[i].as<tvm::IntImmNode>();
        auto rhs = target_type->shape[i].as<tvm::IntImmNode>();
        need_reshape = need_reshape && (lhs != rhs);
      }
    }
    if (need_reshape) {
      Array<Expr> new_args;
      new_args.push_back(curr_let_var);
      new_args.push_back(MakeConstant(op::ArrayToIntTuple(target_type->shape)));
      auto reshape_call = Call(reshape_op, new_args);
      curr_let_var = scope->Push(reshape_call);
      let_vars_.emplace(curr_let_var, reshape_call);
      reshape_call->checked_type_ = target_let_var->checked_type();
      curr_let_var->checked_type_ = reshape_call->checked_type();
      tensor_infos_.AddLetVar(curr_let_var, tensor_infos_.GetTensorInfoFromLetVar(target_let_var));
    }

    return curr_let_var;
  }

  /*!
   * \brief Rematerize the given tensor (var). Note that this function may generate a sequence of
   * nodes if rematerializing the target tensor requires to rematerialize prior tensors.
   * \param scope The current let list scope.
   * \param var The let-binding var to be rematerialized.
   * \param curr_depth The current rematerialization depth.
   * \return The var of rematerialized tensor, or the given var if no rematerilization happened.
   */
  Var Rematerialize(LetList* scope, const Var& var, size_t curr_depth = 1) {
    // Stop if it has no binded expression (e.g., parameters).
    if (let_vars_.count(var) == 0) {
      return var;
    }

    auto tensor_infos = tensor_infos_.GetTensorInfoFromLetVar(var);
    auto latest_let_var = tensor_infos[0]->let_var;
    if (let_vars_.count(latest_let_var) == 0) {
      // The representative let-var may has no binded expression (i.e., parameter).
      return var;
    }
    auto call_node = let_vars_[latest_let_var].as<CallNode>();
    CHECK(call_node != nullptr) << "Tensor " << latest_let_var << " with alias " << var
                                << " is not a parameter nor generated by a call node: "
                                << raf::ir::AsText(let_vars_[latest_let_var]);

    // If the tensor is alive at this point, we can directly use it whatever it is
    // the original or the rematerialized tensor.
    bool all_alive = true;
    size_t total_size = 0;
    for (auto tensor_info : tensor_infos) {
      // If the tensor is already live, then its size should be already
      // included in curr_memory_trace_. Only when its dead should we increment
      // the size.
      bool tensor_is_alive = !tensor_info->IsDead();
      if (!tensor_is_alive) {
        total_size += tensor_info->size;
        // To be more conservative we can add the workspace size too
        total_size += tensor_info->workspace_size;
      }
      all_alive = all_alive && tensor_is_alive;
    }
    if (all_alive) {
      return CorrectType(scope, var);
    }

    // Recursively rematerialize arguments if necessary.
    Array<Expr> new_args;
    for (auto arg : call_node->args) {
      if (auto var_node = arg.as<VarNode>()) {
        auto arg_var = GetRef<Var>(var_node);
        // TODO: we should do use count tracking here.
        new_args.push_back(Rematerialize(scope, arg_var, curr_depth + 1));
      } else {
        new_args.push_back(arg);
      }
    }

    auto remat_call = Call(call_node->op, new_args, call_node->attrs, call_node->type_args);
    auto remat_var = scope->Push(remat_call);
    remat_var->checked_type_ = call_node->checked_type();
    remat_call->checked_type_ = call_node->checked_type();
    let_vars_.emplace(remat_var, remat_call);

    // Record for final report.
    n_recompute_ops_++;
    if (profiler_) {
      auto exec_time_and_ws_size = profiler_->ProfileOp(remat_call);
      // Default is to repeat once, so we take the first element
      auto compute_cost = exec_time_and_ws_size.first[0];
      total_recompute_cost_ += compute_cost;
    }

    // Update the let_var to be the rematerialized one and mark the tensor as live again.
    // It can be reused by future executions directly if no OOM happens in between;
    // otherwise it will be force freed at the future OOM point and will need to be
    // rematerialized again.
    tensor_infos_.UpdateLetVar(remat_var, tensor_infos);
    VERBOSE_LOG << "|-Remat: " << latest_let_var->name_hint() << " as " << remat_var->name_hint()
                << " with " << total_size / kMegaBytes << " MBs (depth " << curr_depth << ")";
    curr_mem_trace_ += total_size;
    // The tensor is revived, it's now live again, add to live set
    for (auto tensor_info : tensor_infos) {
      auto liveness_var = tensor_info->liveness_var;
      curr_live_in_vars_.insert(liveness_var);
    }
    // Add the newly-rematerialized tensor to the set
    newly_remat_tensors_.insert(remat_var);
    return CorrectType(scope, var);
  }

  /*!
   * \brief Estimate the rematerialization cost of the given tensor (let_var). The cost is estimated
   * by the equation `(cost * use_count) / size`, where cost is the latency cost of rematerializing
   * this tensor, and it might be calculated recursively if we have to rematerialize other tensors
   * prior to rematerialize this tensor. The idea of this equation is to prioritize a tensor with
   * large size, low cost to rematerialize, and less used in the rest of the graph.
   * \param liveness_var The liveness var to be estimated.
   * \param curr_call_node The current processing call node.
   * \param curr_depth The current back trace depth of estimating cost of a tensor.
   * \return The cost (lower the better). Note that -1 means rematerializing this tensor is invalid.
   */
  float EstimateRematCost(const Var& liveness_var, const CallNode* curr_call_node,
                          size_t curr_depth = 1) {
    if (curr_depth == MAX_REMAT_DEPTH) {
      return std::numeric_limits<float>::max();
    }

    // Canoncialize the target var to be the latest let binding var.
    auto tensor_info = tensor_infos_.GetTensorInfoFromLivenessVar(liveness_var);
    auto let_var = tensor_info->let_var;

    // Do not rematerialize input parameter, in-place update or small (< 1MB) tensors.
    if (tensor_info->is_param || !tensor_info->share_storage.empty() ||
        tensor_info->size < kMegaBytes) {
      return -1;
    }

    // Rematerialize the argument of the current processing call node is meaningless.
    for (auto arg : curr_call_node->args) {
      if (auto var_node = arg.as<VarNode>()) {
        for (auto tensor_info : tensor_infos_.GetTensorInfoFromLetVar(GetRef<Var>(var_node))) {
          if (tensor_info->let_var.same_as(let_var)) {
            return -1;
          }
        }
      }
    }

    // Only rematerialize tensors generated by a call node.
    CHECK_GT(let_vars_.count(let_var), 0U) << "Cannot find the binded expr for let " << let_var;
    auto expr = let_vars_[let_var];
    if (!expr.as<CallNode>()) {
      return -1;
    }
    auto remat_call = Downcast<Call>(expr);

    // Do not rematerialize tensors with an invalid cost.
    float cost = tensor_info->compute_cost;
    if (cost == -1) {
      return -1;
    }

    for (auto arg : remat_call->args) {
      if (auto var = arg.as<VarNode>()) {
        auto arg_var = GetRef<Var>(var);
        size_t n_live_tensor = 0;
        auto tensor_infos = tensor_infos_.GetTensorInfoFromLetVar(arg_var);
        for (auto tensor_info : tensor_infos) {
          n_live_tensor += (tensor_info->IsDead()) ? 0 : 1;
        }
        if (n_live_tensor == tensor_infos.size()) {
          // All tensors are alive, so no cost.
          continue;
        } else if (n_live_tensor > 0) {
          // Only a part of tensors in this tuple are alive. In this case, we heuristically
          // give up rematerializing the dead tensors.
          return -1;
        }
        auto arg_cost =
            EstimateRematCost(tensor_infos[0]->liveness_var, curr_call_node, curr_depth + 1);
        if (arg_cost == std::numeric_limits<float>::max()) {
          // Give up to rematerialize this tensor if it needs to rematerialize too many ops.
          return -1;
        } else if (arg_cost != -1) {
          cost += arg_cost;
        }
      }
    }

    cost += 0.1;  // Avoid 0 cost.
    return (cost * (tensor_info->GetUseCount() + 1)) / (tensor_info->size / kGigaBytes);
  }

  /*! \brief the function to be muatated. */
  const Function& func_;
  /*! \brief The scope stack of the let list. */
  std::vector<std::unique_ptr<LetList>> scopes_;
  /*! \brief The current processing let var. */
  Var curr_let_;
  /*! \brief The current live-in set. */
  VSet curr_live_in_vars_;
  /*! \brief The analyzed tensor infos. */
  TensorInfos tensor_infos_;
  /*! \brief The mapping from let bound var to its expr. */
  StdMap<Expr> let_vars_;
  /*! \brief The liveness analyzer, including liveness analysis results. */
  liveness_analysis::LivenessAnalyzer* analyzer_;
  /*! \brief The profiler used in rematerialization. */
  op_profiler::OpProfiler* profiler_;
  /*! \brief The memory budget in bytes. */
  int64_t budget_;
  /*! \brief The current memory consumption in bytes. */
  int64_t curr_mem_trace_ = 0;
  /*! \brief Peak mremory. */
  int64_t peak_memory_ = 0;
  /*! \brief The number of cloned recompute ops. */
  int64_t n_recompute_ops_ = 0;
  /*! \brief The total recompute cost. */
  float total_recompute_cost_ = 0;
  /*! \brief A set of rematerialized tensors before each call. */
  VSet newly_remat_tensors_;
};

/*!
 * \brief A simple visitor to analyze the size and use count of each tensor. The use count
 * will be used by Rematerializer as one criteria to determine the rematerialization cost.
 */
class Rematerializer::TensorAnalyzer : public ExprVisitor {
 public:
  TensorAnalyzer(const Device& device, const Function& func, const IRModule& mod,
                 liveness_analysis::LivenessAnalyzer* analyzer, op_profiler::OpProfiler* profiler)
      : func_(func), analyzer_(analyzer), ell_(ExplicitLetList::make(func)), profiler_(profiler) {
    CHECK(analyzer_->IsSuccess());
    if (!profiler_) {
      op_flops_estimater_.Run(device, func, mod);
    }
  }

  ~TensorAnalyzer() {
  }

  /*!
   * \brief Get a list of liveness vars for the current let var. This function uses a DFS to handle
   * nested tuples. The returned array contains a flat list of vars. The relative order of tuple
   * fields is preserved.
   */
  tvm::Array<Var> GetLivenessVars(const Var& curr_let) {
    tvm::Array<Var> result_vars;
    tvm::Array<Var> var_stack;
    var_stack.push_back(curr_let);
    while (!var_stack.empty()) {
      auto v = var_stack.back();
      var_stack.pop_back();
      auto liveness_vars = analyzer_->GetTensorVars(v);
      CHECK_GT(liveness_vars.size(), 0U);
      if (liveness_vars.size() > 1) {
        // If the current let var corresponds to a tuple, the tuple fields should be processed later
        for (auto it = liveness_vars.rbegin(); it != liveness_vars.rend(); it++)
          var_stack.push_back(*it);
      } else if (let_var_set_.count(liveness_vars[0])) {
        // If this "liveness var" points to a real var rather than an actual liveness var
        // it should also be processed later
        var_stack.push_back(liveness_vars[0]);
      } else {
        // Otherwise, add this var to the result
        result_vars.push_back(liveness_vars[0]);
      }
    }
    return result_vars;
  }

  /*! \brief Visit each let statement and return the analyzed information of each tensor. */
  TensorInfos Run() {
    // Analyze parameters.
    for (const auto& var : func_->params) {
      auto liveness_vars = analyzer_->GetTensorVars(var);
      tensor_infos_.CreateTensorInfo(var, liveness_vars, true);
    }

    const auto& vars = ell_->vars;
    const auto& exprs = ell_->exprs;
    CHECK_EQ(vars.size(), exprs.size());

    for (auto var : vars) {
      let_var_set_.insert(var);
    }

    size_t n = exprs.size();
    for (int i = 0; i < n; ++i) {
      curr_let_ = vars[i];
      auto liveness_vars = GetLivenessVars(curr_let_);

      // Visit the expression to analyze the use count
      ExprVisitor::VisitExpr(exprs[i]);

      float compute_cost = 0.0f;
      int64_t ws_size = 0;

      // Get the call node op if applicable.
      Op op;
      if (auto call_node = exprs[i].as<CallNode>()) {
        if (auto op_node = call_node->op.as<OpNode>()) {
          op = GetRef<Op>(op_node);
        }
      }

      if (op.defined() && (IsNonDeterministicOp(op) || IsCollectiveOp(op))) {
        // Non-deterministic and collective ops cannot be recomputed
        compute_cost = std::numeric_limits<float>::max();
      } else if (profiler_) {
        // Try to profile the op
        auto exec_time_and_ws_size = profiler_->ProfileOp(exprs[i]);
        // Default is to repeat once, so we take the first element
        compute_cost = exec_time_and_ws_size.first[0];
        ws_size = exec_time_and_ws_size.second;
      } else {
        // Use FLOPS estimator instead, the workspace size won't be tracked in this case
        compute_cost = op_flops_estimater_.GetFLOPS(curr_let_);
      }

      // Create the tensor info for this op
      tensor_infos_.CreateTensorInfo(curr_let_, liveness_vars, false, compute_cost, ws_size);
    }

    return tensor_infos_;
  }

  void VisitExpr_(const CallNode* node) override {
    for (auto arg : node->args) {
      if (auto var_node = arg.as<VarNode>()) {
        for (auto tensor_info : tensor_infos_.GetTensorInfoFromLetVar(GetRef<Var>(var_node))) {
          tensor_info->IncUseCount();
        }
      }
    }
  }

 private:
  /*! \brief the function to be analyzed. */
  const Function& func_;
  /*! \brief The current processing let var. */
  Var curr_let_;
  /*! \brief The let list. */
  std::unique_ptr<ExplicitLetList> ell_{nullptr};
  /*! \brief Op cost estimater. */
  estimate_flops::FLOPSEstimater op_flops_estimater_ = estimate_flops::FLOPSEstimater();
  /*! \brief The liveness analyzer, including liveness analysis results. */
  liveness_analysis::LivenessAnalyzer* analyzer_;
  /*! \brief The analyzed tensor infos. */
  TensorInfos tensor_infos_;
  /*! \brief The profiler used in rematerialization. */
  op_profiler::OpProfiler* profiler_;
  /*! \brief A set of all let vars in the function. */
  VSet let_var_set_;
};

TensorInfos Rematerializer::AnalyzeTensors(const Device& device, const Function& func,
                                           const IRModule& mod,
                                           liveness_analysis::LivenessAnalyzer* analyzer,
                                           op_profiler::OpProfiler* profiler) {
  return TensorAnalyzer(device, func, mod, analyzer, profiler).Run();
}

}  // namespace rematerialization

TVM_REGISTER_PASS_CONFIG_OPTION("raf.memory_budget", IntImm);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.remat.use_gflops_cost", IntImm);

Pass Rematerialization() {
  PassContext pass_ctx = PassContext::Current();
  int64_t memory_budget =
      pass_ctx->GetConfig("raf.memory_budget", Integer(static_cast<int>(0))).value().IntValue();
  // Turn profiler on by default. With caching it is pretty fast now.
  bool use_profiler = !(pass_ctx->GetConfig("raf.remat.use_gflops_cost", Bool(false)).value());
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    // We use budget 0 to diable this pass because it is guaranteed to fail.
    if (memory_budget == 0) {
      return f;
    }
    auto device = Device::Current();
    if (device.device_type() == DevType::kUnknown() && device.device_id() == -1) {
      LOG(WARNING) << "Target device is undefined. Skip rematerialization.";
      return f;
    }

    VERBOSE_LOG << "Memory budget for rematerialization: "
                << (float)memory_budget / rematerialization::kMegaBytes << " MBs";

    auto analyzer = liveness_analysis::LivenessAnalyzer(f);
    analyzer.Run();
    if (!analyzer.IsSuccess()) {
      LOG(WARNING) << "Rematerialization is disabled because liveness analysis was failed";
      return f;
    }

    op_profiler::OpProfiler* profiler = nullptr;
    if (use_profiler) {
      LOG(INFO)
          << "Using profiler-based cost estimation for rematerialization. This may take a while. ";
      profiler = op_profiler::OpProfiler::Get(device);
    } else {
      LOG(INFO) << "Using GFLOPS-based cost estimation. ";
    }
    return Downcast<Function>(
        rematerialization::Rematerializer(&analyzer, device, f, m, memory_budget, profiler).Run());
  };

  Pass func_pass = CreateRAFFunctionPass(pass_func, 2, "RematerializationHelper", {});
  PassInfo pass_info(2, "Rematerialization", {});
  return RAFSequential({InferType(), func_pass, DeadCodeElimination()}, pass_info);
}

RAF_REGISTER_GLOBAL("raf.pass_.Rematerialization").set_body_typed(Rematerialization);

}  // namespace pass
}  // namespace raf
