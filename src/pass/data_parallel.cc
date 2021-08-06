/*!
 * Copyright (c) 2019 by Contributors
 * \file data_parallel.cc
 * \brief Data Parallel pass
 */
#include <set>
#include <sstream>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "mnm/dist_context.h"
#include "mnm/profiler.h"
#include "mnm/stream_pool.h"
#include "./common.h"

namespace mnm {
namespace pass {
namespace data_parallel {

using namespace mnm::ir;
using namespace mnm::op;
using mnm::distributed::DistContext;
using mnm::value::NoGradValue;
using profiler::Profiler;
using profiler::ProfileStat;
using stream_pool::StreamTagEnum;

struct DataParallel {
  /* =================================================================
  Description:
      Data Parallel Pass will mainly modify the backward closure currently. The modification is
      1) adding communication op after the op which generate the local gradient.
      2) update the returned gradient from local gradient to aggregated global gradient.
      3) adding a stream_sync op before the end of backward closure to ensure communication is done.
  Example:
        Backward closure before DataParallel Pass:
        ```
        let %closure = fn (%dy) {
            let %x1 = mnm.op.nll_loss_dtrue(%y_true, %a4);
            let %x2 = mnm.op.nll_loss_dpred(%y_true, %a4);
            %0 = mnm.op.get_reduce_axis(%x2, %a3);
            %1 = mnm.op.get_kept_dims(%x2, %a3);
            let %x3 = mnm.op.sum(%x2, %0, %1);
            %2 = mnm.op.get_reduce_axis(%x2, %linear1.b);
            %3 = mnm.op.get_kept_dims(%x2, %linear1.b);
            let %x4 = mnm.op.sum(%x2, %2, %3);
            let %x5 = mnm.op.matmul(%x3, %linear1.w);
            let %x6 = mnm.op.matmul_tn(%x3, %a2);
            %4 = mnm.op.batch_norm_train_dxwb(%x5, %x, %bn1.w, %bn1.b, -114514);
            let %x7 = %4.0;
            let %x8 = %4.1;
            let %x9 = %4.2;
            let %x10 = (%x7, %x1, %x9, -114514, -114514, %x8, %x4, %x6);
            %x10
        };
        ```
        Backward closure after DataParallel Pass:
        ```
        let %closure = fn (%dy) {
            let %x1 = mnm.op.nll_loss_dtrue(%y_true, %a4);
            %0 = (%x1,);
            let %g = mnm.op._allreduce(%0);
            let %x2 = mnm.op.nll_loss_dpred(%y_true, %a4);
            %1 = mnm.op.get_reduce_axis(%x2, %a3);
            %2 = mnm.op.get_kept_dims(%x2, %a3);
            let %x3 = mnm.op.sum(%x2, %1, %2);
            %3 = mnm.op.get_reduce_axis(%x2, %linear1.b);
            %4 = mnm.op.get_kept_dims(%x2, %linear1.b);
            let %x4 = mnm.op.sum(%x2, %3, %4);
            %5 = (%x4,);
            let %g1 = mnm.op._allreduce(%5);
            let %x5 = mnm.op.matmul(%x3, %linear1.w);
            let %x6 = mnm.op.matmul_tn(%x3, %a2);
            %6 = (%x6,);
            let %g2 = mnm.op._allreduce(%6);
            %7 = mnm.op.batch_norm_train_dxwb(%x5, %x, %bn1.w, %bn1.b, -114514);
            let %x7 = %7.0;
            %8 = (%x7,);
            let %g3 = mnm.op._allreduce(%8);
            let %x8 = %7.1;
            %9 = (%x8,);
            let %g4 = mnm.op._allreduce(%9);
            let %x9 = %7.2;
            %10 = (%x9,);
            let %g5 = mnm.op._allreduce(%10);
            let %null = mnm.op.stream_sync(%g5, -114514);
            let %x10 = (%g3, %g, %g5, -114514, -114514, %g4, %g1, %g2);
            %x10
        };
        ```
        After enabling overlap between communication and forward, the IR could(not must) be:
        Backward closure after DataParallel Pass:
        ```
        let %closure = fn (%dy) {
            let %x1 = mnm.op.nll_loss_dtrue(%y_true, %a4);
            %0 = (%x1,);
            let %g = mnm.op._allreduce(%0);
            let %x2 = mnm.op.nll_loss_dpred(%y_true, %a4);
            %1 = mnm.op.get_reduce_axis(%x2, %a3);
            %2 = mnm.op.get_kept_dims(%x2, %a3);
            let %x3 = mnm.op.sum(%x2, %1, %2);
            %3 = mnm.op.get_reduce_axis(%x2, %linear1.b);
            %4 = mnm.op.get_kept_dims(%x2, %linear1.b);
            let %x4 = mnm.op.sum(%x2, %3, %4);
            %5 = (%x4,);
            let %g1 = mnm.op._allreduce(%5);
            let %x5 = mnm.op.matmul(%x3, %linear1.w);
            let %x6 = mnm.op.matmul_tn(%x3, %a2);
            %7 = mnm.op.batch_norm_train_dxwb(%x5, %x, %bn1.w, %bn1.b, -114514);
            let %x7 = %7.0;
            let %x8 = %7.1;
            let %x9 = %7.2;
            %8 = (%x9,);
            let %g3 = mnm.op._allreduce(%8);
            %9 = (%x8,);
            let %g4 = mnm.op._allreduce(%9);
            %10 = (%x7,);
            let %g5 = mnm.op._allreduce(%10);
            %11 = (%x6,);
            let %g2 = mnm.op._allreduce(%11);
            let %null = mnm.op.stream_sync(%g5, -114514);
            let %x10 = (%g3, %g, %g5, -114514, -114514, %g4, %g1, %g2);
            %x10
        };
        ```
   */
 public:
  explicit DataParallel(const FunctionNode* func)
      : func(func), fp_ell(ExplicitLetList::make(func->body)) {
  }

  // Compute the dctx->scheduling_param according to the analysis of op profiling.
  void GetSchedulingParameters() {
    auto dctx = DistContext::Global();
    static int prof_level = Profiler::Get()->profile_level();  // Store user's config

    // Store the running time of all ops.
    static std::vector<std::pair<std::string, int64_t> > op_running_time;
    if (dctx->iteration < dctx->auto_dp_profiling_start_iter) {
      Profiler::Get()->set_profile_level(0);  // Disable profiling to warm up
    } else if (dctx->iteration <= dctx->auto_dp_profiling_end_iter) {
      Profiler::Get()->set_profile_level(1);  // Profiling the execution
      if (op_running_time.empty()) {
        for (auto& item : Profiler::Get()->GetProfileStats()) {
          if (item.categories_ == "SchedulingCommunication") {
            int64_t duration = item.items_[1].timestamp_ - item.items_[0].timestamp_;
            if (scheduled_communication_ops.find(item.name_) != scheduled_communication_ops.end()) {
              // Using negative value to represent communication ops.
              op_running_time.push_back({item.name_, -duration});
            } else {
              // Using positive value to represent non-communication ops.
              op_running_time.push_back({item.name_, duration});
            }
          }
        }
      } else {
        int op_count = 0;
        for (auto& item : Profiler::Get()->GetProfileStats()) {
          if (item.categories_ == "SchedulingCommunication") {
            int64_t duration = item.items_[1].timestamp_ - item.items_[0].timestamp_;
            CHECK_EQ(item.name_, op_running_time[op_count].first);
            bool is_comp = op_running_time[op_count].second >= 0;
            op_running_time[op_count].second += is_comp ? duration : -duration;
            op_count++;
          }
        }
      }
    } else if (dctx->iteration == dctx->auto_dp_profiling_end_iter + 1) {
      Profiler::Get()->set_profile_level(prof_level);  // Enbale user's config
      // Analyse the profiling result of iter2-iter4,
      // and figure out a scheduling strategy.
      int64_t comp_total_time = 0;
      int first_comm_op = 0;
      int bp_order_grad_count = 0;

      // Get the location of the communication op.
      for (int i = 0; i < op_running_time.size(); i++) {
        if (op_running_time[i].second < 0) break;
        first_comm_op++;
      }

      // Compute the total execution time of comp ops after the first allreduce.
      for (int i = first_comm_op + 1; i < op_running_time.size(); i++)
        if (op_running_time[i].second >= 0) comp_total_time += op_running_time[i].second;

      // Choose the scheduling point, in which computation has finished but communication hasn't.
      for (int i = first_comm_op; i < op_running_time.size(); i++) {
        if (op_running_time[i].second < 0) {
          comp_total_time += op_running_time[i].second;
          bp_order_grad_count++;
          if (comp_total_time <= 0) break;
        }
      }
      // Currently we only have one scheduling parameter, set it here.
      dctx->scheduling_param = bp_order_grad_count;

      // clear the cached profiling analysis.
      op_running_time.clear();
    }
  }

  Function Run() {
    auto dctx = DistContext::Global();

    // If we want to overlap communication and forward pass,
    // we need to analyze the running time of Ops
    if (dctx->overlap_comm_forward && dctx->iteration <= dctx->auto_dp_profiling_end_iter + 1) {
      GetSchedulingParameters();
    }

    size_t fp_n = fp_ell->vars.size();
    auto closure_expr = fp_ell->exprs.at(fp_n - 2);
    Array<Var> bp_params;
    if (const auto* func = closure_expr.as<FunctionNode>()) {
      bp_ell = ExplicitLetList::make(func->body);
      bp_params = func->params;
    }
    size_t bp_n = bp_ell->vars.size();
    auto bp_grads = bp_ell->exprs.at(bp_n - 1);
    std::set<const VarNode*> gradset;  // All the gradients that returned by backward IR.
    if (const auto* tuple = bp_grads.as<TupleNode>()) {
      for (auto g : tuple->fields) {
        if (const auto* var = g.as<VarNode>()) {
          gradset.insert(var);
        }
      }
    } else if (const auto* var = bp_grads.as<VarNode>()) {
      gradset.insert(var);
    } else {
      LOG(FATAL) << "Return of backward IR must be Var or tuple of Vars in Data Parallel Pass.";
    }

    // The map from original local gradient to aggregated global gradient.
    std::map<mnm::ir::Expr, mnm::ir::Var> var_var_map;
    // Enlarge the size of bp_ell to fit the allreduce ops.
    bp_ell->vars.resize(bp_n + gradset.size());
    bp_ell->exprs.resize(bp_n + gradset.size());
    // p1 tracks the processing var/expr (from end to begin)
    // p2 tracks the next vacant position to paste the processing var/expr or allreduce op.
    int p1 = bp_n - 1, p2 = bp_n - 1 + gradset.size();

    bp_ell->vars[p2] = bp_ell->vars[p1];
    bp_ell->exprs[p2] = bp_ell->exprs[p1];
    --p2;
    --p1;

    for (int i = p1; i >= 0; --i) {
      if (gradset.find(bp_ell->vars[i].operator->()) != gradset.end()) {
        // If the current expr is an op-expr which generate local gradient,
        // we should add a allreduce op after it.
        static Op op_allreduce = Op::Get("mnm.op._allreduce");
        // Here we name the var as 'g'(global gradient), to help us identify it easier.
        bp_ell->vars[p2] = mnm::ir::MakeVar("g", {});
        bp_ell->exprs[p2] = Call(op_allreduce, {Tuple({bp_ell->vars[i]})});
        var_var_map.insert({bp_ell->vars[i], bp_ell->vars[p2]});
        --p2;
      }
      bp_ell->vars[p2] = bp_ell->vars[i];
      bp_ell->exprs[p2] = bp_ell->exprs[i];
      --p2;
    }

    Array<Expr> new_bp_rt;
    if (const auto* tuple = bp_grads.as<TupleNode>()) {
      for (int i = 0; i < tuple->fields.size(); ++i) {
        if (tuple->fields[i]->IsInstance<VarNode>()) {
          auto it = var_var_map.find(tuple->fields[i]);
          new_bp_rt.push_back(it->second);
        } else {
          new_bp_rt.push_back(tuple->fields[i]);
        }
      }
    } else if (bp_grads->IsInstance<VarNode>()) {
      auto it = var_var_map.find(bp_grads);
      new_bp_rt.push_back(it->second);
    } else {
      LOG(FATAL) << "Return of backward IR must be Var or tuple of Vars in Data Parallel Pass.";
    }

    if (!dctx->overlap_comm_forward) {
      static Op op_sync = Op::Get("mnm.op.stream_sync");
      auto args_x = bp_ell->vars[bp_ell->vars.size() - 2];
      auto args_stream = MakeConstant(value::ScalarValue::make(StreamTagEnum::CudaCommunicate()));
      bp_ell->vars.insert(--bp_ell->vars.end(), mnm::ir::MakeVar("null", {}));
      bp_ell->exprs.insert(--bp_ell->exprs.end(), Call(op_sync, {args_x, args_stream}));
    } else if (dctx->iteration > dctx->auto_dp_profiling_end_iter) {
      // Start scheduling from the this iteration.
      bp_n = bp_ell->vars.size();
      int fp_order_grad_count = var_var_map.size() - dctx->scheduling_param;
      int bp_order_grad_count = 0;
      int p_j = bp_n - 1;
      std::vector<Var> fp_order_grad_var;    // grads whose trasmission order will be reversed.
      std::vector<Expr> fp_order_grad_expr;  // grads whose trasmission order will be reversed.
      for (int i = 0; i < bp_n - 1; i++) {
        // If name_hint is 'g', the it means taht this is a global gradient
        // ,and there is a communication operator in bp_ell->exprs[i].
        if (bp_ell->vars[i]->name_hint() == "g") {
          bp_order_grad_count++;
          if (bp_order_grad_count > dctx->scheduling_param) {
            fp_order_grad_var.push_back(bp_ell->vars[i]);
            fp_order_grad_expr.push_back(bp_ell->exprs[i]);
            if (bp_order_grad_count == dctx->scheduling_param + 1) p_j = i;
            while (i + 1 < bp_n - 1 && bp_ell->vars[i + 1]->name_hint() != "g") {
              bp_ell->vars[p_j] = bp_ell->vars[i + 1];
              bp_ell->exprs[p_j] = bp_ell->exprs[i + 1];
              p_j++;
              i++;
            }
          }
        }
      }
      CHECK_EQ(fp_order_grad_count, fp_order_grad_var.size());
      for (int i = fp_order_grad_count - 1; i >= 0; i--) {
        bp_ell->vars[p_j] = fp_order_grad_var[i];
        bp_ell->exprs[p_j] = fp_order_grad_expr[i];
        p_j++;
      }
      CHECK_EQ(p_j, bp_n - 1);
    }
    dctx->iteration++;

    bp_n = bp_ell->vars.size();
    if (new_bp_rt.size() == 1) {
      bp_ell->exprs[bp_n - 1] = new_bp_rt[0];
    } else {
      bp_ell->exprs[bp_n - 1] = Tuple(new_bp_rt);
    }

    fp_ell->exprs[fp_n - 2] = Function(bp_params, bp_ell->AsExpr(), {}, {});
    return Function(func->params, fp_ell->AsExpr(), {}, {});
  }

 private:
  // initialized in constructor
  const FunctionNode* func;
  std::unique_ptr<ExplicitLetList> fp_ell{nullptr};
  // initialized in Run
  std::unique_ptr<ExplicitLetList> bp_ell{nullptr};
  // The comminication operators whose profiling will be collected for scheduling.
  const std::set<std::string> scheduled_communication_ops = {"mnm.op._allreduce"};
};

}  // namespace data_parallel

Pass AutoDataParallel() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return data_parallel::DataParallel(f.operator->()).Run();
  };
  return CreateMNMFunctionPass(pass_func, 0, "AutoDataParallel", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.AutoDataParallel").set_body_typed(AutoDataParallel);

}  // namespace pass
}  // namespace mnm
