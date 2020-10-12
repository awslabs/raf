/*!
 * Copyright (c) 2019 by Contributors
 * \file data_parallel.cc
 * \brief Data Parallel pass
 */
#include <sstream>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/stream_pool.h"
#include "./common.h"

namespace mnm {
namespace pass {
namespace data_parallel {

using namespace mnm::ir;
using namespace mnm::op;
using mnm::value::NoGradValue;
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

   */
 public:
  explicit DataParallel(const FunctionNode* func)
      : func(func), fp_ell(ExplicitLetList::make(func->body)) {
  }

  Function Run() {
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
    static Op op_sync = Op::Get("mnm.op.stream_sync");
    auto args_x = bp_ell->vars[bp_ell->vars.size() - 2];
    auto args_stream = MakeConstant(value::IntValue::make(StreamTagEnum::CudaCommunicate()));
    bp_ell->vars.insert(--bp_ell->vars.end(), mnm::ir::MakeVar("null", {}));
    bp_ell->exprs.insert(--bp_ell->exprs.end(), Call(op_sync, {args_x, args_stream}));

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
};

}  // namespace data_parallel

ir::Function AutoDataParallel(ir::Function func) {
  return data_parallel::DataParallel(func.operator->()).Run();
}

MNM_REGISTER_GLOBAL("mnm.pass_.AutoDataParallel").set_body_typed(AutoDataParallel);

}  // namespace pass
}  // namespace mnm
