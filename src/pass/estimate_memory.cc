/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file estimate_memory.cc
 * \brief Estimate the memory footprint. Note that this can only be used after ManifestAlloc pass.
 */
#include "raf/device.h"
#include "raf/op.h"
#include "raf/op_profiler.h"
#include "raf/pass.h"
#include "./let_list.h"
#include "./common.h"
#include "../common/shape_utils.h"

namespace raf {
namespace pass {
namespace estimate_memory {

using namespace raf::op;

using MemoryTrace = Array<Array<ObjectRef>>;

constexpr float kMegaBytes = 1048576;

/*!
 * \brief A visitor to visit after ManifestAlloc ANF IR and estimate the memory footprint.
 */
class MemoryTracer : public ExprVisitor {
 public:
  MemoryTracer(const Device& device, const Function& func, const IRModule& mod, bool include_params)
      : ell_(ExplicitLetList::make(func->body)), device_(device), mod_(mod) {
    profiler_ = op_profiler::OpProfiler::Get(device);
    if (include_params) {
      for (const auto param : func->params) {
        auto size = common::shape_utils::BytesCompactType(param->checked_type());
        curr_memoey_mbs_ += size / kMegaBytes;
      }
    }
  };

  MemoryTrace Run() {
    const auto& vars = ell_->vars;
    const auto& exprs = ell_->exprs;
    CHECK_EQ(vars.size(), exprs.size());
    int n = exprs.size();

    LOG(INFO) << "Estimating memory footprint...";
    int next_print = 0;
    for (int i = 0; i < n; ++i) {
      if (i == next_print) {
        LOG(INFO) << "Estimated " << std::fixed << std::setprecision(2) << 100.0 * i / n << "% ("
                  << i << "/" << n << ")";
        next_print += n / 10;
      }
      let_map_.Set(vars[i], exprs[i]);
      curr_let_ = vars[i];
      ExprVisitor::VisitExpr(exprs[i]);
    }

    // Add the final trace. At this point, the memory usage should just include the outputs.
    trace_.push_back({String("out"), FloatImm(DataType::Float(32), curr_memoey_mbs_)});
    return trace_;
  }

  void VisitExpr_(const CallNode* call) final {
    static const Op& alloc_storage_op = Op::Get("raf.op.vm.alloc_storage");
    static const Op& free_op = Op::Get("raf.op.vm.free");
    static const Op& invoke_op = Op::Get("raf.op.vm.invoke_op");

    const auto* op_node = call->op.as<OpNode>();
    CHECK(op_node != nullptr)
        << "Found a call node that invokes a clusure/function. Did you run ManifestAlloc?";
    if (GetRef<Op>(op_node) == invoke_op) {
      // Invoke ops are the real call nodes in the IR, so create memory trace based on them.
      // Build OpEnv to get the op name and the possible workspace memory.
      auto callee_op = let_map_[Downcast<Var>(call->args[0])];
      auto args = Downcast<Tuple>(let_map_[Downcast<Var>(call->args[1])])->fields;
      auto callee = pass::InferType(Call(callee_op, args));

      // Use op profiler to build the OpEnv. Note that we only care workspace memory size
      // and the op name, so we ignore the latency profiling by setting numbers to 0.
      auto exec_time_and_ws_size = profiler_->ProfileOp(callee, 0, 0, 0);
      auto op_env = profiler_->GetOpEnv(callee);
      std::string name = (op_env != nullptr) ? op_env->name() : "unknown";
      auto ws_size = exec_time_and_ws_size.second / kMegaBytes;

      trace_.push_back({String(name), FloatImm(DataType::Float(32), curr_memoey_mbs_ + ws_size)});
    } else if (GetRef<Op>(op_node) == alloc_storage_op) {
      // Alloc a new buffer.
      auto size = call->args[0].as<ConstantNode>()->value.as<IntValueObj>()->value / kMegaBytes;
      curr_memoey_mbs_ += size;
      storage_vars_[curr_let_] = size;
    } else if (GetRef<Op>(op_node) == free_op) {
      // Free a buffer.
      auto storage_var = Downcast<Var>(call->args[0]);
      CHECK_GE(storage_vars_.count(storage_var), 1U);
      curr_memoey_mbs_ -= storage_vars_[storage_var];
      storage_vars_.erase(storage_var);
    }
  }

 private:
  /*! \brief The current processing let var. */
  Var curr_let_;
  /*! \brief Let binding vars to the expression. */
  Map<Var, Expr> let_map_;
  /*! \brief Current live storage vars to sizes. */
  std::unordered_map<Var, float, ObjectPtrHash, ObjectPtrEqual> storage_vars_;
  /*! \brief the explicit let list of func_ */
  std::unique_ptr<ExplicitLetList> ell_{nullptr};
  /*! \brief The profiler used in rematerialization. */
  op_profiler::OpProfiler* profiler_;
  /*! \brief The IR module that the target function belongs to. This is used to look up
   * other functions called by global symbols (TODO: not implemented yet). */
  IRModule mod_;
  /*! \brief The target device used to estimate the rematerialization cost. */
  Device device_;
  /*! \brief The collected memory trace. */
  MemoryTrace trace_;
  /*! \brief Current memory usage. */
  float curr_memoey_mbs_ = 0;
};

}  // namespace estimate_memory

estimate_memory::MemoryTrace EstimateMemory(const IRModule& mod, const Device& device,
                                            bool include_params) {
  auto entry = mod->GetGlobalVar("main");
  auto func = Downcast<Function>(mod->Lookup(entry));
  auto estimator = estimate_memory::MemoryTracer(device, func, mod, include_params);
  return estimator.Run();
}

RAF_REGISTER_GLOBAL("raf.pass_.EstimateMemory").set_body_typed(EstimateMemory);

}  // namespace pass
}  // namespace raf
