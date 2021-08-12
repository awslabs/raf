/*!
 * Copyright (c) 2021 by Contributors
 * \file estimate_flops.cc
 * \brief Estimate the computation FLOPS of the given function.
 */
#include "estimate_flops.h"

namespace mnm {
namespace pass {
namespace estimate_flops {

using namespace mnm::op;

template <typename T>
using StdMap = std::unordered_map<Var, T, ObjectPtrHash, ObjectPtrEqual>;

void FLOPSEstimater::VisitExpr_(const LetNode* op) {
  auto pre_visit = [this](const LetNode* op) {
    Expr ovalue = op->value;
    Var var = op->var;
    Expr value = ovalue;
    curr_let_ = var;

    auto fn_node = value.as<FunctionNode>();
    bool visit_body = !fn_node || !fn_node->HasNonzeroAttr(attr::kPrimitive);
    if (visit_body) {
      this->VisitExpr(ovalue);
    }

    if (value.as<ConstantNode>()) {
      return;
    }
  };
  auto post_visit = [this](const LetNode* op) {
    Expr ovalue = op->value;
    Var var = op->var;
    Expr value = ovalue;

    auto fn_node = value.as<FunctionNode>();
    bool visit_body = !fn_node || !fn_node->HasNonzeroAttr(attr::kPrimitive);
    if (visit_body) {
      this->VisitExpr(ovalue);
    }

    this->visit_counter_[op] += 1;
    if (value.as<ConstantNode>()) {
      return;
    }
    VisitExpr(op->body);
  };
  ExpandANormalForm(op, pre_visit, post_visit);
}

void FLOPSEstimater::VisitExpr_(const CallNode* call) {
  Array<Type> param_types;
  for (auto arg : call->args) {
    param_types.push_back(arg->checked_type());
  }
  auto ret_type = call->checked_type();

  // Generate call values.
  CallValues call_values = CallValues::make();

  // Make argument values.
  Array<Value> arg_values;
  for (const auto& arg : call->args) {
    arg_values.push_back(GetValue(arg));
  }
  call_values->args = MakeListArgs(arg_values);

  // Assign or make the callee.
  if (call->op.as<OpNode>()) {
    // Wrap a single op to a Relay function.
    std::vector<Var> params;
    for (int i = 0, n = param_types.size(); i < n; ++i) {
      auto var = mnm::ir::MakeVar("", param_types[i]);
      var->checked_type_ = param_types[i];
      params.push_back(var);
    }
    Function func =
        Function(params, Call(call->op, {params.begin(), params.end()}, call->attrs), ret_type, {});
    func->body->checked_type_ = ret_type;
    func->checked_type_ = FuncType(param_types, ret_type, {}, {});
    call_values->callee = ClosureValue::make({}, func);
  } else if (auto fn = call->op.as<FunctionNode>()) {
    call_values->callee = ClosureValue::make({}, GetRef<Function>(fn));
  } else if (auto gvn = call->op.as<GlobalVarNode>()) {
    // Look up the function body from the module.
    call_values->callee =
        ClosureValue::make({}, Downcast<Function>(mod_->Lookup(GetRef<GlobalVar>(gvn))));
  } else {
    LOG(FATAL) << "Unrecognized call op type: " << call->op->GetTypeKey();
    throw;
  }
  var_flops_map_[curr_let_] =
      tvm_dialect::CalcFuncFLOPS(call_values, param_types, ret_type, target_);
}

}  // namespace estimate_flops

using PackedFLOPSMap = Map<Var, Integer>;

estimate_flops::StdMap<int64_t> EstimateFLOPS(const IRModule& mod, const Target& target) {
  auto entry = mod->GetGlobalVar("main");
  auto func = Downcast<Function>(mod->Lookup(entry));
  auto estimator = estimate_flops::FLOPSEstimater();
  return estimator.Run(target, func, mod);
}

// Put the flops to Map as std::unordered_map is not in the object system.
PackedFLOPSMap EstimateFLOPSPacked(const IRModule& mod) {
  auto target = tvm::Target::Current();
  if (!target.defined()) {
    LOG(FATAL) << "Target device is undefined.";
    throw;
  }

  PackedFLOPSMap ret;
  auto res = EstimateFLOPS(mod, target);
  for (const auto& it : res) {
    ret.Set(it.first, it.second);
  }
  return ret;
}

MNM_REGISTER_GLOBAL("mnm.pass_.EstimateFLOPS").set_body_typed(EstimateFLOPSPacked);

}  // namespace pass
}  // namespace mnm
