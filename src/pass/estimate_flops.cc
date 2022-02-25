/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file estimate_flops.cc
 * \brief Estimate the computation FLOPS of the given function.
 */
#include "estimate_flops.h"

namespace raf {
namespace pass {
namespace estimate_flops {

using namespace raf::op;

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
      auto var = raf::ir::MakeVar("", param_types[i]);
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
      tvm_dialect::CalcFuncGFLOPS(call_values, param_types, ret_type, device_);
}

}  // namespace estimate_flops

using PackedFLOPSMap = Map<Var, FloatImm>;

estimate_flops::StdMap<float> EstimateGFLOPS(const IRModule& mod, const Device& device) {
  auto entry = mod->GetGlobalVar("main");
  auto func = Downcast<Function>(mod->Lookup(entry));
  auto estimator = estimate_flops::FLOPSEstimater();
  return estimator.Run(device, func, mod);
}

// Put the flops to Map as std::unordered_map is not in the object system.
PackedFLOPSMap EstimateGFLOPSPacked(const IRModule& mod) {
  auto device = Device::Current(false);

  PackedFLOPSMap ret;
  auto res = EstimateGFLOPS(mod, device);
  for (const auto& it : res) {
    ret.Set(it.first, FloatImm(DataType::Float(32), it.second));
  }
  return ret;
}

RAF_REGISTER_GLOBAL("raf.pass_.EstimateGFLOPS").set_body_typed(EstimateGFLOPSPacked);

}  // namespace pass
}  // namespace raf
