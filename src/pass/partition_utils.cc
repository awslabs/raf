/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file partition_utils.cc
 * \brief Utility functions for partitioning ANF into multiple functions. An
 * example can be found in the comments of anf_partition.cc.
 */

#include "./partition_utils.h"

namespace raf {
namespace pass {

/**
 * \brief Find the boundary of the partitions, so to keep the rest IR unchanged.
 * \param f the original un-partitioned function.
 * \return the boundary variable defines the last partition.
 */
Var GetPartitionBoundary(Function f) {
  auto ell = ExplicitLetList::make(f->body);
  std::unordered_set<const VarNode*> ret_vars{ell->ret.get()};
  for (int i = ell->vars.size() - 1; i >= 0; --i) {
    Var var = ell->vars[i];
    Expr value = ell->exprs[i];
    // skip
    // let %var = (..., ...) or
    // let %var = tuple.0
    // etc.
    if (!value.as<TupleNode>() && !value.as<TupleGetItemNode>()) {
      return var;
    }
  }
  return ell->ret;
}

/*!
 * \brief Remove the outputs which won't be used later.
 * \param analyzer LinvenessAnalyzer which has been run on the original program.
 * \param next_var The next variable follows the current PartitionFunction.
 * \param is_final_ret whether next_var is the final return output of the origianl program.
 */
void PartitionFunction::TrimOutputs(liveness_analysis::LivenessAnalyzer& analyzer,
                                    const Var& next_var, bool is_final_ret) {
  if (is_final_ret) {
    outs_ = {next_var};
  } else if (analyzer.IsSuccess()) {
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> live_var =
        analyzer.GetLiveVars(next_var);
    std::vector<Expr> trimmed_outs;
    for (auto out_expr : outs_) {
      Var out = Downcast<Var>(out_expr);
      if (analyzer.IsAlive(out, live_var)) {
        trimmed_outs.push_back(out);
      }
    }
    outs_ = trimmed_outs;
  }
}

/*!
 * \brief Export the partition functions into ExplicitLetList.
 * \param part_func_vars The map from old vars into func_named vars
 * \return The ExplicitLetList with partition function packed.
 */
void PartitionFunction::ExportTo(ExplicitLetList* ret_ell,
                                 Map<Var, Var>& intermediate_var_2_func_out) {
  // Because anf will auto-capture the global vars, we don't need to push ins_ into params.
  // If the Var inside ins_ is inside the outs_, which indicate that this input is given by
  // the expr inside this function. Then replace the usage of old vars with func_named vars.
  Array<Var> params_array;

  // Surround the Values in the outs_ with a TupleNode. And replace the old
  // vars with part_func named vars.
  CHECK_GT(outs_.size(), 0);
  if (outs_.size() > 1) {
    Tuple outs_tuple = Tuple(outs_);
    std::string outs_var_name = func_name_ + "_outs";
    Var outs_var = MakeVar(outs_var_name, {});
    ell_.vars.push_back(outs_var);
    ell_.exprs.push_back(outs_tuple);
    ell_.ret = outs_var;
  } else {
    ell_.ret = Downcast<Var>(outs_[0]);
  }
  // Assemble the partition function.
  Expr body = ell_.AsExpr();
  // replace the usage of intermediate tensors in previous function
  // to be their function outputs.
  VarSubstitutor substitutor(intermediate_var_2_func_out);
  body = substitutor.Substitute(body);
  // it's a closure with 0 params and outputs live intermediate variables.
  /*
  func() {
    let %a0 = ...
    let %a1 = ...
    let %a2 = ...
    let %func_outs = (%a0, %a1, %a2)
    %func_outs
  }
  */
  auto func = Function({}, body, {}, {});

  // Insert the CallNode for the function
  // and TupleGetItemNode to get the outputs from the function.
  Var func_var = MakeVar(func_name_, {});
  ret_ell->vars.push_back(func_var);
  ret_ell->exprs.push_back(func);

  // Call the partition function
  auto func_call = Call(func_var, {}, Attrs());
  std::string ret_var_name = func_var->name_hint() + "_ret";
  Var ret_var = MakeVar(ret_var_name, {});
  // let %func = func() {}
  // let %func_ret = Call(%func, {})
  ret_ell->vars.push_back(ret_var);
  ret_ell->exprs.push_back(func_call);
  if (outs_.size() > 1) {
    // get the outputs TupleNode,
    // let %func_0 = %func_ret.0
    // let %func_1 = %func_ret.1
    // let %func_2 = %func_ret.2
    for (size_t i = 0; i < outs_.size(); ++i) {
      int index = i;
      String var_name = String(func_name_ + "_ret_" + std::to_string(index));
      TupleGetItem tgi = TupleGetItem(ret_var, index, {});
      Var tgi_var = MakeVar(var_name, {});
      ret_ell->vars.push_back(tgi_var);
      ret_ell->exprs.push_back(tgi);
      // placeholder ret
      CHECK(!intermediate_var_2_func_out.count(Downcast<Var>(outs_[i])))
          << "Duplicated output " << outs_[i];
      intermediate_var_2_func_out.Set(Downcast<Var>(outs_[i]), tgi_var);
    }
  } else {
    CHECK(!intermediate_var_2_func_out.count(Downcast<Var>(outs_[0])))
        << "Duplicated output " << outs_[0];
    intermediate_var_2_func_out.Set(Downcast<Var>(outs_[0]), ret_var);
  }
}

}  // namespace pass
}  // namespace raf
