/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file anf_partition.cc
 * \brief Partition ANF program to be multiple functions.
 * Currently we simply make each partition equal or smaller to the given partition size.
 * We will introduce more partition strategies (e.g., optimize memory, etc) in the future.
 * e.g.,
 * fn(%x: Tensor[(10, 10), float32]) {
 *   let %a1 = mnm.op.relu(%x);
 *   let %a2 = mnm.op.abs(%a1);
 *   let %a3 = mnm.op.tanh(%a1);
 *   let %a4 = mnm.op.add(%a2, %a3, None, None);
 *   %a4
 * }
 * After partition with partition size 2:
 * fn(%x) {
 *   let %func_partition_0 = fn () {
 *     let %a1 = mnm.op.relu(%x);
 *     let %a2 = mnm.op.abs(%a1);
 *     let %func_partition_0_outs = (%a1, %a2);
 *     %func_partition_0_outs
 *   };
 *   let %func_partition_0_ret = %func_partition_0();
 *   let %func_partition_0_ret_0 = %func_partition_0_ret.0;
 *   let %func_partition_0_ret_1 = %func_partition_0_ret.1;
 *   let %func_partition_1 = fn () {
 *     let %a3 = mnm.op.tanh(%func_partition_0_ret_0);
 *     let %a4 = mnm.op.add(%func_partition_0_ret_1, %a3, None, None);
 *     %a4
 *   };
 *   let %func_partition_1_ret = %func_partition_1();
 *   %func_partition_1_ret
 * }
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/binding.h"
#include "mnm/pass.h"
#include "mnm/ir_ext.h"
#include <utility>
#include <vector>
#include "./common.h"
#include "./liveness_analysis.h"

namespace mnm {
namespace pass {
namespace anf_partition {

using namespace mnm::ir;
using namespace mnm::op;
using namespace mnm::binding;
using binding::BindingEntry;
using binding::BindNDArray;
using binding::LookupBinding;

class PartitionFunction {
 public:
  explicit PartitionFunction(std::string name) : func_name_(std::move(name)) {
  }

  /*!
   * \brief Add a non-annotated Expr to the Function,
   * find out the inputs and outputs of this partition function.
   * \param var The Var that the expr bind to.
   * \param expr The Expr to be pushed.
   */
  void Push(Var var, Expr expr) {
    // Push the inputs and outputs into ins_ and outs_.
    outs_.push_back(var);
    // Push the Expr into ell_.
    ell_.vars.push_back(var);
    ell_.exprs.push_back(expr);
  }

  /*!
   * \brief Remove the outputs which won't be used later.
   * \param analyzer LinvenessAnalyzer which has been run on the original program.
   * \param next_var The next variable follows the current PartitionFunction.
   * \param is_final_ret whether next_var is the final return output of the origianl program.
   */
  void TrimOutputs(liveness_analysis::LivenessAnalyzer& analyzer, const Var& next_var,
                   bool is_final_ret) {
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
  void ExportTo(ExplicitLetList* ret_ell, Map<Var, Var>& intermediate_var_2_func_out) {
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

  /*! \brief The function name of the partition function. */
  std::string func_name_;
  /*! \brief The LetNodes to construct the partition function. */
  ExplicitLetList ell_;
  /*! \brief The outputs of this partition function. */
  std::vector<Expr> outs_;
};

class Partitioner final : public ExprMutator {
 public:
  explicit Partitioner(int max_num_ops, Var boundary, liveness_analysis::LivenessAnalyzer& analyzer)
      : max_num_ops_(max_num_ops), boundary_(boundary), analyzer_(analyzer) {
  }

  Expr VisitExpr_(const LetNode* let_node) final {
    int num_visited = 0;
    int partition_id = 0;
    auto curr = GetRef<Expr>(let_node);
    PartitionFunction par_func("func_partition_" + std::to_string(partition_id++));
    ExplicitLetList ret_ell;
    Map<Var, Var> intermediate_var_2_func_out;
    while (!curr.as<VarNode>()) {
      const LetNode* let = curr.as<LetNode>();
      CHECK(let) << "Expect ANF (let-binding), but got " << curr->GetTypeKey();
      if (let->value.as<CallNode>() /* don't partition trivial statement */ &&
          num_visited >= max_num_ops_) {
        par_func.TrimOutputs(analyzer_, let->var, false);
        par_func.ExportTo(&ret_ell, intermediate_var_2_func_out);
        par_func = PartitionFunction("func_partition_" + std::to_string(partition_id++));
        num_visited = 0;
      }
      par_func.Push(let->var, let->value);
      num_visited++;
      curr = let->body;
      if (let->var == boundary_) {
        // the remains are just outputs combine.
        break;
      }
    }
    if (num_visited > 0) {
      if (curr.as<VarNode>()) {
        // return node
        par_func.TrimOutputs(analyzer_, Downcast<Var>(curr), true);
      } else {
        const LetNode* let = curr.as<LetNode>();
        CHECK(let) << "Expect ANF (let-binding), but got " << curr->GetTypeKey();
        par_func.TrimOutputs(analyzer_, let->var, false);
      }
      par_func.ExportTo(&ret_ell, intermediate_var_2_func_out);
    }
    // the remaining let statements which do not have any CallNode.
    VarSubstitutor substitutor(intermediate_var_2_func_out);
    while (!curr.as<VarNode>()) {
      const LetNode* let = curr.as<LetNode>();
      CHECK(let) << "Expect ANF (let-binding), but got " << curr->GetTypeKey();
      ret_ell.vars.push_back(let->var);
      ret_ell.exprs.push_back(substitutor.Substitute(let->value));
      curr = let->body;
    }
    auto ret_var = Downcast<Var>(curr);
    ret_ell.ret =
        intermediate_var_2_func_out.count(ret_var) ? intermediate_var_2_func_out[ret_var] : ret_var;
    return ret_ell.AsExpr();
  }

 private:
  /*! \brief max number of operations in a sub-function */
  int max_num_ops_{1};
  /*! \brief the variable defines the end of partition.
   * If the program combines a set of variables to be tuples in the end,
   * we don't have these operations included in a partitioned function,
   * just simply leave them there.
   */
  Var boundary_;
  /*! \brief linveness analyzer to get var liveness when traversing the graph */
  liveness_analysis::LivenessAnalyzer& analyzer_;
};

/**
 * \brief Find the boundary of the partitions, so to keep the rest IR unchanged.
 * It tries not to include tuples/tgis in the last partitioned function,
 * this helps to avoid an output of func1 being captured by func2 for nothing
 * but just to return as a final result, for example,
 * func(...) {
 *   out1 = ...;
 *   out2 = ...;    // <- this is the boundary.
 *   final_res = (out1, out2);
 *   return final_res;
 * }
 *
 * If we partition everything, the previous program might end-up as,
 * func(...) {
 *   func1() {
 *     return out1;
 *   }
 *   out1 = func1();
 *   func2() {
 *     out2 = ...
 *     return (out1, out2);  // out1 is captured for nothing but serving as a final output.
 *   }
 *   final_res = func2();
 *   return final_res;
 * }
 *
 * Instead it is much easier to do analysis later on the following program,
 * func(...) {
 *   func1() {
 *     return out1;
 *   }
 *   out1 = func1();
 *   func2() {
 *     return out2;
 *   }
 *   out2 = func2();
 *   final_res = (out1, out2);
 *   return final_res;
 * }
 *
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

}  // namespace anf_partition

Pass PartitionANF(int max_num_ops) {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    Var boundary = anf_partition::GetPartitionBoundary(f);
    auto analyzer = liveness_analysis::LivenessAnalyzer(f);
    analyzer.Run();
    anf_partition::Partitioner partitioner(max_num_ops, boundary, analyzer);
    return Downcast<Function>(partitioner(f));
  };
  return CreateMNMFunctionPass(pass_func, 0, "PartitionANF", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.PartitionANF").set_body_typed(PartitionANF);

}  // namespace pass
}  // namespace mnm
