/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file anf_partition.cc
 * \brief Partition ANF program to be multiple functions.
 * Currently we simply make each partition equal or smaller to the given partition size.
 * We will introduce more partition strategies (e.g., optimize memory, etc) in the future.
 * e.g.,
 * fn(%x: Tensor[(10, 10), float32]) {
 *   let %a1 = raf.op.relu(%x);
 *   let %a2 = raf.op.abs(%a1);
 *   let %a3 = raf.op.tanh(%a1);
 *   let %a4 = raf.op.add(%a2, %a3, None, None);
 *   %a4
 * }
 * After partition with partition size 2:
 * fn(%x) {
 *   let %func_partition_0 = fn () {
 *     let %a1 = raf.op.relu(%x);
 *     let %a2 = raf.op.abs(%a1);
 *     let %func_partition_0_outs = (%a1, %a2);
 *     %func_partition_0_outs
 *   };
 *   let %func_partition_0_ret = %func_partition_0();
 *   let %func_partition_0_ret_0 = %func_partition_0_ret.0;
 *   let %func_partition_0_ret_1 = %func_partition_0_ret.1;
 *   let %func_partition_1 = fn () {
 *     let %a3 = raf.op.tanh(%func_partition_0_ret_0);
 *     let %a4 = raf.op.add(%func_partition_0_ret_1, %a3, None, None);
 *     %a4
 *   };
 *   let %func_partition_1_ret = %func_partition_1();
 *   %func_partition_1_ret
 * }
 */
#include "raf/op.h"
#include "raf/ir_ext.h"
#include "raf/binding.h"
#include "raf/pass.h"
#include "raf/ir_ext.h"
#include <utility>
#include <vector>
#include "./common.h"
#include "./liveness_analysis.h"
#include "./partition_utils.h"

namespace raf {
namespace pass {
namespace anf_partition {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::binding;
using binding::BindingEntry;
using binding::BindNDArray;
using binding::LookupBinding;

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

}  // namespace anf_partition

Pass PartitionANF(int max_num_ops) {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    Var boundary = pass::GetPartitionBoundary(f);
    auto analyzer = liveness_analysis::LivenessAnalyzer(f);
    analyzer.Run();
    anf_partition::Partitioner partitioner(max_num_ops, boundary, analyzer);
    return Downcast<Function>(partitioner(f));
  };
  return CreateRAFFunctionPass(pass_func, 0, "PartitionANF", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.PartitionANF").set_body_typed(PartitionANF);

}  // namespace pass
}  // namespace raf
