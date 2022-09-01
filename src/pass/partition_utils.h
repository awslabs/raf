/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file partition_utils.h
 * \brief Utility functions for partitioning ANF into multiple functions. An
 * example can be found in the comments of anf_partition.cc.
 */

#include "raf/op.h"
#include "raf/ir_ext.h"
#include "raf/binding.h"
#include "raf/pass.h"
#include "raf/ir_ext.h"
#include <utility>
#include <vector>
#include <algorithm>
#include "./common.h"
#include "./liveness_analysis.h"

namespace raf {
namespace pass {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::binding;
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
                   bool is_final_ret);

  /*!
   * \brief Export the partition functions into ExplicitLetList.
   * \param part_func_vars The map from old vars into func_named vars
   * \return The ExplicitLetList with partition function packed.
   */
  void ExportTo(ExplicitLetList* ret_ell, Map<Var, Var>& intermediate_var_2_func_out);

  /*! \brief The function name of the partition function. */
  std::string func_name_;
  /*! \brief The LetNodes to construct the partition function. */
  ExplicitLetList ell_;
  /*! \brief The outputs of this partition function. */
  std::vector<Expr> outs_;
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
Var GetPartitionBoundary(Function f);

}  // namespace pass
}  // namespace raf
