/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file estimate_flops.h
 * \brief Estimate the computation FLOPS of the given function.
 */
#pragma once

#include <tvm/ir/type_functor.h>
#include "raf/device.h"
#include "raf/op.h"
#include "raf/pass.h"

#include "./common.h"
#include "../op/dialect/tvm/tvm_utils.h"

namespace raf {
namespace pass {
namespace estimate_flops {

template <typename T>
using StdMap = std::unordered_map<Var, T, ObjectPtrHash, ObjectPtrEqual>;

/*!
 * \brief A visitor to traverse an ANF graph and esitmate the compute FLOPS of each let var
 * that binds to a call expression on the target device. Since we done that by analyzing
 * TVM defined arithmetic expression of an op, the FLOPS may not be accurate if the op is
 * actually dispatched to a non-TVM-dialect, but it should still be sufficient for
 * rematerialization to estimate the relative latency cost between tensors generated
 * by different ops.
 */
class FLOPSEstimater : public ExprVisitor {
 public:
  StdMap<float> Run(const Device& target, const Function& func, const IRModule& mod) {
    device_ = target;
    mod_ = mod;
    this->VisitExpr(func);
    return var_flops_map_;
  }

  float GetFLOPS(const Var& var) {
    if (var_flops_map_.count(var) == 0) {
      DLOG(WARNING) << "Var " << var->name_hint() << " does not have GFLOPS";
      return -1;
    }
    return var_flops_map_[var];
  }

  void VisitExpr_(const LetNode* op) override;
  void VisitExpr_(const CallNode* op) override;

 private:
  /*! \brief The current processing let var. */
  Var curr_let_;
  /*! \brief The IR module that the target function belongs to. This is used to look up
   * and estimate the FLOPS of other functions called by global symbols. */
  IRModule mod_;
  /*! \brief The target device used to estimate the rematerialization cost. */
  Device device_;
  /*! \brief Mapping from the let binding var to the GFLOPS of its expression. */
  StdMap<float> var_flops_map_;
};

}  // namespace estimate_flops
}  // namespace pass
}  // namespace raf
