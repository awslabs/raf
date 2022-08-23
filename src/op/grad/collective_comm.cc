/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/grad/collective_comm.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"
#include "raf/pass.h"
#include "raf/ir_ext.h"

namespace raf {
namespace op {
namespace grad {

using namespace raf::ir;

Array<Expr> AllToAllGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                         const Expr& dy) {
  static auto op_all_to_all = Op::Get("raf.op._all_to_all");
  if (orig_args[0]->checked_type_.defined()) {
    auto orig_arg_tt = Downcast<TupleType>(orig_args[0]->checked_type());
    if (orig_arg_tt->fields.size() == 1) {
      return {Tuple({Call(op_all_to_all, {Tuple({dy})})})};
    } else {
      return {Call(op_all_to_all, {dy})};
    }
  } else {
    // assume input tuple size is 1
    return {Tuple({Call(op_all_to_all, {Tuple({dy})})})};
  }
}

RAF_OP_GRAD("raf.op._all_to_all", AllToAllGrad);

}  // namespace grad
}  // namespace op
}  // namespace raf
