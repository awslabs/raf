/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/grad/vision.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"
#include "raf/pass.h"
#include "raf/ir.h"

namespace raf {
namespace op {
namespace grad {

using namespace raf::ir;

Array<Expr> RoiAlignGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                         const Expr& dy) {
  static auto op_dx = Op::Get("raf.op.roi_align_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& data = call->args[0];
  const Expr& rois = call->args[1];
  const Expr& pooled_size = call->args[2];
  const Expr& spatial_scale = call->args[3];
  const Expr& sample_ratio = call->args[4];
  const Expr& layout = call->args[5];
  const Expr& mode = call->args[6];
  return {Call(op_dx, {data, rois, dy, pooled_size, spatial_scale, sample_ratio, layout, mode})};
}

RAF_OP_GRAD("raf.op.roi_align", RoiAlignGrad);

RAF_OP_GRAD("raf.op.non_max_suppression", NoGrads<1>);

}  // namespace grad
}  // namespace op
}  // namespace raf
