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
 * \file src/op/grad/vision.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"
#include "mnm/pass.h"
#include "mnm/ir.h"

namespace mnm {
namespace op {
namespace grad {

using namespace mnm::ir;

Array<Expr> RoiAlignGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                         const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.roi_align_dx");
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

MNM_OP_GRAD("mnm.op.roi_align", RoiAlignGrad);

MNM_OP_GRAD("mnm.op.non_max_suppression", NoGrads<1>);

}  // namespace grad
}  // namespace op
}  // namespace mnm
