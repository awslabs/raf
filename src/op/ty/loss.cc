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
 * \file src/op/ty/loss.cc
 * \brief Typing relations of loss operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/loss.h"
#include "./utils.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using schema::LossArgs;
using schema::LossDtpArgs;

Type NLLLossInfer(const CallValues& value) {
  const auto* args = value->args.as<LossArgs>();
  CHECK(args != nullptr);
  TensorType pred = Downcast<TensorType>(GetType(args->y_pred));
  TensorType true_ = Downcast<TensorType>(GetType(args->y_true));
  CHECK(TypeCheckCompare(pred->shape[0], true_->shape[0], std::equal_to<int>()));
  Array<tvm::PrimExpr> oshape = {1};
  return TensorType(oshape, pred->dtype);
}

Type NLLLossBack(const CallValues& value) {
  const auto* args = value->args.as<LossDtpArgs>();
  CHECK(args != nullptr);
  TensorType pred = Downcast<TensorType>(GetType(args->y_pred));
  TensorType true_ = Downcast<TensorType>(GetType(args->y_true));
  CHECK(TypeCheckCompare(pred->shape[0], true_->shape[0], std::equal_to<int>()));
  /* pred and true_ share the same shape here */
  Array<tvm::PrimExpr> oshape = {pred->shape[0], pred->shape[1]};
  return TensorType(oshape, pred->dtype);
}

Type SmoothL1Infer(const CallValues& value) {
  const auto* args = value->args.as<LossArgs>();
  CHECK(args != nullptr);
  TensorType pred = Downcast<TensorType>(GetType(args->y_pred));
  TensorType true_ = Downcast<TensorType>(GetType(args->y_true));
  CHECK(pred->shape.size() == true_->shape.size());
  for (int i = 0; i < pred->shape.size(); i++)
    CHECK(TypeCheckCompare(pred->shape[i], true_->shape[i], std::equal_to<int>()));
  Array<tvm::PrimExpr> oshape = {1};
  return TensorType(oshape, pred->dtype);
}

Type SmoothL1Back(const CallValues& value) {
  const auto* args = value->args.as<LossArgs>();
  CHECK(args != nullptr);
  TensorType pred = Downcast<TensorType>(GetType(args->y_pred));
  TensorType true_ = Downcast<TensorType>(GetType(args->y_true));
  CHECK(pred->shape.size() == true_->shape.size());
  for (int i = 0; i < pred->shape.size(); i++)
    CHECK(TypeCheckCompare(pred->shape[i], true_->shape[i], std::equal_to<int>()));
  Array<tvm::PrimExpr> oshape = pred->shape;
  return TensorType(oshape, pred->dtype);
}

MNM_OP_TYPE("mnm.op.nll_loss", "NLLLoss", NLLLossInfer);
MNM_OP_TYPE("mnm.op.nll_loss_dpred", "NLLLossDpred", NLLLossBack);
MNM_OP_TYPE("mnm.op.nll_loss_dtrue", "NLLLossDtrue", NLLLossBack);
MNM_OP_TYPE("mnm.op.smooth_l1_loss", "SmoothL1", SmoothL1Infer);
MNM_OP_TYPE("mnm.op.smooth_l1_loss_dpred", "SmoothL1Dpred", SmoothL1Back);
MNM_OP_TYPE("mnm.op.smooth_l1_loss_dtrue", "SmoothL1Dtrue", SmoothL1Back);
MNM_OP_TYPE("mnm.op.cross_entropy", "CrossEntropy", SmoothL1Infer);
MNM_OP_TYPE("mnm.op.cross_entropy_dpred", "CrossEntropyDpred", SmoothL1Back);
MNM_OP_TYPE("mnm.op.cross_entropy_dtrue", "CrossEntropyDtrue", SmoothL1Back);

}  // namespace op
}  // namespace mnm
