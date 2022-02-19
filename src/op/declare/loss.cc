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
 * \file src/op/declare/loss.cc
 * \brief Declaration of loss-specific operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/loss.h"
#include "./declare_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

MNM_OP_DECLARE("mnm.op.smooth_l1_loss", [](const CallValues& call) {
  const auto* args = call->args.as<LossArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  CHECK(pred->ndim == true_->ndim);
  for (int i = 0; i < pred->ndim; i++) CHECK_EQ(pred->shape[i], true_->shape[i]);
  call->out = TensorValue::Assemble(/*dev=*/true_->device,
                                    /*dtype=*/true_->dtype,
                                    /*shape=*/{1});
  call->device = true_->device;
});

MNM_OP_DECLARE("mnm.op.smooth_l1_loss_dpred", [](const CallValues& call) {
  const auto* args = call->args.as<LossArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  std::vector<int64_t> shape(pred->shape, pred->shape + pred->ndim);
  call->out = TensorValue::Assemble(pred->device, pred->dtype, shape);
  call->device = pred->device;
});

MNM_OP_DECLARE("mnm.op.smooth_l1_loss_dtrue", [](const CallValues& call) {
  const auto* args = call->args.as<LossArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  std::vector<int64_t> shape(true_->shape, true_->shape + true_->ndim);
  call->out = TensorValue::Assemble(true_->device, pred->dtype, shape);
  call->device = true_->device;
});

MNM_OP_DECLARE("mnm.op.nll_loss", [](const CallValues& call) {
  const auto* args = call->args.as<LossArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  CHECK_EQ(pred->ndim, 2);
  CHECK_EQ(pred->shape[0], true_->shape[0]);
  call->out = TensorValue::Assemble(/*dev=*/pred->device,
                                    /*dtype=*/pred->dtype,
                                    /*shape=*/{1});
  call->device = pred->device;
});

MNM_OP_DECLARE("mnm.op.nll_loss_dpred", [](const CallValues& call) {
  const auto* args = call->args.as<LossDtpArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  CHECK_EQ(pred->ndim, 2);
  CHECK_EQ(pred->shape[0], true_->shape[0]);
  call->out = TensorValue::Assemble(pred->device, pred->dtype,
                                    std::vector<int64_t>{pred->shape[0], pred->shape[1]});
  call->device = pred->device;
});

MNM_OP_DECLARE("mnm.op.nll_loss_dtrue", [](const CallValues& call) {
  const auto* args = call->args.as<LossDtpArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  CHECK_EQ(pred->ndim, 2);
  CHECK_EQ(true_->ndim, 2);
  CHECK_EQ(pred->shape[0], true_->shape[0]);
  CHECK_EQ(pred->shape[1], true_->shape[1]);
  call->out = TensorValue::Assemble(true_->device, pred->dtype,
                                    std::vector<int64_t>{true_->shape[0], true_->shape[1]});
  call->device = true_->device;
});

MNM_OP_DECLARE("mnm.op.cross_entropy", [](const CallValues& call) {
  const auto* args = call->args.as<LossArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  CHECK_EQ(pred->ndim, 2);
  CHECK_EQ(true_->ndim, 2);
  CHECK_EQ(pred->shape[0], true_->shape[0]);
  CHECK_EQ(pred->shape[1], true_->shape[1]);
  call->out = TensorValue::Assemble(/*dev=*/true_->device,
                                    /*dtype=*/true_->dtype,
                                    /*shape=*/std::vector<int64_t>{1});
  call->device = true_->device;
});

MNM_OP_DECLARE("mnm.op.cross_entropy_dpred", [](const CallValues& call) {
  const auto* args = call->args.as<LossArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  CHECK_EQ(pred->ndim, 2);
  CHECK_EQ(true_->ndim, 2);
  CHECK_EQ(pred->shape[0], true_->shape[0]);
  CHECK_EQ(pred->shape[1], true_->shape[1]);
  call->out = TensorValue::Assemble(pred->device, pred->dtype,
                                    std::vector<int64_t>{pred->shape[0], pred->shape[1]});
  call->device = pred->device;
});

MNM_OP_DECLARE("mnm.op.cross_entropy_dtrue", [](const CallValues& call) {
  const auto* args = call->args.as<LossArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  CHECK_EQ(pred->ndim, 2);
  CHECK_EQ(true_->ndim, 2);
  CHECK_EQ(pred->shape[0], true_->shape[0]);
  CHECK_EQ(pred->shape[1], true_->shape[1]);
  call->out = TensorValue::Assemble(true_->device, pred->dtype,
                                    std::vector<int64_t>{true_->shape[0], true_->shape[1]});
  call->device = true_->device;
});

}  // namespace declare
}  // namespace op
}  // namespace mnm
