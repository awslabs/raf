/*!
 * Copyright (c) 2019 by Contributors
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
  call->out = TensorValue::Assemble(/*ctx=*/true_->ctx,
                                    /*dtype=*/true_->dtype,
                                    /*shape=*/{1});
  call->ctx = true_->ctx;
}).set_attr<TOpPattern>("TOpPattern", kCommReduce);

MNM_OP_DECLARE("mnm.op.smooth_l1_loss_dpred", [](const CallValues& call) {
  const auto* args = call->args.as<LossArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  std::vector<int64_t> shape(pred->shape, pred->shape + pred->ndim);
  call->out = TensorValue::Assemble(pred->ctx, pred->dtype, shape);
  call->ctx = pred->ctx;
}).set_attr<TOpPattern>("TOpPattern", kElemWise);

MNM_OP_DECLARE("mnm.op.smooth_l1_loss_dtrue", [](const CallValues& call) {
  const auto* args = call->args.as<LossArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  std::vector<int64_t> shape(true_->shape, true_->shape + true_->ndim);
  call->out = TensorValue::Assemble(true_->ctx, pred->dtype, shape);
  call->ctx = true_->ctx;
}).set_attr<TOpPattern>("TOpPattern", kElemWise);

MNM_OP_DECLARE("mnm.op.nll_loss", [](const CallValues& call) {
  const auto* args = call->args.as<LossArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  CHECK_EQ(pred->ndim, 2);
  CHECK_EQ(true_->ndim, 2);
  CHECK_EQ(pred->shape[0], true_->shape[0]);
  CHECK_EQ(pred->shape[1], true_->shape[1]);
  call->out = TensorValue::Assemble(/*ctx=*/true_->ctx,
                                    /*dtype=*/true_->dtype,
                                    /*shape=*/{1});
  call->ctx = true_->ctx;
}).set_attr<TOpPattern>("TOpPattern", kCommReduce);

MNM_OP_DECLARE("mnm.op.nll_loss_dpred", [](const CallValues& call) {
  const auto* args = call->args.as<LossArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  CHECK_EQ(pred->ndim, 2);
  CHECK_EQ(true_->ndim, 2);
  CHECK_EQ(pred->shape[0], true_->shape[0]);
  CHECK_EQ(pred->shape[1], true_->shape[1]);
  call->out = TensorValue::Assemble(pred->ctx, pred->dtype, {pred->shape[0], pred->shape[1]});
  call->ctx = pred->ctx;
}).set_attr<TOpPattern>("TOpPattern", kElemWise);

MNM_OP_DECLARE("mnm.op.nll_loss_dtrue", [](const CallValues& call) {
  const auto* args = call->args.as<LossArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  CHECK_EQ(pred->ndim, 2);
  CHECK_EQ(true_->ndim, 2);
  CHECK_EQ(pred->shape[0], true_->shape[0]);
  CHECK_EQ(pred->shape[1], true_->shape[1]);
  call->out = TensorValue::Assemble(true_->ctx, pred->dtype, {true_->shape[0], true_->shape[1]});
  call->ctx = true_->ctx;
}).set_attr<TOpPattern>("TOpPattern", kElemWise);

MNM_OP_DECLARE("mnm.op.cross_entropy", [](const CallValues& call) {
  const auto* args = call->args.as<LossArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  CHECK_EQ(pred->ndim, 2);
  CHECK_EQ(true_->ndim, 2);
  CHECK_EQ(pred->shape[0], true_->shape[0]);
  CHECK_EQ(pred->shape[1], true_->shape[1]);
  call->out = TensorValue::Assemble(/*ctx=*/true_->ctx,
                                    /*dtype=*/true_->dtype,
                                    /*shape=*/{1});
  call->ctx = true_->ctx;
}).set_attr<TOpPattern>("TOpPattern", kCommReduce);

MNM_OP_DECLARE("mnm.op.cross_entropy_dpred", [](const CallValues& call) {
  const auto* args = call->args.as<LossArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  CHECK_EQ(pred->ndim, 2);
  CHECK_EQ(true_->ndim, 2);
  CHECK_EQ(pred->shape[0], true_->shape[0]);
  CHECK_EQ(pred->shape[1], true_->shape[1]);
  call->out = TensorValue::Assemble(pred->ctx, pred->dtype, {pred->shape[0], pred->shape[1]});
  call->ctx = pred->ctx;
}).set_attr<TOpPattern>("TOpPattern", kCommReduce);

MNM_OP_DECLARE("mnm.op.cross_entropy_dtrue", [](const CallValues& call) {
  const auto* args = call->args.as<LossArgs>();
  CHECK(args != nullptr);
  const DLTensor* pred = args->y_pred;
  const DLTensor* true_ = args->y_true;
  CHECK_EQ(pred->ndim, 2);
  CHECK_EQ(true_->ndim, 2);
  CHECK_EQ(pred->shape[0], true_->shape[0]);
  CHECK_EQ(pred->shape[1], true_->shape[1]);
  call->out = TensorValue::Assemble(true_->ctx, pred->dtype, {true_->shape[0], true_->shape[1]});
  call->ctx = true_->ctx;
}).set_attr<TOpPattern>("TOpPattern", kCommReduce);

}  // namespace declare
}  // namespace op
}  // namespace mnm
