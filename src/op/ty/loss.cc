/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/loss.cc
 * \brief Typing relations of loss operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/loss.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using namespace tvm;
using namespace tvm::relay;
using schema::LossArgs;

Type NLLLossInfer(const CallValues& value) {
  const auto* args = value->args.as<LossArgs>();
  CHECK(args != nullptr);
  TensorType pred = Downcast<TensorType>(GetType(args->y_pred));
  TensorType true_ = Downcast<TensorType>(GetType(args->y_true));
  CHECK(pred->shape.size() == 2 && true_->shape.size() == 2);
  CHECK(TypeCheckCompare(pred->shape[0], true_->shape[0], std::equal_to<int>()));
  CHECK(TypeCheckCompare(pred->shape[1], true_->shape[1], std::equal_to<int>()));
  Array<tvm::PrimExpr> oshape = {1};
  return TensorType(oshape, pred->dtype);
}

Type NLLLossBack(const CallValues& value) {
  const auto* args = value->args.as<LossArgs>();
  CHECK(args != nullptr);
  TensorType pred = Downcast<TensorType>(GetType(args->y_pred));
  TensorType true_ = Downcast<TensorType>(GetType(args->y_true));
  CHECK(pred->shape.size() == 2 && true_->shape.size() == 2);
  CHECK(TypeCheckCompare(pred->shape[0], true_->shape[0], std::equal_to<int>()));
  CHECK(TypeCheckCompare(pred->shape[1], true_->shape[1], std::equal_to<int>()));
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

}  // namespace type
}  // namespace op
}  // namespace mnm
