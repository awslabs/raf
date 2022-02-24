/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/tvm/nn.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <array>
#include "./tvm_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/loss.h"

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::ir;
using schema::LossArgs;
using schema::LossDtpArgs;

std::vector<Value> LossSchema2Args(const LossArgs* args) {
  return {args->y_true, args->y_pred};
}

std::vector<std::string> LossSchemaArgNames(const op::CallValues& call) {
  return {"y_true", "y_pred"};
}

std::vector<Value> LossDtpSchema2Args(const LossDtpArgs* args) {
  return {args->dy, args->y_true, args->y_pred};
}

std::vector<std::string> LossDtpSchemaArgNames(const op::CallValues& call) {
  return {"dy", "y_true", "y_pred"};
}

HashKey LossHasher(const std::vector<Type>& param_types, const Type& y_type, const LossArgs* args) {
  HashKey key;
  key << Downcast<TensorType>(param_types[0]);
  key << Downcast<TensorType>(param_types[1]);
  key << Downcast<TensorType>(y_type);
  return key;
}

RAF_TVM(smooth_l1_loss, SmoothL1Loss, LossArgs, LossSchema2Args, LossSchemaArgNames, GenericAttrs,
        LossHasher, kCommReduce);
RAF_TVM(smooth_l1_loss_dpred, SmoothL1LossDpred, LossArgs, LossSchema2Args, LossSchemaArgNames,
        GenericAttrs, LossHasher, kElemWise);
RAF_TVM(smooth_l1_loss_dtrue, SmoothL1LossDtrue, LossArgs, LossSchema2Args, LossSchemaArgNames,
        GenericAttrs, LossHasher, kElemWise);
RAF_TVM(nll_loss, NLLLoss, LossArgs, LossSchema2Args, LossSchemaArgNames, GenericAttrs, LossHasher,
        kCommReduce);
RAF_TVM(nll_loss_dpred, NllLossDpred, LossDtpArgs, LossDtpSchema2Args, LossDtpSchemaArgNames,
        GenericAttrs, GenericHasher, kElemWise);
RAF_TVM(nll_loss_dtrue, NllLossDtrue, LossDtpArgs, LossDtpSchema2Args, LossDtpSchemaArgNames,
        GenericAttrs, GenericHasher, kElemWise);
RAF_TVM(cross_entropy, CrossEntropy, LossArgs, LossSchema2Args, LossSchemaArgNames, GenericAttrs,
        LossHasher, kCommReduce);
RAF_TVM(cross_entropy_dpred, CrossEntropyDpred, LossArgs, LossSchema2Args, LossSchemaArgNames,
        GenericAttrs, LossHasher, kCommReduce);
RAF_TVM(cross_entropy_dtrue, CrossEntropyDtrue, LossArgs, LossSchema2Args, LossSchemaArgNames,
        GenericAttrs, LossHasher, kCommReduce);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
