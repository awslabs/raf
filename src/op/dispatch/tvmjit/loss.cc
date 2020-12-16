/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/nn.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <array>
#include "./tvmjit_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/loss.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using schema::LossArgs;

std::vector<Value> LossSchema2Args(const LossArgs* args) {
  return {args->y_true, args->y_pred};
}

std::vector<std::string> LossSchemaArgNames() {
  return {"y_true", "y_pred"};
}

HashKey LossHasher(const std::vector<Type>& param_types, const Type& y_type, const LossArgs* args) {
  HashKey key;
  key << Downcast<TensorType>(param_types[0]);
  key << Downcast<TensorType>(param_types[1]);
  key << Downcast<TensorType>(y_type);
  return key;
}

MNM_TVMJIT(SmoothL1Loss, "mnm.op.smooth_l1_loss", LossArgs, LossSchema2Args, LossSchemaArgNames,
           GenericAttrs, LossHasher);
MNM_TVMJIT(SmoothL1LossDpred, "mnm.op.smooth_l1_loss_dpred", LossArgs, LossSchema2Args,
           LossSchemaArgNames, GenericAttrs, LossHasher);
MNM_TVMJIT(SmoothL1LossDtrue, "mnm.op.smooth_l1_loss_dtrue", LossArgs, LossSchema2Args,
           LossSchemaArgNames, GenericAttrs, LossHasher);
MNM_TVMJIT(NLLLoss, "mnm.op.nll_loss", LossArgs, LossSchema2Args, LossSchemaArgNames, GenericAttrs,
           LossHasher);
MNM_TVMJIT(NllLossDpred, "mnm.op.nll_loss_dpred", LossArgs, LossSchema2Args, LossSchemaArgNames,
           GenericAttrs, LossHasher);
MNM_TVMJIT(NllLossDtrue, "mnm.op.nll_loss_dtrue", LossArgs, LossSchema2Args, LossSchemaArgNames,
           GenericAttrs, LossHasher);
MNM_TVMJIT(CrossEntropy, "mnm.op.cross_entropy", LossArgs, LossSchema2Args, LossSchemaArgNames,
           GenericAttrs, LossHasher);
MNM_TVMJIT(CrossEntropyDpred, "mnm.op.cross_entropy_dpred", LossArgs, LossSchema2Args,
           LossSchemaArgNames, GenericAttrs, LossHasher);
MNM_TVMJIT(CrossEntropyDtrue, "mnm.op.cross_entropy_dtrue", LossArgs, LossSchema2Args,
           LossSchemaArgNames, GenericAttrs, LossHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
