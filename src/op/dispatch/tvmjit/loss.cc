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

Attrs LossNormalizer(TVMOpEnv* env, const LossArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs.resize(2);
  env->inputs[0] = GetDLTensor(args->y_true);
  env->inputs[1] = GetDLTensor(args->y_pred);
  return Attrs();
}

void LossTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
      GetTensorType(env->inputs[1]),
  };
}

HashKey LossHasher(const std::vector<Type>& param_types, const Type& y_type, const LossArgs *args) {
  HashKey key;
  key << Downcast<TensorType>(param_types[0]);
  key << Downcast<TensorType>(param_types[1]);
  key << Downcast<TensorType>(y_type);
  return key;
}

MNM_TVMJIT(NLLLoss, "mnm.op.nll_loss", LossArgs, LossNormalizer, LossTyper, LossHasher);
MNM_TVMJIT(NllLossDpred, "mnm.op.nll_loss_dpred", LossArgs, LossNormalizer, LossTyper, LossHasher);
MNM_TVMJIT(NllLossDtrue, "mnm.op.nll_loss_dtrue", LossArgs, LossNormalizer, LossTyper, LossHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
