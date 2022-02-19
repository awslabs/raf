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
 * \file ./src/op/dialect/tvm/nn.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <array>
#include "./tvm_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/loss.h"

namespace mnm {
namespace op {
namespace tvm_dialect {

using namespace mnm::ir;
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

MNM_TVM(smooth_l1_loss, SmoothL1Loss, LossArgs, LossSchema2Args, LossSchemaArgNames, GenericAttrs,
        LossHasher, kCommReduce);
MNM_TVM(smooth_l1_loss_dpred, SmoothL1LossDpred, LossArgs, LossSchema2Args, LossSchemaArgNames,
        GenericAttrs, LossHasher, kElemWise);
MNM_TVM(smooth_l1_loss_dtrue, SmoothL1LossDtrue, LossArgs, LossSchema2Args, LossSchemaArgNames,
        GenericAttrs, LossHasher, kElemWise);
MNM_TVM(nll_loss, NLLLoss, LossArgs, LossSchema2Args, LossSchemaArgNames, GenericAttrs, LossHasher,
        kCommReduce);
MNM_TVM(nll_loss_dpred, NllLossDpred, LossDtpArgs, LossDtpSchema2Args, LossDtpSchemaArgNames,
        GenericAttrs, GenericHasher, kElemWise);
MNM_TVM(nll_loss_dtrue, NllLossDtrue, LossDtpArgs, LossDtpSchema2Args, LossDtpSchemaArgNames,
        GenericAttrs, GenericHasher, kElemWise);
MNM_TVM(cross_entropy, CrossEntropy, LossArgs, LossSchema2Args, LossSchemaArgNames, GenericAttrs,
        LossHasher, kCommReduce);
MNM_TVM(cross_entropy_dpred, CrossEntropyDpred, LossArgs, LossSchema2Args, LossSchemaArgNames,
        GenericAttrs, LossHasher, kCommReduce);
MNM_TVM(cross_entropy_dtrue, CrossEntropyDtrue, LossArgs, LossSchema2Args, LossSchemaArgNames,
        GenericAttrs, LossHasher, kCommReduce);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace mnm
