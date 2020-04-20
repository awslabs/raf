/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/nn.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <array>
#include "./tvmjit_utils.h"
#include "../../schema/ufunc.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using schema::BinaryArgs;

Attrs GEMMNormalizer(TVMOpEnv* env, const BinaryArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x1),
      GetDLTensor(args->x2),
  };
  return Attrs();
}

void GEMMTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
      GetTensorType(env->inputs[1]),
  };
}

MNM_TVMJIT(BatchMatmul, "mnm.op.batch_matmul", BinaryArgs, GEMMNormalizer, GEMMTyper,
           GenericHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
