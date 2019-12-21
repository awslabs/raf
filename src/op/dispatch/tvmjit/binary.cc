/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/binary.cc
 * \brief Binary operators bridged from TVM.
 */
#include <array>
#include "./tvmjit_utils.h"
#include "../../schema/ufunc.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using common::shape_utils::GetNumel;
using schema::BinaryUfuncArgs;
using schema::BinaryDxArgs;

Attrs BinaryNormalizer(TVMOpEnv* env, const BinaryUfuncArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x1),
      GetDLTensor(args->x2),
  };
  return Attrs();
}

void BinaryTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
      GetTensorType(env->inputs[1]),
  };
}

MNM_TVMJIT(Add, "mnm.op.add", BinaryUfuncArgs, BinaryNormalizer, BinaryTyper);
MNM_TVMJIT(Subtract, "mnm.op.subtract", BinaryUfuncArgs, BinaryNormalizer, BinaryTyper);
MNM_TVMJIT(Multiply, "mnm.op.multiply", BinaryUfuncArgs, BinaryNormalizer, BinaryTyper);

Attrs BinaryDxNormalizer(TVMOpEnv* env, const BinaryDxArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x1),
      GetDLTensor(args->x2),
      GetDLTensor(args->y),
      GetDLTensor(args->dy),
  };
  return Attrs();
}

void BinaryDxTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
      GetTensorType(env->inputs[1]),
      GetTensorType(env->inputs[2]),
      GetTensorType(env->inputs[3]),
  };
}

MNM_TVMJIT(AddDx, "mnm.op.add_dx", BinaryDxArgs, BinaryDxNormalizer, BinaryDxTyper);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
