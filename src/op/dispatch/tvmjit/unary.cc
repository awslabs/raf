/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/unary.cc
 * \brief Unary operators bridged from TVM.
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
using schema::UnaryArgs;
using schema::UnaryDxArgs;
using schema::UnaryUfuncArgs;

template <typename T>
Attrs UnaryNormalizer(TVMOpEnv* env, const T* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs.resize(1);
  env->shape_slots.resize(1);
  DLTensor& x = env->inputs[0] = GetDLTensor(args->x);
  DLTensor& y = env->outputs[0];
  std::vector<int64_t>& shape = env->shape_slots[0] = {GetNumel(x)};
  x.ndim = y.ndim = 1;
  x.shape = y.shape = dmlc::BeginPtr(shape);
  x.strides = y.strides = nullptr;
  return Attrs();
}

void UnaryTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {GetTensorType(env->inputs[0])};
}

Attrs UnaryDxNormalizer(TVMOpEnv* env, const UnaryDxArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x),
      GetDLTensor(args->dy),
      GetDLTensor(args->y),
  };
  return Attrs();
}

void UnaryDxTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
      GetTensorType(env->inputs[1]),
      GetTensorType(env->inputs[2]),
  };
}

MNM_TVMJIT(Copy, "mnm.op.copy", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Abs, "mnm.op.abs", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Ceil, "mnm.op.ceil", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Floor, "mnm.op.floor", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Log, "mnm.op.log", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Exp, "mnm.op.exp", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Cos, "mnm.op.cos", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Relu, "mnm.op.relu", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(ReluDx, "mnm.op.relu_dx", UnaryDxArgs, UnaryDxNormalizer, UnaryDxTyper, GenericHasher);
MNM_TVMJIT(Erf, "mnm.op.erf", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(ErfDx, "mnm.op.erf_dx", UnaryDxArgs, UnaryDxNormalizer, UnaryDxTyper, GenericHasher);
MNM_TVMJIT(Sqrt, "mnm.op.sqrt", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Atan, "mnm.op.atan", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Negative, "mnm.op.negative", UnaryUfuncArgs, UnaryNormalizer, UnaryTyper, GenericHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm