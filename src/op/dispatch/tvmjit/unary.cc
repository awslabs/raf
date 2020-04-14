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

Attrs UnaryNormalizer(TVMOpEnv* env, const UnaryArgs* args) {
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

MNM_TVMJIT(Copy, "mnm.op.copy", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Abs, "mnm.op.abs", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Ceil, "mnm.op.ceil", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Floor, "mnm.op.floor", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Log, "mnm.op.log", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Cos, "mnm.op.cos", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Erf, "mnm.op.erf", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);
MNM_TVMJIT(Sqrt, "mnm.op.sqrt", UnaryArgs, UnaryNormalizer, UnaryTyper, GenericHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
