/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/backend/tvmjit/unary.cc
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
using op::CallValues;
using schema::UnaryArgs;

void UnaryNormalizer(TVMOpEnv* env) {
  CHECK_EQ(env->inputs.size(), 1U);
  CHECK_EQ(env->outputs.size(), 1U);
  DLTensor& x = env->inputs[0];
  DLTensor& y = env->outputs[0];
  env->shape_slots.resize(1);
  std::vector<int64_t>& shape = env->shape_slots[0];
  shape.resize(1);
  shape[0] = GetNumel(x);
  x.ndim = y.ndim = 1;
  x.shape = y.shape = dmlc::BeginPtr(shape);
  x.strides = y.strides = nullptr;
}

void UnaryTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  param_types->resize(1);
  param_types->at(0) = GetTensorType(env->inputs[0]);
}

MNM_TVMJIT(Copy, "mnm.op.copy", UnaryArgs, UnaryNormalizer, UnaryTyper);
MNM_TVMJIT(Abs, "mnm.op.abs", UnaryArgs, UnaryNormalizer, UnaryTyper);
MNM_TVMJIT(Ceil, "mnm.op.ceil", UnaryArgs, UnaryNormalizer, UnaryTyper);
MNM_TVMJIT(Floor, "mnm.op.floor", UnaryArgs, UnaryNormalizer, UnaryTyper);
MNM_TVMJIT(Log, "mnm.op.log", UnaryArgs, UnaryNormalizer, UnaryTyper);
MNM_TVMJIT(Cos, "mnm.op.cos", UnaryArgs, UnaryNormalizer, UnaryTyper);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
