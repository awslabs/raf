/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/binary.cc
 * \brief Binary operators bridged from TVM.
 */
#include <array>
#include <numeric>
#include "./tvm_attrs.h"
#include "./tvmjit_utils.h"
#include "../../schema/reduce.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using namespace mnm::op::schema;
using common::shape_utils::GetNumel;

// use tvm::relay::ReduceAttrs here

Attrs ReduceNormalizer(TVMOpEnv* env, const ReduceArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x),
  };
  auto attrs = make_object<tvm_attrs::ReduceAttrs>();
  for (int i = 0, n = args->axis.size(); i < n; ++i) {
    attrs->axis.push_back(args->axis[i]);
  }
  attrs->keepdims = args->keepdims;
  attrs->exclude = false;
  return Attrs(attrs);
}

void ReduceTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
  };
}

HashKey ReduceHasher(const std::vector<Type>& param_types,
                     const Type& ret_type,
                     const ReduceArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, ret_type, nullptr);
  for (int i = 0, n = args->axis.size(); i < n; ++i) {
    key << args->axis[i];
  }
  key << args->keepdims;
  return key;
}

Attrs MeanReduceNormalizer(TVMOpEnv* env, const ReduceArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x),
  };
  auto attrs = make_object<tvm_attrs::ReduceAttrs>();
  // expand the empty axis
  DLTensor *x = args->x;
  auto ndim = x->ndim;
  std::vector<int64_t> axis;
  if (args->axis.empty()) {
    axis.resize(ndim);
    std::iota(axis.begin(), axis.end(), 0);
  } else {
    axis = args->axis;
  }
  for (int i = 0, n = axis.size(); i < n; ++i) {
    attrs->axis.push_back(axis[i]);
  }
  attrs->keepdims = args->keepdims;
  attrs->exclude = false;
  return Attrs(attrs);
}

Attrs ReduceDxNormalizer(TVMOpEnv* env, const ReduceDxArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x), GetDLTensor(args->y), GetDLTensor(args->dy),
  };
  auto attrs = make_object<tvm_attrs::ReduceAttrs>();
  // expand the empty axis
  DLTensor *x = args->x;
  auto ndim = x->ndim;
  std::vector<int64_t> axis;
  if (args->axis.empty()) {
    axis.resize(ndim);
    std::iota(axis.begin(), axis.end(), 0);
  } else {
    axis = args->axis;
  }
  for (int i = 0, n = axis.size(); i < n; ++i) {
    attrs->axis.push_back(axis[i]);
  }
  attrs->keepdims = args->keepdims;
  attrs->exclude = false;
  return Attrs(attrs);
}

void ReduceDxTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]), GetTensorType(env->inputs[1]), GetTensorType(env->inputs[2]),
  };
}

HashKey ReduceDxHasher(const std::vector<Type>& param_types,
                     const Type& ret_type,
                     const ReduceDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, ret_type, nullptr);
  for (int i = 0, n = args->axis.size(); i < n; ++i) {
    key << args->axis[i];
  }
  key << args->keepdims;
  return key;
}

MNM_TVMJIT(Argmax, "mnm.op.argmax", ReduceArgs, ReduceNormalizer, ReduceTyper, ReduceHasher);
MNM_TVMJIT(Argmin, "mnm.op.argmin", ReduceArgs, ReduceNormalizer, ReduceTyper, ReduceHasher);
MNM_TVMJIT(All, "mnm.op.all", ReduceArgs, ReduceNormalizer, ReduceTyper, ReduceHasher);
MNM_TVMJIT(Any, "mnm.op.any", ReduceArgs, ReduceNormalizer, ReduceTyper, ReduceHasher);
MNM_TVMJIT(Mean, "mnm.op.mean", ReduceArgs, MeanReduceNormalizer, ReduceTyper, ReduceHasher);
MNM_TVMJIT(MeanDx, "mnm.op.mean_dx", ReduceDxArgs, ReduceDxNormalizer, ReduceDxTyper, ReduceDxHasher); //NOLINT

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
