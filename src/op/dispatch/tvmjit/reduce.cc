/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/binary.cc
 * \brief Binary operators bridged from TVM.
 */
#include <array>
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

MNM_TVMJIT(Argmax, "mnm.op.argmax", ReduceArgs, ReduceNormalizer, ReduceTyper, ReduceHasher);
MNM_TVMJIT(Argmin, "mnm.op.argmin", ReduceArgs, ReduceNormalizer, ReduceTyper, ReduceHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
