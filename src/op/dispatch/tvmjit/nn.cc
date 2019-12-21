/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/nn.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <array>
#include "./tvmjit_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/nn.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using schema::BiasAddArgs;
using schema::BiasAddDbArgs;
using tvm_attrs::BiasAddAttrs;

Attrs BiasAddNormalizer(TVMOpEnv* env, const BiasAddArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs.resize(2);
  env->inputs[0] = GetDLTensor(args->x);
  env->inputs[1] = GetDLTensor(args->b);
  auto attrs = make_object<BiasAddAttrs>();
  attrs->axis = args->axis;
  return Attrs(attrs);
}

void BiasAddTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
    GetTensorType(env->inputs[0]),
    GetTensorType(env->inputs[1]),
  };
}

MNM_TVMJIT(BiasAdd, "mnm.op.bias_add", BiasAddArgs, BiasAddNormalizer, BiasAddTyper);

Attrs BiasAddDbNormalizer(TVMOpEnv* env, const BiasAddDbArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs.resize(2);
  env->inputs[0] = GetDLTensor(args->dy);
  env->inputs[1] = GetDLTensor(args->b);
  auto attrs = make_object<BiasAddAttrs>();
  attrs->axis = args->axis;
  return Attrs(attrs);
}

MNM_TVMJIT(BiasAddDb, "mnm.op.bias_add_db", BiasAddDbArgs, BiasAddDbNormalizer, BiasAddTyper);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
