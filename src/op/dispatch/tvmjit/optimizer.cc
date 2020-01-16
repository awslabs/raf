/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/nn.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <array>
#include "./tvmjit_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/optimizer.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using schema::SgdArgs;

struct SgdAttrs : public tvm::AttrsNode<SgdAttrs> {
  double mu;
  double learning_rate;
  // declare attribute fields in header file
  TVM_DECLARE_ATTRS(SgdAttrs, "attrs.SgdAttrs") {
    TVM_ATTR_FIELD(mu);
    TVM_ATTR_FIELD(learning_rate);
  }
};
TVM_REGISTER_NODE_TYPE(SgdAttrs);

Attrs SgdNormalizer(TVMOpEnv* env, const SgdArgs* args) {
  CHECK_EQ(env->outputs.size(), 2U);
  env->inputs.resize(3);
  env->inputs[0] = GetDLTensor(args->x);
  env->inputs[1] = GetDLTensor(args->dx);
  env->inputs[2] = GetDLTensor(args->v);
  auto attrs = make_object<SgdAttrs>();
  attrs->learning_rate = args->learning_rate;
  attrs->mu = args->mu;
  return Attrs(attrs);
}

void SgdTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  y_type[0] = GetTupleType({env->outputs[0], env->outputs[1]});
  *param_types = {
      GetTensorType(env->inputs[0]),
      GetTensorType(env->inputs[1]),
      GetTensorType(env->inputs[2]),
  };
}

HashKey SgdHasher(const std::vector<Type>& param_types,
               const Type &y_type,
               const SgdArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->mu;
  key << args->learning_rate;
  return key;
}

MNM_TVMJIT(OptimizerSgd, "mnm.op.sgd", SgdArgs, SgdNormalizer, SgdTyper, SgdHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
