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

std::vector<Value> SgdSchema2Args(const SgdArgs* args) {
  return {args->x, args->dx, args->v};
}

std::vector<std::string> SgdSchemaArgNames(const op::CallValues& call) {
  return {"x", "dx", "v"};
}

Attrs SgdSchema2Attrs(const SgdArgs* args) {
  auto attrs = make_object<SgdAttrs>();
  attrs->learning_rate = args->learning_rate;
  attrs->mu = args->mu;
  return Attrs(attrs);
}

HashKey SgdHasher(const std::vector<Type>& param_types, const Type& y_type, const SgdArgs* args) {
  HashKey key = GenericHasher<std::nullptr_t>(param_types, y_type, nullptr);
  key << args->mu;
  key << args->learning_rate;
  return key;
}

MNM_TVMJIT(OptimizerSgd, "mnm.op.sgd", SgdArgs, SgdSchema2Args, SgdSchemaArgNames, SgdSchema2Attrs,
           SgdHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
