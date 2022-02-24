/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/tvm/nn.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <vector>
#include "./tvm_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/optimizer.h"

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::ir;
using schema::SgdArgs;

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

RAF_TVM(sgd, OptimizerSgd, SgdArgs, SgdSchema2Args, SgdSchemaArgNames, SgdSchema2Attrs, SgdHasher,
        kInjective);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
