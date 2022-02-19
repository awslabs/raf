/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file ./src/op/dialect/tvm/nn.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <vector>
#include "./tvm_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/optimizer.h"

namespace mnm {
namespace op {
namespace tvm_dialect {

using namespace mnm::ir;
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

MNM_TVM(sgd, OptimizerSgd, SgdArgs, SgdSchema2Args, SgdSchemaArgNames, SgdSchema2Attrs, SgdHasher,
        kInjective);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace mnm
