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
 * \file src/op/ty/optimizer.cc
 * \brief Typing of optimizer operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/optimizer.h"
#include "./utils.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace mnm::value;
using namespace schema;

Type SgdInfer(const CallValues& value) {
  const auto* args = value->args.as<SgdArgs>();
  CHECK(args != nullptr);
  TensorType x0 = Downcast<TensorType>(GetType(args->x));
  TensorType dx = Downcast<TensorType>(GetType(args->dx));
  TensorType v0 = Downcast<TensorType>(GetType(args->v));
  CHECK_EQ(x0->shape.size(), dx->shape.size());
  CHECK_EQ(v0->shape.size(), dx->shape.size());
  for (size_t i = 0; i < dx->shape.size(); ++i) {
    CHECK(TypeCheckCompare(x0->shape[i], dx->shape[i], std::equal_to<int>()));
    CHECK(TypeCheckCompare(v0->shape[i], dx->shape[i], std::equal_to<int>()));
  }
  Array<Type> res;
  res.push_back(v0);
  res.push_back(x0);
  return TupleType(res);
}

MNM_OP_TYPE("mnm.op.sgd", "Sgd", SgdInfer);

Type LansInfer(const CallValues& value) {
  const auto* args = value->args.as<LansArgs>();
  CHECK(args != nullptr);
  int ntensors = args->tensor_list.size() / 4;
  CHECK(args->tensor_list.size() % 4 == 0);
  Array<Type> res;
  for (int i = 0; i < args->tensor_list.size(); ++i) {
    res.push_back(Downcast<TensorType>(GetType(args->tensor_list[i])));
  }
  return TupleType(res);
}

MNM_OP_TYPE("mnm.op.lans", "Lans", LansInfer);

}  // namespace op
}  // namespace mnm
