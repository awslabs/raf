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
 * \file src/op/ty/random.cc
 * \brief Typing of random operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/random.h"
#include "../declare/declare_utils.h"
#include "./utils.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace mnm::value;
using namespace schema;

Type ThreefryGenerateInfer(const CallValues& value) {
  const auto* args = value->args.as<ThreefryGenerateArgs>();
  CHECK(args != nullptr);
  TensorType key = Downcast<TensorType>(GetType(args->key));
  Array<PrimExpr> shape;
  for (auto& s : args->shape) {
    shape.push_back(Integer(s));
  }
  auto new_key = TensorType(key->shape, key->dtype);
  auto random_array = TensorType(shape, key->dtype);
  return TupleType({new_key, random_array});
}

MNM_OP_TYPE("mnm.op.threefry_generate", "ThreefryGenerate", ThreefryGenerateInfer);

Type ThreefrySplitInfer(const CallValues& value) {
  const auto* args = value->args.as<ThreefrySplitArgs>();
  CHECK(args != nullptr);
  TensorType key = Downcast<TensorType>(GetType(args->key));
  auto new_key = TensorType(key->shape, key->dtype);
  auto new_subkey = TensorType(key->shape, key->dtype);
  return TupleType({new_key, new_subkey});
}

MNM_OP_TYPE("mnm.op.threefry_split", "ThreefrySplit", ThreefrySplitInfer);

}  // namespace op
}  // namespace mnm
