/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/random.cc
 * \brief Typing of random operators
 */
#include <tvm/relay/type.h>
#include "raf/type.h"
#include "../schema/random.h"
#include "../declare/declare_utils.h"
#include "./utils.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;
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

RAF_OP_TYPE("raf.op.threefry_generate", "ThreefryGenerate", ThreefryGenerateInfer);

Type ThreefrySplitInfer(const CallValues& value) {
  const auto* args = value->args.as<ThreefrySplitArgs>();
  CHECK(args != nullptr);
  TensorType key = Downcast<TensorType>(GetType(args->key));
  auto new_key = TensorType(key->shape, key->dtype);
  auto new_subkey = TensorType(key->shape, key->dtype);
  return TupleType({new_key, new_subkey});
}

RAF_OP_TYPE("raf.op.threefry_split", "ThreefrySplit", ThreefrySplitInfer);

}  // namespace op
}  // namespace raf
