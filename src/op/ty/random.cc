/*!
 * Copyright (c) 2021 by Contributors
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
namespace type {

using namespace mnm::value;
using namespace schema;
using tvm::relay::Type;
using namespace tvm;
using namespace relay;

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

}  // namespace type
}  // namespace op
}  // namespace mnm
