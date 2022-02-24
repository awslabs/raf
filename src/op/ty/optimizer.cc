/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/optimizer.cc
 * \brief Typing of optimizer operators
 */
#include <tvm/relay/type.h>
#include "raf/type.h"
#include "../schema/optimizer.h"
#include "./utils.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;
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

RAF_OP_TYPE("raf.op.sgd", "Sgd", SgdInfer);

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

RAF_OP_TYPE("raf.op.lans", "Lans", LansInfer);

}  // namespace op
}  // namespace raf
