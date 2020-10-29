/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/optimizer.cc
 * \brief Typing of optimizer operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/optimizer.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using namespace mnm::value;
using namespace schema;
using tvm::relay::Type;
using namespace tvm;
using namespace tvm::relay;

Type SgdInfer(const CallValues& value) {
  const auto* args = value->args.as<SgdArgs>();
  CHECK(args != nullptr);
  TensorType x0 = Downcast<TensorType>(GetType(args->x));
  TensorType dx = Downcast<TensorType>(GetType(args->dx));
  TensorType v0 = Downcast<TensorType>(GetType(args->v));
  CHECK_EQ(x0->shape.size(), dx->shape.size());
  CHECK_EQ(v0->shape.size(), dx->shape.size());
  for (size_t i = 0; i < dx->shape.size(); ++i) {
    CHECK(TypeCheckEqual(x0->shape[i], dx->shape[i]));
    CHECK(TypeCheckEqual(v0->shape[i], dx->shape[i]));
  }
  Array<Type> res;
  res.push_back(v0);
  res.push_back(x0);
  return TupleType(res);
}

MNM_OP_TYPE("mnm.op.sgd", "Sgd", SgdInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
