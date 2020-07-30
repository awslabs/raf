/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/unary.cc
 * \brief Declaration of unary operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/init.h"

namespace mnm {
namespace op {
namespace init {

using namespace mnm::op::schema;
using namespace mnm::value;

void Ones(const CallValues& call) {
  const auto* args = call->args.as<ShapeDtypeArgs>();
  CHECK(args != nullptr);
  const std::vector<int64_t>& shape = args->shape;
  // TODO(@junrushao1994): hacky here
  call->ctx = Context(DevType::kCUDA(), 0);
  call->out = TensorValue::Assemble(call->ctx, DType(DTypeCode::kFloat(), 32), shape);
}

MNM_OP_DECLARE("mnm.op.ones", Ones).set_attr<TOpPattern>("TOpPattern", kElemWise);

}  // namespace init
}  // namespace op
}  // namespace mnm
