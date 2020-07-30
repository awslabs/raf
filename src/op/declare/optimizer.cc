/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/nn.cc
 * \brief Declaration of nn-specific operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/optimizer.h"
#include "./declare_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

MNM_OP_DECLARE("mnm.op.sgd", [](const CallValues& call) {
  const auto* args = call->args.as<SgdArgs>();
  CHECK(args != nullptr);
  const DLTensor* x0 = args->x;
  const DLTensor* dx = args->dx;
  const DLTensor* v0 = args->v;
  CHECK_EQ(x0->ndim, dx->ndim);
  CHECK_EQ(v0->ndim, dx->ndim);
  for (int i = 0; i < x0->ndim; ++i) {
    CHECK_EQ(x0->shape[i], dx->shape[i]);
    CHECK_EQ(v0->shape[i], dx->shape[i]);
  }
  auto v1 = TensorValue::Assemble(/*ctx=*/dx->ctx,
                                  /*dtype=*/dx->dtype,
                                  /*shape=*/std::vector<int64_t>(dx->shape, dx->shape + dx->ndim));
  auto x1 = TensorValue::Assemble(/*ctx=*/dx->ctx,
                                  /*dtype=*/dx->dtype,
                                  /*shape=*/std::vector<int64_t>(dx->shape, dx->shape + dx->ndim));
  call->out = TupleValue::make(tvm::Array<Value>({v1, x1}));
  call->ctx = dx->ctx;
}).set_attr<TOpPattern>("TOpPattern", kElemWise);

}  // namespace declare
}  // namespace op
}  // namespace mnm
