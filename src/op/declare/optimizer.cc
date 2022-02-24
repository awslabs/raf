/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/nn.cc
 * \brief Declaration of nn-specific operators
 */
#include "raf/op.h"
#include "raf/tensor.h"
#include "../schema/optimizer.h"
#include "./declare_utils.h"

namespace raf {
namespace op {
namespace declare {

using namespace raf::op::schema;
using namespace raf::value;

RAF_OP_DECLARE("raf.op.sgd", [](const CallValues& call) {
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
  auto v1 = TensorValue::Assemble(
      /*dev=*/dx->device,
      /*dtype=*/dx->dtype,
      /*shape=*/std::vector<int64_t>(dx->shape, dx->shape + dx->ndim));
  auto x1 = TensorValue::Assemble(
      /*dev=*/dx->device,
      /*dtype=*/dx->dtype,
      /*shape=*/std::vector<int64_t>(dx->shape, dx->shape + dx->ndim));
  call->out = TupleValue::make(tvm::Array<Value>({v1, x1}));
  call->device = dx->device;
});

void LansDecl(const CallValues& call) {
  const auto* args = call->args.as<LansArgs>();
  CHECK(args != nullptr);
  CHECK(args->tensor_list.size() % 4 == 0);
  const DLTensor* x = args->tensor_list[0];
  call->device = x->device;
  int ntensors = args->tensor_list.size() / 4;
  Array<Value> output;
  for (int i = 0; i < args->tensor_list.size(); ++i) {
    output.push_back(args->tensor_list[i]);
  }
  call->out = TupleValue::make(output);
}

RAF_OP_DECLARE("raf.op.lans", LansDecl)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFInplaceUpdate>("TRAFInplaceUpdate", {{0, 0}});
}  // namespace declare
}  // namespace op
}  // namespace raf
