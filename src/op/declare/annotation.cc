/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/annotation.cc
 * \brief Declaration of annotation operators
 */
#include "raf/op.h"
#include "raf/tensor.h"
#include "../schema/ufunc.h"

namespace raf {
namespace op {
namespace declare {

using namespace raf::op::schema;
using namespace raf::value;

RAF_OP_DECLARE("raf.op.compiler_begin", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  call->out = TensorValue::Assemble(/*dev=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/std::vector<int64_t>(x->shape, x->shape + x->ndim));
  call->device = x->device;
});

RAF_OP_DECLARE("raf.op.compiler_end", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  call->out = TensorValue::Assemble(/*dev=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/std::vector<int64_t>(x->shape, x->shape + x->ndim));
  call->device = x->device;
});

}  // namespace declare
}  // namespace op
}  // namespace raf
