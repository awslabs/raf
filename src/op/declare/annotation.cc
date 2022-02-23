/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/annotation.cc
 * \brief Declaration of annotation operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/annotation.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

MNM_OP_DECLARE("mnm.op.compiler_begin", [](const CallValues& call) {
  const auto* args = call->args.as<CompilerArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  call->out = TensorValue::Assemble(/*dev=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/std::vector<int64_t>(x->shape, x->shape + x->ndim));
  call->device = x->device;
});

MNM_OP_DECLARE("mnm.op.compiler_end", [](const CallValues& call) {
  const auto* args = call->args.as<CompilerArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  call->out = TensorValue::Assemble(/*dev=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/std::vector<int64_t>(x->shape, x->shape + x->ndim));
  call->device = x->device;
});

}  // namespace declare
}  // namespace op
}  // namespace mnm
