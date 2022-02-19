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
