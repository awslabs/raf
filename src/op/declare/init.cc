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
 * \file src/op/declare/unary.cc
 * \brief Declaration of unary operators
 */
#include "mnm/op.h"
#include "mnm/op_utils.h"
#include "mnm/tensor.h"
#include "../schema/init.h"

namespace mnm {
namespace op {
namespace init {

using namespace mnm::op::schema;
using namespace mnm::value;

MNM_OP_DECLARE("mnm.op.zeros", [](const CallValues& call) {
  const auto* args = call->args.as<InitOpArgs>();
  CHECK(args != nullptr);
  std::string dtype = args->dtype;
  std::vector<int64_t> shape = GetShapeVecFromValue(args->shape);
  const auto* f = tvm::runtime::Registry::Get("mnm._core.core_utils.str2dev");
  tvm::Device tvm_dev = (*f)(args->device);
  Device device(tvm_dev);
  call->out = TensorValue::Assemble(/*dev=*/device,
                                    /*dtype=*/ir::String2DLDataType(dtype),
                                    /*shape=*/shape);
  call->device = device;
});

MNM_OP_DECLARE("mnm.op.ones", [](const CallValues& call) {
  const auto* args = call->args.as<InitOpArgs>();
  CHECK(args != nullptr);
  std::string dtype = args->dtype;
  std::vector<int64_t> shape = GetShapeVecFromValue(args->shape);
  const auto* f = tvm::runtime::Registry::Get("mnm._core.core_utils.str2dev");
  tvm::Device tvm_dev = (*f)(args->device);
  Device device(tvm_dev);
  call->out = TensorValue::Assemble(/*dev=*/device,
                                    /*dtype=*/ir::String2DLDataType(dtype),
                                    /*shape=*/shape);
  call->device = device;
});

MNM_OP_DECLARE("mnm.op.one_hot", [](const CallValues& call) {
  const auto* args = call->args.as<OneHotArgs>();
  CHECK(args != nullptr);
  std::string dtype = args->dtype;
  DLTensor* indices = args->indices;
  std::vector<int64_t> shape(indices->shape, indices->shape + indices->ndim);
  CHECK_GE(args->depth, 0);
  shape.push_back(args->depth);
  const auto* f = tvm::runtime::Registry::Get("mnm._core.core_utils.str2dev");
  tvm::Device tvm_dev = (*f)(args->device);
  Device device(tvm_dev);
  call->out = TensorValue::Assemble(/*dev=*/device,
                                    /*dtype=*/ir::String2DLDataType(dtype),
                                    /*shape=*/shape);
  call->device = device;
});

}  // namespace init
}  // namespace op
}  // namespace mnm
