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
 * \file src/op/declare/memory.cc
 * \brief Declaration of memory-related operators.
 */
#include <string>
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "mnm/op_utils.h"
#include "../schema/memory.h"
#include "../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;
using namespace mnm::common::shape_utils;

MNM_OP_DECLARE("mnm.op.device_copy", [](const CallValues& call) {
  const auto* args = call->args.as<DeviceCopyArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  std::vector<int64_t> shape(data->shape, data->shape + data->ndim);

  const auto* str2dev = tvm::runtime::Registry::Get("mnm._core.core_utils.str2dev");
  auto data_device = Device((tvm::Device)(*str2dev)(data->device));
  auto src_device = Device((tvm::Device)(*str2dev)(args->src_device));
  auto dst_device = Device((tvm::Device)(*str2dev)(args->dst_device));
  CHECK(data_device == src_device);

  call->out = TensorValue::Assemble(/*dev=*/dst_device,
                                    /*dtype=*/data->dtype,
                                    /*shape=*/shape);
  call->device = dst_device;
}).set_attr<TOpPattern>("TOpPattern", kOpaque);

MNM_OP_DECLARE("mnm.op.fuse_tensor", [](const CallValues& call) {
  const auto* args = call->args.as<FuseTensorArgs>();
  CHECK(args != nullptr);
  auto& tv = args->data;
  const DLTensor* x = tv[0];
  call->device = x->device;
  int64_t total_size = 0;
  for (int i = 0; i < tv.size(); ++i) {
    const DLTensor* x = tv[i];
    total_size += BytesCompactTensor(*x) / ((x->dtype.bits + 7) / 8);
  }
  if (tv.size() == 0) {
    call->callee = ir::NullValue<OpValue>();
  } else {
    std::vector<int64_t> shape = {total_size};
    ir::Array<Value> ret;
    call->out = TensorValue::Assemble(/*dev=*/x->device,
                                      /*dtype=*/x->dtype,
                                      /*shape=*/shape);
  }
}).set_attr<TOpPattern>("TOpPattern", kOpaque);

MNM_OP_DECLARE("mnm.op.defuse_tensor", [](const CallValues& call) {
  const auto* args = call->args.as<DefuseTensorArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->data;
  const std::vector<int64_t>& sizes = args->sizes;
  const std::vector<int64_t>& shape_indices = args->shape_indices;
  const std::vector<int64_t>& shapes = args->shapes;
  call->device = x->device;
  ir::Array<Value> ret;
  size_t total_size = 0;
  for (int i = 0; i < sizes.size(); ++i) {
    total_size += sizes[i];
  }
  CHECK(total_size == (int64_t)(BytesCompactTensor(*x) / ((x->dtype.bits + 7) / 8)))
      << "Input tensor size should be " << total_size << ", got "
      << (int64_t)(BytesCompactTensor(*x) / ((x->dtype.bits + 7) / 8)) << ".";
  int64_t start_idx = 0;
  for (int i = 0; i < shape_indices.size(); ++i) {
    std::vector<int64_t> shape(shapes.begin() + start_idx, shapes.begin() + shape_indices[i]);
    ret.push_back(TensorValue::Assemble(/*dev=*/x->device,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape));
    start_idx = shape_indices[i];
  }
  if (ret.size() == 0) {
    call->callee = ir::NullValue<OpValue>();
  } else {
    call->out = TupleValue::make(ret);
  }
}).set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace declare
}  // namespace op
}  // namespace mnm
