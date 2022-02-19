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
 * \file src/op/declare/vision.cc
 * \brief Declaration of algorithm-specific operators
 */
#include "mnm/op.h"
#include "mnm/op_utils.h"
#include "mnm/tensor.h"
#include "../schema/algorithm.h"
#include "./declare_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

MNM_OP_DECLARE("mnm.op.argsort", [](const CallValues& call) {
  const auto* args = call->args.as<ArgsortArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  std::vector<int64_t> oshape(data->shape, data->shape + data->ndim);
  std::string dtype = args->dtype;

  call->out = TensorValue::Assemble(/*dev=*/data->device,
                                    /*dtype=*/ir::String2DLDataType(dtype),
                                    /*shape=*/oshape);
  call->device = data->device;
});

MNM_OP_DECLARE("mnm.op.sort", [](const CallValues& call) {
  const auto* args = call->args.as<SortArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  std::vector<int64_t> oshape(data->shape, data->shape + data->ndim);

  call->out = TensorValue::Assemble(/*dev=*/data->device,
                                    /*dtype=*/data->dtype,
                                    /*shape=*/oshape);
  call->device = data->device;
});

MNM_OP_DECLARE("mnm.op.topk", [](const CallValues& call) {
  const auto* args = call->args.as<TopkArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  int64_t k = args->k.defined() ? GetScalarValueData<int64_t>(args->k) : 1;
  int axis = args->axis;
  CHECK((axis >= -data->ndim) && (axis < data->ndim));
  if (axis < 0) {
    axis += data->ndim;
  }
  CHECK(k <= data->shape[axis]);
  std::string ret_type = args->ret_type;
  CHECK(ret_type.compare("values") || ret_type.compare("indices") || ret_type.compare("both"));
  std::vector<int64_t> oshape = std::vector<int64_t>();
  for (int i = 0; i < data->ndim; i++) {
    if (axis == i) {
      oshape.push_back(k);
    } else {
      oshape.push_back(data->shape[i]);
    }
  }
  Value out_a = TensorValue::Assemble(/*dev=*/data->device,
                                      /*dtype=*/data->dtype,
                                      /*shape=*/oshape);
  Value out_b = TensorValue::Assemble(/*dev=*/data->device,
                                      /*dtype=*/ir::String2DLDataType(args->dtype),
                                      /*shape=*/oshape);
  if (ret_type == "both") {
    ir::Array<Value> field;
    field.push_back(out_a);
    field.push_back(out_b);
    call->out = TupleValue::make(field);
  } else if (ret_type == "values") {
    call->out = out_a;
  } else {
    call->out = out_b;
  }
  call->device = data->device;
});

}  // namespace declare
}  // namespace op
}  // namespace mnm
