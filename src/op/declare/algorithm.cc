/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/vision.cc
 * \brief Declaration of algorithm-specific operators
 */
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/tensor.h"
#include "../schema/algorithm.h"
#include "./declare_utils.h"

namespace raf {
namespace op {
namespace declare {

using namespace raf::op::schema;
using namespace raf::value;

RAF_OP_DECLARE("raf.op.argsort", [](const CallValues& call) {
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

RAF_OP_DECLARE("raf.op.sort", [](const CallValues& call) {
  const auto* args = call->args.as<SortArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  std::vector<int64_t> oshape(data->shape, data->shape + data->ndim);

  call->out = TensorValue::Assemble(/*dev=*/data->device,
                                    /*dtype=*/data->dtype,
                                    /*shape=*/oshape);
  call->device = data->device;
});

RAF_OP_DECLARE("raf.op.topk", [](const CallValues& call) {
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
}  // namespace raf
