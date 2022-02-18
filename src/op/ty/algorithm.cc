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
 * \file src/op/ty/algorithm.cc
 * \brief Typing of algorithm operators
 */
#include <tvm/relay/type.h>
#include "mnm/op_utils.h"
#include "mnm/type.h"
#include "../schema/algorithm.h"
#include "./utils.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace mnm::value;
using namespace schema;

Type ArgsortInfer(const CallValues& value) {
  const auto* args = value->args.as<ArgsortArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  DataType dtype = DataType(ir::String2DLDataType(args->dtype));
  return TensorType(data->shape, dtype);
}

MNM_OP_TYPE("mnm.op.argsort", "Argsort", ArgsortInfer);

Type SortInfer(const CallValues& value) {
  const auto* args = value->args.as<SortArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  return data;
}

MNM_OP_TYPE("mnm.op.sort", "Sort", SortInfer);

Type TopkInfer(const CallValues& value) {
  const auto* args = value->args.as<TopkArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  PrimExpr k = args->k.defined() ? GetIntExprFromValue(args->k) : 1;
  int axis = args->axis;
  if (axis < 0) {
    axis += (int)(data->shape).size();
  }
  std::string ret_type = args->ret_type;
  DataType dtype_a = data->dtype;
  DataType dtype_b = DataType(ir::String2DLDataType(args->dtype));
  Array<tvm::PrimExpr> oshape;
  for (int i = 0; i < (int)(data->shape).size(); i++) {
    if (axis == i) {
      oshape.push_back(k);
    } else {
      oshape.push_back(data->shape[i]);
    }
  }
  if (ret_type == "both") {
    Array<Type> fields;
    fields.push_back(TensorType(oshape, dtype_a));
    fields.push_back(TensorType(oshape, dtype_b));
    return TupleType(fields);
  } else if (ret_type == "values") {
    return TensorType(oshape, dtype_a);
  }
  return TensorType(oshape, dtype_b);
}

MNM_OP_TYPE("mnm.op.topk", "TopK", TopkInfer);

}  // namespace op
}  // namespace mnm
