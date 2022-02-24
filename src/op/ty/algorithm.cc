/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/algorithm.cc
 * \brief Typing of algorithm operators
 */
#include <tvm/relay/type.h>
#include "raf/op_utils.h"
#include "raf/type.h"
#include "../schema/algorithm.h"
#include "./utils.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;
using namespace schema;

Type ArgsortInfer(const CallValues& value) {
  const auto* args = value->args.as<ArgsortArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  DataType dtype = DataType(ir::String2DLDataType(args->dtype));
  return TensorType(data->shape, dtype);
}

RAF_OP_TYPE("raf.op.argsort", "Argsort", ArgsortInfer);

Type SortInfer(const CallValues& value) {
  const auto* args = value->args.as<SortArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  return data;
}

RAF_OP_TYPE("raf.op.sort", "Sort", SortInfer);

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

RAF_OP_TYPE("raf.op.topk", "TopK", TopkInfer);

}  // namespace op
}  // namespace raf
