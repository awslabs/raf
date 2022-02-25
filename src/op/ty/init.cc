/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/init.cc
 * \brief Typing of init operators
 */
#include <tvm/relay/type.h>
#include "raf/op_utils.h"
#include "raf/type.h"
#include "../schema/init.h"
#include "../declare/declare_utils.h"
#include "./utils.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;
using namespace schema;

Type InitOpInfer(const CallValues& value) {
  const auto* args = value->args.as<InitOpArgs>();
  CHECK(args != nullptr);
  Array<PrimExpr> shape = GetShapeExprFromValue(args->shape);
  DataType dtype = DataType(ir::String2DLDataType(args->dtype));
  return TensorType(shape, dtype);
}

RAF_OP_TYPE("raf.op.zeros", "Zeros", InitOpInfer);
RAF_OP_TYPE("raf.op.ones", "Ones", InitOpInfer);

Type OneHotInfer(const CallValues& value) {
  const auto* args = value->args.as<OneHotArgs>();
  CHECK(args != nullptr);
  TensorType indices = Downcast<TensorType>(GetType(args->indices));
  Array<PrimExpr> shape = indices->shape;
  CHECK_GE(args->depth, 0);
  shape.push_back(Integer(args->depth));
  DataType dtype = DataType(ir::String2DLDataType(args->dtype));
  return TensorType(shape, dtype);
}

RAF_OP_TYPE("raf.op.one_hot", "OneHot", OneHotInfer);

}  // namespace op
}  // namespace raf
