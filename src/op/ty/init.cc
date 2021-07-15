/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/ty/init.cc
 * \brief Typing of init operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/init.h"
#include "../declare/declare_utils.h"
#include "./utils.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace mnm::value;
using namespace schema;

Type InitOpInfer(const CallValues& value) {
  const auto* args = value->args.as<InitOpArgs>();
  CHECK(args != nullptr);
  Array<PrimExpr> shape;
  for (auto& s : args->shape) {
    shape.push_back(Integer(s));
  }
  DataType dtype = DataType(ir::String2DLDataType(args->dtype));
  return TensorType(shape, dtype);
}

MNM_OP_TYPE("mnm.op.zeros", "Zeros", InitOpInfer);
MNM_OP_TYPE("mnm.op.ones", "Ones", InitOpInfer);

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

MNM_OP_TYPE("mnm.op.one_hot", "OneHot", OneHotInfer);

}  // namespace op
}  // namespace mnm
