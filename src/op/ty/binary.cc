/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/binary.cc
 * \brief Typing of binary operators
 */
#include <tvm/relay/type.h>
#include <tvm/tir/op.h>
#include "raf/type.h"
#include "../schema/ufunc.h"
#include "./utils.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::op::schema;

Type BroadcastInfer(const CallValues& value) {
  const auto* args = value->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  TensorType x1 = Downcast<TensorType>(GetType(args->x1));
  TensorType x2 = Downcast<TensorType>(GetType(args->x2));
  CHECK_EQ(x1->dtype, x2->dtype) << "Data types mismatch (" << x1->dtype << " vs " << x2->dtype
                                 << ")";
  Array<PrimExpr> oshape = BroadcastShape(x1, x2);
  return TensorType(oshape, x1->dtype);
}

Type BroadcastUfuncInfer(const CallValues& value) {
  const auto* args = value->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  TensorType x1 = Downcast<TensorType>(GetType(args->x1));
  TensorType x2 = Downcast<TensorType>(GetType(args->x2));
  CHECK_EQ(x1->dtype, x2->dtype) << "Data types mismatch (" << x1->dtype << " vs " << x2->dtype
                                 << ")";
  Array<PrimExpr> oshape = BroadcastShape(x1, x2);
  return TensorType(oshape, x1->dtype);
}

Type LogicalBroadcastInfer(const CallValues& value) {
  const auto* args = value->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  TensorType x1 = Downcast<TensorType>(GetType(args->x1));
  TensorType x2 = Downcast<TensorType>(GetType(args->x2));
  CHECK_EQ(x1->dtype, x2->dtype) << "Data types mismatch";
  Array<PrimExpr> oshape = BroadcastShape(x1, x2);
  return TensorType(oshape, DataType::Bool(x1->dtype.lanes()));
}

RAF_OP_TYPE("raf.op.add", "BroadcastUfunc", BroadcastUfuncInfer);
RAF_OP_TYPE("raf.op.subtract", "BroadcastUfunc", BroadcastUfuncInfer);
RAF_OP_TYPE("raf.op.multiply", "Broadcast", BroadcastInfer);
RAF_OP_TYPE("raf.op.divide", "Broadcast", BroadcastInfer);
RAF_OP_TYPE("raf.op.floor_divide", "Broadcast", BroadcastInfer);
RAF_OP_TYPE("raf.op.mod", "Broadcast", BroadcastInfer);
RAF_OP_TYPE("raf.op.maximum", "Broadcast", BroadcastInfer);
RAF_OP_TYPE("raf.op.minimum", "Broadcast", BroadcastInfer);
RAF_OP_TYPE("raf.op.power", "Power", BroadcastInfer);
RAF_OP_TYPE("raf.op.right_shift", "Broadcast", BroadcastInfer);
RAF_OP_TYPE("raf.op.less", "LogicalBroadcast", LogicalBroadcastInfer);
RAF_OP_TYPE("raf.op.greater", "LogicalBroadcast", LogicalBroadcastInfer);
RAF_OP_TYPE("raf.op.less_equal", "LogicalBroadcast", LogicalBroadcastInfer);
RAF_OP_TYPE("raf.op.greater_equal", "LogicalBroadcast", LogicalBroadcastInfer);
RAF_OP_TYPE("raf.op.equal", "LogicalBroadcast", LogicalBroadcastInfer);
RAF_OP_TYPE("raf.op.not_equal", "LogicalBroadcast", LogicalBroadcastInfer);
RAF_OP_TYPE("raf.op.logical_and", "LogicalBroadcast", LogicalBroadcastInfer);
RAF_OP_TYPE("raf.op.left_shift", "Broadcast", BroadcastInfer);

Type AxisTypeInfer(const CallValues& value) {
  const auto* args = value->args.as<BinaryArgs>();
  TensorType x1 = Downcast<TensorType>(GetType(args->x1));
  TensorType x2 = Downcast<TensorType>(GetType(args->x2));

  if (x1.as<TensorTypeNode>() && x2.as<TensorTypeNode>()) {
    CHECK_LE(x2->shape.size(), x1->shape.size());
    Array<tvm::PrimExpr> shape;
    shape.push_back(Integer(x1->shape.size()));
    return TensorType(shape, tvm::runtime::DataType::UInt(32));
  } else {
    return IncompleteType(tvm::kType);
  }
}

RAF_OP_TYPE("raf.op.get_reduce_axis", "ReduceAxis", AxisTypeInfer);
RAF_OP_TYPE("raf.op.get_kept_dims", "KeptDims", AxisTypeInfer);

}  // namespace op
}  // namespace raf
