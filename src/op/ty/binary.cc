/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/binary.cc
 * \brief Typing of binary operators
 */
#include <tvm/relay/type.h>
#include <tvm/tir/op.h>
#include "mnm/type.h"
#include "../schema/ufunc.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using namespace mnm::value;
using namespace mnm::op::schema;
using namespace tvm;
using namespace tvm::relay;

Array<PrimExpr> BroadcastShape(const TensorType& x1, const TensorType& x2) {
  size_t ndim_1 = x1->shape.size();
  size_t ndim_2 = x2->shape.size();
  size_t ndim = std::max(ndim_1, ndim_2);
  Array<PrimExpr> oshape(ndim, 0);
  for (size_t i = 0; i < ndim; ++i) {
    PrimExpr lhs = (i < ndim_1) ? x1->shape[ndim_1 - 1 - i] : Integer(1);
    PrimExpr rhs = (i < ndim_2) ? x2->shape[ndim_2 - 1 - i] : Integer(1);

    if (tir::is_const_int(lhs, 1)) {
      oshape.Set(ndim - 1 - i, rhs);
    } else if (tir::is_const_int(rhs, 1)) {
      oshape.Set(ndim - 1 - i, lhs);
    } else if (lhs.as<AnyNode>()) {
      oshape.Set(ndim - 1 - i, rhs);
    } else if (rhs.as<AnyNode>()) {
      oshape.Set(ndim - 1 - i, lhs);
    } else if (TypeCheckCompare(lhs, rhs, std::equal_to<int>())) {
      oshape.Set(ndim - 1 - i, lhs);
    } else {
      LOG(FATAL) << "Incompatible broadcast type " << x1 << " and " << x2;
    }
  }
  return oshape;
}

Type BroadcastInfer(const CallValues& value) {
  using namespace tvm;
  using namespace tvm::relay;
  const auto* args = value->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  TensorType x1 = Downcast<TensorType>(GetType(args->x1));
  TensorType x2 = Downcast<TensorType>(GetType(args->x2));
  CHECK_EQ(x1->dtype, x2->dtype) << "Data types mismatch (" << x1->dtype << " vs " << x2->dtype
                                 << ")";
  Array<PrimExpr> oshape = BroadcastShape(x1, x2);
  return TensorType(oshape, x1->dtype);
}

Type LogicalBroadcastInfer(const CallValues& value) {
  using namespace tvm;
  using namespace tvm::relay;
  const auto* args = value->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  TensorType x1 = Downcast<TensorType>(GetType(args->x1));
  TensorType x2 = Downcast<TensorType>(GetType(args->x2));
  CHECK_EQ(x1->dtype, x2->dtype) << "Data types mismatch";
  Array<PrimExpr> oshape = BroadcastShape(x1, x2);
  return TensorType(oshape, DataType::Bool(x1->dtype.lanes()));
}

MNM_OP_TYPE("mnm.op.add", "Broadcast", BroadcastInfer);
MNM_OP_TYPE("mnm.op.subtract", "Broadcast", BroadcastInfer);
MNM_OP_TYPE("mnm.op.multiply", "Broadcast", BroadcastInfer);
MNM_OP_TYPE("mnm.op.divide", "Broadcast", BroadcastInfer);
MNM_OP_TYPE("mnm.op.mod", "Broadcast", BroadcastInfer);
MNM_OP_TYPE("mnm.op.maximum", "Broadcast", BroadcastInfer);
MNM_OP_TYPE("mnm.op.minimum", "Broadcast", BroadcastInfer);
MNM_OP_TYPE("mnm.op.power", "Power", BroadcastInfer);
MNM_OP_TYPE("mnm.op.right_shift", "Broadcast", BroadcastInfer);
MNM_OP_TYPE("mnm.op.less", "LogicalBroadcast", LogicalBroadcastInfer);
MNM_OP_TYPE("mnm.op.greater", "LogicalBroadcast", LogicalBroadcastInfer);
MNM_OP_TYPE("mnm.op.less_equal", "LogicalBroadcast", LogicalBroadcastInfer);
MNM_OP_TYPE("mnm.op.greater_equal", "LogicalBroadcast", LogicalBroadcastInfer);
MNM_OP_TYPE("mnm.op.equal", "LogicalBroadcast", LogicalBroadcastInfer);
MNM_OP_TYPE("mnm.op.not_equal", "LogicalBroadcast", LogicalBroadcastInfer);
MNM_OP_TYPE("mnm.op.logical_and", "LogicalBroadcast", LogicalBroadcastInfer);
MNM_OP_TYPE("mnm.op.left_shift", "Broadcast", BroadcastInfer);

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

MNM_OP_TYPE("mnm.op.get_reduce_axis", "ReduceAxis", AxisTypeInfer);
MNM_OP_TYPE("mnm.op.get_kept_dims", "KeptDims", AxisTypeInfer);
}  // namespace type
}  // namespace op
}  // namespace mnm
