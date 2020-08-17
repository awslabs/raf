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
using schema::BinaryArgs;
using schema::BinaryUfuncArgs;
using tvm::relay::Type;

Type BroadcastInfer(const CallValues& value) {
  using namespace tvm;
  using namespace tvm::relay;
  const auto* args = value->args.as<BinaryUfuncArgs>();
  TensorType x1 = Downcast<TensorType>(GetType(args->x1));
  TensorType x2 = Downcast<TensorType>(GetType(args->x2));
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
    } else if (TypeCheckEqual(lhs, rhs)) {
      oshape.Set(ndim - 1 - i, lhs);
    } else {
      LOG(FATAL) << "Incompatible broadcast type " << x1 << " and " << x2;
    }
  }
  return TensorType(oshape, x1->dtype);
}

MNM_OP_TYPE("mnm.op.add", "Broadcast", BroadcastInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
