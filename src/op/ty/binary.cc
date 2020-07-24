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
using tvm::relay::Type;
using schema::BinaryUfuncArgs;
using schema::BinaryArgs;

Type BroadcastInfer(const CallValues& value) {
  using namespace tvm;
  using namespace tvm::relay;
  const auto* args = value->args.as<BinaryUfuncArgs>();
  TensorType x1 = Downcast<TensorType>(GetType(args->x1));
  TensorType x2 = Downcast<TensorType>(GetType(args->x2));
  CHECK_EQ(x1->shape.size(), x2->shape.size());
  Array<PrimExpr> oshape;
  for (size_t i = 0; i < x1->shape.size(); ++i) {
    PrimExpr lhs = x1->shape[i];
    PrimExpr rhs = x2->shape[i];
    if (tir::is_const_int(lhs, 1)) {
      oshape.push_back(rhs);
    } else if (tir::is_const_int(rhs, 1)) {
      oshape.push_back(lhs);
    } else if (lhs.as<AnyNode>()) {
      oshape.push_back(rhs);
    } else if (rhs.as<AnyNode>()) {
      oshape.push_back(lhs);
    } else if (TypeCheckEqual(lhs, rhs)) {
      oshape.push_back(lhs);
    } else {
      LOG(FATAL) << "Incompatible broadcast type "
                 << x1 << " and " << x2;
    }
  }
  return TensorType(oshape, x1->dtype);
}

MNM_OP_TYPE("mnm.op.add", "Broadcast", BroadcastInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
