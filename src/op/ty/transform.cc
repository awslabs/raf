/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/transform.cc
 * \brief Typing of transform operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/transform.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using namespace mnm::value;
using schema::TransposeArgs;
using schema::TransposeDxArgs;
using tvm::relay::Type;

Type TransposeInfer(const CallValues& value) {
  using namespace tvm;
  using namespace tvm::relay;
  const auto* args = value->args.as<TransposeArgs>();
  const std::vector<int64_t>& axes = args->axes;
  TensorType x = Downcast<TensorType>(GetType(args->x));
  size_t ndim = x->shape.size();
  Array<tvm::PrimExpr> oshape;
  if (axes.size() != 0) {
    CHECK_EQ(axes.size(), ndim);
    for (size_t i = 0; i < ndim; ++i) {
      int64_t axis = axes[i] < 0 ? axes[i] + ndim : axes[i];
      oshape.push_back(x->shape[axis]);
    }
  } else {
    for (size_t i = 0; i < ndim; ++i) {
      oshape.push_back(x->shape[ndim - i - 1]);
    }
  }
  return TensorType(oshape, x->dtype);
}

MNM_OP_TYPE("mnm.op.transpose", "Transpose", TransposeInfer);

Type TransposeDxInfer(const CallValues& value) {
  using namespace tvm;
  using namespace tvm::relay;
  const auto* args = value->args.as<TransposeDxArgs>();
  CHECK(args != nullptr);
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.transpose_dx", "TransposeDx", TransposeDxInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
