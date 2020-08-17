/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/transform.cc
 * \brief Typing of transform operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/ufunc.h"
#include "../schema/nn.h"
#include "../schema/transform.h"
#include "../declare/declare_utils.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using namespace mnm::value;
using declare::NormalizeAxis;
using schema::ConcatenateArgs;
using schema::TransposeArgs;
using schema::TransposeDxArgs;
using schema::UnaryArgs;
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

Type BatchFlattenInfer(const CallValues& value) {
  using namespace tvm;
  using namespace tvm::relay;
  const auto* args = value->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  auto ndim = x->shape.size();
  CHECK_GE(ndim, 2) << "ValueError: batch_flatten only works with ndim >= 2";
  PrimExpr flat = x->shape[1];
  for (size_t i = 2; i < ndim; i++) {
    flat *= x->shape[i];
  }
  return TensorType(Array<PrimExpr>{x->shape[0], flat}, x->dtype);
}

MNM_OP_TYPE("mnm.op.batch_flatten", "BatchFlatten", BatchFlattenInfer);

Type ConcatenateInfer(const CallValues& value) {
  using namespace tvm;
  using namespace relay;
  const auto* args = value->args.as<ConcatenateArgs>();
  CHECK(args != nullptr);
  CHECK(args->x.size() > 0);
  Array<Type> x;
  std::transform(args->x.begin(), args->x.end(), std::back_inserter(x), GetType);
  TensorType y0 = Downcast<TensorType>(x[0]);
  int axis = NormalizeAxis(args->axis, y0->shape.size());
  PrimExpr dimsize = 0;
  for (auto& i : x) {
    TensorType y = Downcast<TensorType>(i);
    CHECK(y->shape.size() == y0->shape.size());
    for (int k = 0; k < y0->shape.size(); k++) {
      if (k != axis) {
        CHECK(TypeCheckEqual(y->shape[k], y0->shape[k]));
      }
    }
    dimsize += y->shape[axis];
  }
  Array<PrimExpr> shape(y0->shape.begin(), y0->shape.end());
  shape.Set(axis, dimsize);
  return TensorType(shape, y0->dtype);
}

MNM_OP_TYPE("mnm.op.concatenate", "Concatenate", ConcatenateInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
