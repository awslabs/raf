/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/binary.cc
 * \brief Typing of gemm operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/ufunc.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using tvm::relay::Type;
using namespace mnm::value;
using schema::BinaryArgs;

template<bool transpose_a, bool transpose_b>
Type MatmulInfer(const CallValues& value) {
  using namespace tvm;
  using namespace tvm::relay;
  const auto* args = value->args.as<BinaryArgs>();
  TensorType x = Downcast<TensorType>(GetType(args->x1));
  TensorType y = Downcast<TensorType>(GetType(args->x2));
  CHECK(x->shape.size() == 2 && y->shape.size() == 2);
  PrimExpr n1 = x->shape[0];
  PrimExpr m1 = x->shape[1];
  PrimExpr n2 = y->shape[0];
  PrimExpr m2 = y->shape[1];
  if (transpose_a) {
    std::swap(n1, m1);
  }
  if (transpose_b) {
    std::swap(n2, m2);
  }
  CHECK(TypeCheckEqual(m1, n2))
      << "Matmul: shapes of x and y is inconsistent, "
      << " x shape=" << x->shape
      << ", y shape=" << y->shape;

  Array<tvm::PrimExpr> oshape = {n1, m2};
  return TensorType(oshape, x->dtype);
}

MNM_OP_TYPE("mnm.op.matmul", "Matmul", (MatmulInfer<false, false>));
MNM_OP_TYPE("mnm.op.matmul_nt", "MatmulNT", (MatmulInfer<false, true>));
MNM_OP_TYPE("mnm.op.matmul_tn", "MatmulTN", (MatmulInfer<true, false>));
MNM_OP_TYPE("mnm.op.matmul_tt", "MatmulTT", (MatmulInfer<true, true>));

}  // namespace type
}  // namespace op
}  // namespace mnm
