/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/gemm.cc
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
using namespace tvm;
using namespace tvm::relay;

template <bool transpose_a, bool transpose_b>
Type MatmulInfer(const CallValues& value) {
  const auto* args = value->args.as<BinaryArgs>();
  CHECK(args != nullptr);
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
  CHECK(TypeCheckCompare(m1, n2, std::equal_to<int>()))
      << "Matmul: shapes of x and y is inconsistent, "
      << " x shape=" << x->shape << ", y shape=" << y->shape;
  Array<tvm::PrimExpr> oshape = {n1, m2};
  return TensorType(oshape, x->dtype);
}

Type BatchMatmulInfer(const CallValues& value) {
  const auto* args = value->args.as<schema::BinaryArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x1));
  TensorType y = Downcast<TensorType>(GetType(args->x2));
  // a is of shape [k1, n1, m1]
  // b is of shape [k2, n2, m2]
  CHECK(x->shape.size() == 3 && y->shape.size() == 3);
  PrimExpr k1 = x->shape[0];
  PrimExpr n1 = x->shape[1];
  PrimExpr m1 = x->shape[2];
  PrimExpr k2 = y->shape[0];
  PrimExpr n2 = y->shape[1];
  PrimExpr m2 = y->shape[2];
  CHECK(TypeCheckCompare(m1, m2, std::equal_to<int>()))
      << "BatchMatmul: shapes of x and y is inconsistent, "
      << " x shape=" << x->shape << ", y shape=" << y->shape;
  CHECK(TypeCheckCompare(k1, k2, std::equal_to<int>()))
      << "BatchMatmul: batch size of x and y is inconsistent, "
      << " x shape=" << x->shape << ", y shape=" << y->shape;
  Array<tvm::PrimExpr> oshape = {k1, n1, n2};
  return TensorType(oshape, x->dtype);
}

MNM_OP_TYPE("mnm.op.matmul", "Matmul", (MatmulInfer<false, false>));
MNM_OP_TYPE("mnm.op.matmul_nt", "MatmulNT", (MatmulInfer<false, true>));
MNM_OP_TYPE("mnm.op.matmul_tn", "MatmulTN", (MatmulInfer<true, false>));
MNM_OP_TYPE("mnm.op.matmul_tt", "MatmulTT", (MatmulInfer<true, true>));
MNM_OP_TYPE("mnm.op.dense", "DenseInfer", (MatmulInfer<false, true>));
MNM_OP_TYPE("mnm.op.batch_matmul", "BatchMatmulInfer", BatchMatmulInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
