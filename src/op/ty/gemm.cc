/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/gemm.cc
 * \brief Typing of gemm operators
 */
#include <tvm/relay/type.h>
#include "raf/type.h"
#include "../schema/ufunc.h"
#include "./utils.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;
using schema::BinaryArgs;

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

template <bool transpose_a, bool transpose_b>
Type BatchMatmulInfer(const CallValues& value) {
  const auto* args = value->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x1));
  TensorType y = Downcast<TensorType>(GetType(args->x2));
  CHECK(x->shape.size() == 3 && y->shape.size() == 3);
  PrimExpr k1 = x->shape[0];
  PrimExpr n1 = x->shape[1];
  PrimExpr m1 = x->shape[2];
  PrimExpr k2 = y->shape[0];
  PrimExpr n2 = y->shape[1];
  PrimExpr m2 = y->shape[2];
  if (transpose_a) {
    std::swap(n1, m1);
  }
  if (transpose_b) {
    std::swap(n2, m2);
  }
  int64_t k1_v = k1.as<IntImmNode>()->value;
  int64_t k2_v = k2.as<IntImmNode>()->value;
  CHECK(k1_v == k2_v || k1_v == 1 || k2_v == 1)
      << "Incompatible broadcast type " << x << " and " << y;
  PrimExpr k = (k1_v > k2_v) ? k1 : k2;
  Array<tvm::PrimExpr> oshape = {k, n1, m2};
  return TensorType(oshape, x->dtype);
}

RAF_OP_TYPE("raf.op.matmul", "Matmul", (MatmulInfer<false, false>));
RAF_OP_TYPE("raf.op.matmul_nt", "MatmulNT", (MatmulInfer<false, true>));
RAF_OP_TYPE("raf.op.matmul_tn", "MatmulTN", (MatmulInfer<true, false>));
RAF_OP_TYPE("raf.op.matmul_tt", "MatmulTT", (MatmulInfer<true, true>));
RAF_OP_TYPE("raf.op.dense", "DenseInfer", (MatmulInfer<false, true>));
RAF_OP_TYPE("raf.op.batch_matmul", "BatchMatmulNN", (BatchMatmulInfer<false, false>));
RAF_OP_TYPE("raf.op.batch_matmul_nt", "BatchMatmulNT", (BatchMatmulInfer<false, true>));
RAF_OP_TYPE("raf.op.batch_matmul_tn", "BatchMatmulTN", (BatchMatmulInfer<true, false>));
RAF_OP_TYPE("raf.op.batch_matmul_tt", "BatchMatmulTT", (BatchMatmulInfer<true, true>));

}  // namespace op
}  // namespace raf
