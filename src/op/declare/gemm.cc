/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/gemm.cc
 * \brief Declaration of genmm-related operators
 */
#include "raf/op.h"
#include "raf/tensor.h"
#include "../schema/ufunc.h"

namespace raf {
namespace op {
namespace declare {

using namespace raf::op::schema;
using namespace raf::value;

template <bool transpose_a, bool transpose_b>
void MatmulDecl(const CallValues& call) {
  // TODO(@junrushao1994): sanity check
  const auto* args = call->args.as<schema::BinaryArgs>();
  CHECK(args != nullptr);
  const DLTensor* a = args->x1;
  const DLTensor* b = args->x2;
  // a is of shape [n1, m1]
  // b is of shape [n2, m2]
  CHECK_EQ(a->ndim, 2);
  CHECK_EQ(b->ndim, 2);
  int64_t n1 = a->shape[0];
  int64_t m1 = a->shape[1];
  int64_t n2 = b->shape[0];
  int64_t m2 = b->shape[1];
  if (transpose_a) {
    std::swap(n1, m1);
  }
  if (transpose_b) {
    std::swap(n2, m2);
  }
  CHECK_EQ(m1, n2);
  CHECK(a->dtype.code == kDLFloat) << "Only float types are supported!";
  call->out = TensorValue::Assemble(/*dev=*/a->device, /*dtype=*/a->dtype,
                                    /*shape=*/std::vector<int64_t>{n1, m2});
  call->device = a->device;
  if (!n1 || !n2 || !m1 || !m2) {
    call->callee = ir::NullValue<OpValue>();
  }
}

auto MatmulNN = MatmulDecl<false, false>;
auto MatmulNT = MatmulDecl<false, true>;
auto MatmulTN = MatmulDecl<true, false>;
auto MatmulTT = MatmulDecl<true, true>;

RAF_OP_DECLARE("raf.op.matmul", MatmulNN);
RAF_OP_DECLARE("raf.op.matmul_nt", MatmulNT);
RAF_OP_DECLARE("raf.op.matmul_tn", MatmulTN);
RAF_OP_DECLARE("raf.op.matmul_tt", MatmulTT);

template <bool transpose_a, bool transpose_b>
void BatchMatmulDecl(const CallValues& call) {
  const auto* args = call->args.as<schema::BinaryArgs>();
  CHECK(args != nullptr);
  const DLTensor* a = args->x1;
  const DLTensor* b = args->x2;
  // a is of shape [k1, n1, m1]
  // b is of shape [k2, n2, m2]
  CHECK_EQ(a->ndim, 3);
  CHECK_EQ(b->ndim, 3);
  int64_t batch_size = a->shape[0] > b->shape[0] ? a->shape[0] : b->shape[0];
  int64_t k1 = a->shape[0];
  int64_t n1 = a->shape[1];
  int64_t m1 = a->shape[2];
  int64_t k2 = b->shape[0];
  int64_t n2 = b->shape[1];
  int64_t m2 = b->shape[2];
  if (transpose_a) {
    std::swap(n1, m1);
  }
  if (transpose_b) {
    std::swap(n2, m2);
  }
  CHECK_EQ(m1, n2);
  CHECK(k1 == k2 || k1 == 1 || k2 == 1)
      << "Incompatible broadcast batch size " << k1 << " and " << k2;

  CHECK(a->dtype.code == kDLFloat &&
        (a->dtype.bits == 16 || a->dtype.bits == 32 || a->dtype.bits == 64))
      << "Only float and double are supported!";
  int64_t k = k1 > k2 ? k1 : k2;
  call->out = TensorValue::Assemble(/*dev=*/a->device, /*dtype=*/a->dtype,
                                    /*shape=*/std::vector<int64_t>{k, n1, m2});
  call->device = a->device;
  if (!k1 || !k2 || !n1 || !n2 || !m1 || !m2) {
    call->callee = ir::NullValue<OpValue>();
  }
}

auto BatchMatmulNN = BatchMatmulDecl<false, false>;
auto BatchMatmulNT = BatchMatmulDecl<false, true>;
auto BatchMatmulTN = BatchMatmulDecl<true, false>;
auto BatchMatmulTT = BatchMatmulDecl<true, true>;

RAF_OP_DECLARE("raf.op.batch_matmul", BatchMatmulNN);
RAF_OP_DECLARE("raf.op.batch_matmul_nt", BatchMatmulNT);
RAF_OP_DECLARE("raf.op.batch_matmul_tn", BatchMatmulTN);
RAF_OP_DECLARE("raf.op.batch_matmul_tt", BatchMatmulTT);

RAF_OP_DECLARE("raf.op.dense", [](const CallValues& call) {
  const auto* args = call->args.as<schema::BinaryArgs>();
  CHECK(args != nullptr);
  const DLTensor* a = args->x1;
  const DLTensor* b = args->x2;
  CHECK_EQ(a->ndim, 2);
  CHECK_EQ(b->ndim, 2);
  int64_t n1 = a->shape[0];
  int64_t m1 = a->shape[1];
  int64_t n2 = b->shape[0];
  int64_t m2 = b->shape[1];
  CHECK_EQ(m1, m2);
  call->out = TensorValue::Assemble(/*dev=*/a->device, /*dtype=*/a->dtype,
                                    /*shape=*/std::vector<int64_t>{n1, n2});
  call->device = a->device;
  if (!n1 || !n2 || !m1 || !m2) {
    call->callee = ir::NullValue<OpValue>();
  }
});

}  // namespace declare
}  // namespace op
}  // namespace raf
