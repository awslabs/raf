/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/gemm.cc
 * \brief Declaration of genmm-related operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/gemm.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

MNM_OP_DECLARE("mnm.op.matmul", [](const CallValues& call) {
  // TODO(@junrushao1994): sanity check
  const auto* args = call->args.as<MatmulArgs>();
  CHECK(args != nullptr);
  const DLTensor* a = args->a;
  const DLTensor* b = args->b;
  // a is of shape [n1, m1]
  // b is of shape [n2, m2]
  CHECK_EQ(a->ndim, 2);
  CHECK_EQ(b->ndim, 2);
  int64_t n1 = a->shape[0];
  int64_t m1 = a->shape[1];
  int64_t n2 = b->shape[0];
  int64_t m2 = b->shape[1];
  if (args->transpose_a) {
    std::swap(n1, m1);
  }
  if (args->transpose_b) {
    std::swap(n2, m2);
  }
  CHECK_EQ(m1, n2);
  CHECK(a->dtype.code == kDLFloat && (a->dtype.bits == 32 || a->dtype.bits == 64))
      << "Only float and double are supported!";
  call->out = TensorValue::Assemble(/*ctx=*/a->ctx, /*dtype=*/a->dtype, /*shape=*/{n1, m2});
  call->ctx = a->ctx;
  if (!n1 || !n2 || !m1 || !m2) {
    call->callee = ir::NullValue<OpValue>();
  }
});

MNM_OP_DECLARE("mnm.op.matmul_da", [](const CallValues& call) {
  // TODO(@junrushao1994): sanity check
  const auto* args = call->args.as<MatmulDabArgs>();
  CHECK(args != nullptr);
  const DLTensor* dy = args->dy;
  const DLTensor* b = args->a_or_b;
  // a is of shape [n1, m1]
  // b is of shape [n2, m2]
  CHECK_EQ(dy->ndim, 2);
  CHECK_EQ(b->ndim, 2);
  int64_t n1 = dy->shape[0];
  int64_t m1 = dy->shape[1];
  int64_t n2 = b->shape[0];
  int64_t m2 = b->shape[1];
  if (!args->transpose_dx) {
    std::swap(n2, m2);
  }
  CHECK_EQ(m1, n2);
  CHECK(dy->dtype.code == kDLFloat && (dy->dtype.bits == 32 || dy->dtype.bits == 64))
      << "Only float and double are supported!";
  if (args->transpose_dy) {
    std::swap(n1, m2);
  }
  call->out = TensorValue::Assemble(/*ctx=*/dy->ctx, /*dtype=*/dy->dtype, /*shape=*/{n1, m2});
  call->ctx = dy->ctx;
  if (!n1 || !n2 || !m1 || !m2) {
    call->callee = ir::NullValue<OpValue>();
  }
});

MNM_OP_DECLARE("mnm.op.matmul_db", [](const CallValues& call) {
  // TODO(@junrushao1994): sanity check
  const auto* args = call->args.as<MatmulDabArgs>();
  CHECK(args != nullptr);
  const DLTensor* a = args->a_or_b;
  const DLTensor* dy = args->dy;
  // a is of shape [n1, m1]
  // b is of shape [n2, m2]
  CHECK_EQ(a->ndim, 2);
  CHECK_EQ(dy->ndim, 2);
  int64_t n1 = a->shape[0];
  int64_t m1 = a->shape[1];
  int64_t n2 = dy->shape[0];
  int64_t m2 = dy->shape[1];
  if (!args->transpose_dx) {
    std::swap(n1, m1);
  }
  CHECK_EQ(m1, n2);
  CHECK(dy->dtype.code == kDLFloat && (dy->dtype.bits == 32 || dy->dtype.bits == 64))
      << "Only float and double are supported!";
  if (args->transpose_dy) {
    std::swap(n1, m2);
  }
  call->out = TensorValue::Assemble(/*ctx=*/dy->ctx, /*dtype=*/dy->dtype, /*shape=*/{n1, m2});
  call->ctx = dy->ctx;
  if (!n1 || !n2 || !m1 || !m2) {
    call->callee = ir::NullValue<OpValue>();
  }
});

}  // namespace declare
}  // namespace op
}  // namespace mnm
