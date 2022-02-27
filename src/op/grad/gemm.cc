/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/grad/binary.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"

namespace raf {
namespace op {
namespace grad {

using namespace raf::ir;
using namespace raf::value;

template <bool transpose_a, bool transpose_b>
Array<Expr> MatmulGradImpl(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                           const Expr& dy) {
  static auto op_nn = Op::Get("raf.op.matmul");
  static auto op_nt = Op::Get("raf.op.matmul_nt");
  static auto op_tn = Op::Get("raf.op.matmul_tn");
  static auto op_tt = Op::Get("raf.op.matmul_tt");
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& a = call->args[0];
  const Expr& b = call->args[1];
  if (!transpose_a) {
    if (!transpose_b) {
      return {
          Call(op_nt, {dy, b}),
          Call(op_tn, {a, dy}),
      };
    } else {
      return {
          Call(op_nn, {dy, b}),
          Call(op_tn, {dy, a}),
      };
    }
  } else {
    if (!transpose_b) {
      return {
          Call(op_nt, {b, dy}),
          Call(op_nn, {a, dy}),
      };
    } else {
      return {
          Call(op_tt, {b, dy}),
          Call(op_tt, {dy, a}),
      };
    }
  }
  LOG(FATAL) << "Unreachable code";
  throw;
}

auto MatmulGradNN = MatmulGradImpl<false, false>;
auto MatmulGradNT = MatmulGradImpl<false, true>;
auto MatmulGradTN = MatmulGradImpl<true, false>;
auto MatmulGradTT = MatmulGradImpl<true, true>;

RAF_OP_GRAD("raf.op.matmul", MatmulGradNN);
RAF_OP_GRAD("raf.op.matmul_nt", MatmulGradNT);
RAF_OP_GRAD("raf.op.dense", MatmulGradNT);
RAF_OP_GRAD("raf.op.matmul_tn", MatmulGradTN);
RAF_OP_GRAD("raf.op.matmul_tt", MatmulGradTT);

template <bool transpose_a, bool transpose_b>
Array<Expr> BatchMatmulGradImpl(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                                const Expr& dy) {
  static auto op_nn = Op::Get("raf.op.batch_matmul");
  static auto op_nt = Op::Get("raf.op.batch_matmul_nt");
  static auto op_tn = Op::Get("raf.op.batch_matmul_tn");
  static auto op_tt = Op::Get("raf.op.batch_matmul_tt");
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& a = call->args[0];
  const Expr& b = call->args[1];

  if (!transpose_a) {
    if (!transpose_b) {
      return {
          GetCollapseSumLike(Call(op_nt, {dy, b}), a),
          GetCollapseSumLike(Call(op_tn, {a, dy}), b),
      };
    } else {
      return {
          GetCollapseSumLike(Call(op_nn, {dy, b}), a),
          GetCollapseSumLike(Call(op_tn, {dy, a}), b),
      };
    }
  } else {
    if (!transpose_b) {
      return {
          GetCollapseSumLike(Call(op_nt, {b, dy}), a),
          GetCollapseSumLike(Call(op_nn, {a, dy}), b),
      };
    } else {
      return {
          GetCollapseSumLike(Call(op_tt, {b, dy}), a),
          GetCollapseSumLike(Call(op_tt, {dy, a}), b),
      };
    }
  }
  LOG(FATAL) << "Unreachable code";
  throw;
}

auto BatchMatmulGradNN = BatchMatmulGradImpl<false, false>;
auto BatchMatmulGradNT = BatchMatmulGradImpl<false, true>;
auto BatchMatmulGradTN = BatchMatmulGradImpl<true, false>;
auto BatchMatmulGradTT = BatchMatmulGradImpl<true, true>;

RAF_OP_GRAD("raf.op.batch_matmul", BatchMatmulGradNN);
RAF_OP_GRAD("raf.op.batch_matmul_nt", BatchMatmulGradNT);
RAF_OP_GRAD("raf.op.batch_matmul_tn", BatchMatmulGradTN);
RAF_OP_GRAD("raf.op.batch_matmul_tt", BatchMatmulGradTT);

}  // namespace grad
}  // namespace op
}  // namespace raf
