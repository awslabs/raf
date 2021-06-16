/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/grad/binary.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"
#include "mnm/pass.h"
#include "mnm/ir.h"

namespace mnm {
namespace op {
namespace grad {

using namespace mnm::ir;

Array<Expr> AdvIndexGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                         const Expr& dy) {
  static auto adv_index_dx = Op::Get("mnm.op.adv_index_dx");
  static auto zeros_like = Op::Get("mnm.op.zeros_like");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  auto inputs = call->args[0];
  const Expr& ret = Call(adv_index_dx, {dy, inputs});
  Array<Expr> tuple;
  tuple.push_back(TupleGetItem(ret, 0));

  int num_inputs = 1;
  if (auto tuple_node = orig_args[0].as<TupleNode>()) {
    num_inputs = tuple_node->fields.size();
    for (int i = 1; i < num_inputs; i++) {
      auto zero_grad = Call(zeros_like, {tuple_node->fields[i]});
      tuple.push_back(zero_grad);
    }
  }
  return {tvm::relay::Tuple(tuple)};
}

MNM_OP_GRAD("mnm.op.adv_index", AdvIndexGrad);

Array<Expr> BatchFlattenGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                             const Expr& dy) {
  static auto reshape = Op::Get("mnm.op.reshape");
  static auto shape = Op::Get("mnm.op.shape");
  const CallNode* call = orig_call.as<CallNode>();
  return {Call(reshape, {dy, Call(shape, {call->args[0]})})};
}

MNM_OP_GRAD("mnm.op.batch_flatten", BatchFlattenGrad);

Array<Expr> TransposeGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                          const Expr& dy) {
  static auto transpose_dx = Op::Get("mnm.op.transpose_dx");
  static auto shape = Op::Get("mnm.op.shape");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& axes = call->args[1];
  const Expr& primal_shape = Call(shape, {call->args[0]});
  return {Call(transpose_dx, {dy, axes, primal_shape})};
}

MNM_OP_GRAD("mnm.op.transpose", TransposeGrad);

Array<Expr> RepeatGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                       const Expr& dy) {
  static auto repeat_dx = Op::Get("mnm.op.repeat_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& x = call->args[0];
  const Expr& repeats = call->args[1];
  const Expr& axes = call->args[2];
  return {Call(repeat_dx, {x, dy, repeats, axes})};
}

MNM_OP_GRAD("mnm.op.repeat", RepeatGrad);

Array<Expr> SwapAxisGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                         const Expr& dy) {
  static auto swap_axis = Op::Get("mnm.op.swap_axis");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& axes1 = call->args[1];
  const Expr& axes2 = call->args[2];
  return {Call(swap_axis, {dy, axes1, axes2})};
}

MNM_OP_GRAD("mnm.op.swap_axis", SwapAxisGrad);

Array<Expr> BroadcastToGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                            const Expr& dy) {
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& x = call->args[0];
  auto f = [&dy](const Expr& x) {
    static auto collapse_axis = Op::Get("mnm.op.get_reduce_axis");
    static auto collapse_keep = Op::Get("mnm.op.get_kept_dims");
    static auto sum = Op::Get("mnm.op.sum");
    Call axes = Call(collapse_axis, {dy, x});
    Call keep = Call(collapse_keep, {dy, x});
    return Call(sum, {dy, axes, keep});
  };

  return {f(x)};
}

MNM_OP_GRAD("mnm.op.broadcast_to", BroadcastToGrad);

Array<Expr> StackGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                      const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.split");
  static auto op_squeeze = Op::Get("mnm.op.squeeze");
  const CallNode* call = orig_call.as<CallNode>();

  int num_inputs = 1;
  if (auto tuple_node = orig_args[0].as<TupleNode>()) {
    num_inputs = tuple_node->fields.size();
  }

  CHECK_GE(call->args.size(), 2);
  const Expr& x = call->args[0];
  const Expr& axis = call->args[1];
  Expr sections = MakeConstant(mnm::value::ScalarValue::make(num_inputs));
  Expr split = Call(op_dx, {dy, sections, axis});

  Array<Expr> tuple;
  for (int i = 0; i < num_inputs; i++) {
    auto split_i = tvm::relay::TupleGetItem(split, i);
    auto tuple_i = Call(op_squeeze, {split_i, axis});
    tuple.push_back(tuple_i);
  }
  return {tvm::relay::Tuple(tuple)};
}

MNM_OP_GRAD("mnm.op.stack", StackGrad);

Array<Expr> ConcatenateGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                            const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.split");
  static auto op_indices = Op::Get("mnm.op.concatenate_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x = call->args[0];
  const Expr& axis = call->args[1];
  Expr indices = Call(op_indices, {x, axis});
  return {Call(op_dx, {dy, indices, axis})};
}

MNM_OP_GRAD("mnm.op.concatenate", ConcatenateGrad);

Array<Expr> SplitGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                      const Expr& dy) {
  static auto concatenate = Op::Get("mnm.op.concatenate");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  CHECK_GE(call->args.size(), 3);
  const Expr& axis = call->args[2];
  return {Call(concatenate, {dy, axis})};
}

MNM_OP_GRAD("mnm.op.split", SplitGrad);

Array<Expr> MeshGridGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                         const Expr& dy) {
  using namespace mnm::value;
  const CallNode* call = orig_call.as<CallNode>();
  static auto sum = Op::Get("mnm.op.sum");
  CHECK(call != nullptr);
  int num_inputs = 1;
  if (auto tuple_node = orig_args[0].as<TupleNode>()) {
    num_inputs = tuple_node->fields.size();
  }
  const Expr& x = call->args[0];
  Expr exclude = MakeConstant(ScalarValue::make(true));
  Expr keep_dims = MakeConstant(ScalarValue::make((int64_t)0));
  Array<Expr> tuple;
  for (int i = 0; i < num_inputs; i++) {
    auto split_dy = tvm::relay::TupleGetItem(dy, i);
    Expr axis = MakeConstant(ScalarValue::make(i));
    Expr ret_i = Call(sum, {split_dy, axis, keep_dims, exclude});
    tuple.push_back(ret_i);
  }
  return {tvm::relay::Tuple(tuple)};
}

MNM_OP_GRAD("mnm.op.mesh_grid", MeshGridGrad);

Array<Expr> ReverseGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                        const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.reverse");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& axis = call->args[1];
  return {Call(op_dx, {dy, axis})};
}

MNM_OP_GRAD("mnm.op.reverse", ReverseGrad);

Array<Expr> ReverseSequenceGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                                const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.reverse_sequence");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 4);
  const Expr& seq_length = call->args[1];
  const Expr& seq_axis = call->args[2];
  const Expr& batch_axis = call->args[3];
  return {Call(op_dx, {dy, seq_length, seq_axis, batch_axis})};
}

MNM_OP_GRAD("mnm.op.reverse_sequence", ReverseSequenceGrad);

Array<Expr> ClipGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                     const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.clip_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 3);
  const Expr& x = call->args[0];
  const Expr& a_min = call->args[1];
  const Expr& a_max = call->args[2];
  return {Call(op_dx, {x, dy, a_min, a_max})};
}

MNM_OP_GRAD("mnm.op.clip", ClipGrad);

Array<Expr> ExpandDimsGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                           const Expr& dy) {
  static auto reshape = Op::Get("mnm.op.reshape");
  static auto shape = Op::Get("mnm.op.shape");
  const CallNode* call = orig_call.as<CallNode>();
  return {Call(reshape, {dy, Call(shape, {call->args[0]})})};
}

MNM_OP_GRAD("mnm.op.expand_dims", ExpandDimsGrad);

Array<Expr> ReshapeGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                        const Expr& dy) {
  static auto reshape = Op::Get("mnm.op.reshape");
  static auto shape = Op::Get("mnm.op.shape");
  const CallNode* call = orig_call.as<CallNode>();
  return {Call(reshape, {dy, Call(shape, {call->args[0]})})};
}

MNM_OP_GRAD("mnm.op.reshape", ReshapeGrad);

Array<Expr> SqueezeGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                        const Expr& dy) {
  static auto reshape = Op::Get("mnm.op.reshape");
  static auto shape = Op::Get("mnm.op.shape");
  const CallNode* call = orig_call.as<CallNode>();
  return {Call(reshape, {dy, Call(shape, {call->args[0]})})};
}

MNM_OP_GRAD("mnm.op.squeeze", SqueezeGrad);

Array<Expr> TakeGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                     const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.take_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_EQ(call->args.size(), 4);
  const Expr& x = call->args[0];
  const Expr& indices = call->args[1];
  const Expr& axis = call->args[2];
  const Expr& mode = call->args[3];
  return {Call(op_dx, {x, y, dy, indices, axis, mode})};
}

MNM_OP_GRAD("mnm.op.take", TakeGrad);

Array<Expr> ScatterGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                        const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.scatter_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_EQ(call->args.size(), 4);
  const Expr& x = call->args[0];
  const Expr& index = call->args[1];
  const Expr& src = call->args[2];
  const Expr& axis = call->args[3];
  return {Call(op_dx, {x, y, dy, index, src, axis})};
}

MNM_OP_GRAD("mnm.op.scatter", ScatterGrad);

Array<Expr> CastGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                     const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.cast_like");
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& x = call->args[0];
  return {Call(op_dx, {dy, x})};
}

MNM_OP_GRAD("mnm.op.cast", CastGrad);

Array<Expr> GatherGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                       const Expr& dy) {
  static auto gather_dx = Op::Get("mnm.op.gather_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& data = call->args[0];
  const Expr& axis = call->args[1];
  const Expr& indices = call->args[2];
  return {Call(gather_dx, {data, axis, indices, dy})};
}

MNM_OP_GRAD("mnm.op.gather", GatherGrad);

Array<Expr> GatherNdGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                         const Expr& dy) {
  static auto gather_nd_dx = Op::Get("mnm.op.gather_nd_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& data = call->args[0];
  const Expr& indices = call->args[1];
  return {Call(gather_nd_dx, {data, indices, dy})};
}

MNM_OP_GRAD("mnm.op.gather_nd", GatherNdGrad);

MNM_OP_GRAD("mnm.op.full", NoGrads<0>);

MNM_OP_GRAD("mnm.op.full_like", NoGrads<1>);

Array<Expr> StridedSliceGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                             const Expr& dy) {
  static auto op_slice_dx = Op::Get("mnm.op.strided_slice_dx");
  static auto shape = Op::Get("mnm.op.shape");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& begin = call->args[1];
  const Expr& end = call->args[2];
  const Expr& strides = call->args[3];
  const Expr& mode = call->args[4];
  const Expr& primal_shape = Call(shape, {call->args[0]});
  return {Call(op_slice_dx, {dy, primal_shape, begin, end, strides, mode})};
}

MNM_OP_GRAD("mnm.op.strided_slice", StridedSliceGrad);

Array<Expr> WhereGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                      const Expr& dy) {
  static auto where = Op::Get("mnm.op.where");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& cond = call->args[0];
  const Expr& x1 = call->args[1];
  const Expr& x2 = call->args[2];
  static auto zeros_like = Op::Get("mnm.op.zeros_like");
  auto zero = Call(zeros_like, {dy});

  const Expr& dx1 = Call(where, {cond, dy, zero});
  const Expr& dx2 = Call(where, {cond, zero, dy});
  auto f = [](const Expr& dx, const Expr& x) {
    static auto collapse_axis = Op::Get("mnm.op.get_reduce_axis");
    static auto collapse_keep = Op::Get("mnm.op.get_kept_dims");
    static auto sum = Op::Get("mnm.op.sum");
    Call axes = Call(collapse_axis, {dx, x});
    Call keep = Call(collapse_keep, {dx, x});
    return Call(sum, {dx, axes, keep});
  };
  return {NullValue<Expr>(), f(dx1, x1), f(dx2, x2)};
}

MNM_OP_GRAD("mnm.op.where", WhereGrad);

MNM_OP_GRAD("mnm.op.argwhere", NoGrads<1>);

MNM_OP_GRAD("mnm.op.ndarray_size", NoGrads<1>);

MNM_OP_GRAD("mnm.op.resize", NoGrads<1>);

}  // namespace grad
}  // namespace op
}  // namespace mnm
