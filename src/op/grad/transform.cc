/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/grad/binary.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"
#include "raf/pass.h"
#include "raf/ir_ext.h"
#include "raf/type.h"
#include "raf/value.h"

namespace raf {
namespace op {
namespace grad {

using namespace raf::ir;

Array<Expr> AdvIndexGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                         const Expr& dy) {
  static auto adv_index_dx = Op::Get("raf.op.adv_index_dx");
  static auto zeros_like = Op::Get("raf.op.zeros_like");
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

RAF_OP_GRAD("raf.op.adv_index", AdvIndexGrad);

Array<Expr> ReshapeOpGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                          const Expr& dy) {
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& x = call->args[0];
  return {GetReshapeLike(dy, x)};
}

RAF_OP_GRAD("raf.op.batch_flatten", ReshapeOpGrad);

Array<Expr> TransposeGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                          const Expr& dy) {
  static auto transpose_dx = Op::Get("raf.op.transpose_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& axes = call->args[1];
  return {Call(transpose_dx, {dy, axes})};
}

RAF_OP_GRAD("raf.op.transpose", TransposeGrad);

Array<Expr> RepeatGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                       const Expr& dy) {
  static auto repeat_dx = Op::Get("raf.op.repeat_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& x = call->args[0];
  const Expr& repeats = call->args[1];
  const Expr& axes = call->args[2];
  return {Call(repeat_dx, {x, dy, repeats, axes})};
}

RAF_OP_GRAD("raf.op.repeat", RepeatGrad);

Array<Expr> SwapAxisGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                         const Expr& dy) {
  static auto swap_axis = Op::Get("raf.op.swap_axis");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& axes1 = call->args[1];
  const Expr& axes2 = call->args[2];
  return {Call(swap_axis, {dy, axes1, axes2})};
}

RAF_OP_GRAD("raf.op.swap_axis", SwapAxisGrad);

Array<Expr> BroadcastToGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                            const Expr& dy) {
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& x = call->args[0];
  return {GetCollapseSumLike(dy, x)};
}

RAF_OP_GRAD("raf.op.broadcast_to", BroadcastToGrad);

RAF_OP_GRAD("raf.op.broadcast_to_like", BroadcastToGrad);

Array<Expr> StackGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                      const Expr& dy) {
  static auto op_dx = Op::Get("raf.op.split");
  static auto op_squeeze = Op::Get("raf.op.squeeze");
  const CallNode* call = orig_call.as<CallNode>();

  int num_inputs = 1;
  if (auto tuple_node = orig_args[0].as<TupleNode>()) {
    num_inputs = tuple_node->fields.size();
  }

  CHECK_GE(call->args.size(), 2);
  const Expr& x = call->args[0];
  const Expr& axis = call->args[1];
  Expr sections = MakeConstant(raf::value::ScalarValue::make(num_inputs));
  Expr split = Call(op_dx, {dy, sections, axis});

  Array<Expr> tuple;
  for (int i = 0; i < num_inputs; i++) {
    auto split_i = tvm::relay::TupleGetItem(split, i);
    auto tuple_i = Call(op_squeeze, {split_i, axis});
    tuple.push_back(tuple_i);
  }
  return {tvm::relay::Tuple(tuple)};
}

RAF_OP_GRAD("raf.op.stack", StackGrad);

Array<Expr> ConcatenateGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                            const Expr& dy) {
  static auto op_dx = Op::Get("raf.op.split");
  static auto op_indices = Op::Get("raf.op.concatenate_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x = call->args[0];
  const Expr& axis = call->args[1];
  Expr indices = Call(op_indices, {x, axis});
  return {Call(op_dx, {dy, indices, axis})};
}

RAF_OP_GRAD("raf.op.concatenate", ConcatenateGrad);

Array<Expr> SplitGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                      const Expr& dy) {
  static auto concatenate = Op::Get("raf.op.concatenate");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  CHECK_GE(call->args.size(), 3);
  const Expr& axis = call->args[2];
  return {Call(concatenate, {dy, axis})};
}

RAF_OP_GRAD("raf.op.split", SplitGrad);

Array<Expr> MeshGridGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                         const Expr& dy) {
  using namespace raf::value;
  const CallNode* call = orig_call.as<CallNode>();
  static auto sum = Op::Get("raf.op.sum");
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

RAF_OP_GRAD("raf.op.mesh_grid", MeshGridGrad);

Array<Expr> ReverseGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                        const Expr& dy) {
  static auto op_dx = Op::Get("raf.op.reverse");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& axis = call->args[1];
  return {Call(op_dx, {dy, axis})};
}

RAF_OP_GRAD("raf.op.reverse", ReverseGrad);

Array<Expr> ReverseSequenceGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                                const Expr& dy) {
  static auto op_dx = Op::Get("raf.op.reverse_sequence");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 4);
  const Expr& seq_length = call->args[1];
  const Expr& seq_axis = call->args[2];
  const Expr& batch_axis = call->args[3];
  return {Call(op_dx, {dy, seq_length, seq_axis, batch_axis})};
}

RAF_OP_GRAD("raf.op.reverse_sequence", ReverseSequenceGrad);

Array<Expr> ClipGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                     const Expr& dy) {
  static auto op_dx = Op::Get("raf.op.clip_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 3);
  const Expr& x = call->args[0];
  const Expr& a_min = call->args[1];
  const Expr& a_max = call->args[2];
  return {Call(op_dx, {x, dy, a_min, a_max})};
}

RAF_OP_GRAD("raf.op.clip", ClipGrad);

RAF_OP_GRAD("raf.op.expand_dims", ReshapeOpGrad);

RAF_OP_GRAD("raf.op.reshape", ReshapeOpGrad);

RAF_OP_GRAD("raf.op.squeeze", ReshapeOpGrad);

Array<Expr> TakeGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                     const Expr& dy) {
  static auto op_dx = Op::Get("raf.op.take_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_EQ(call->args.size(), 4);
  const Expr& x = call->args[0];
  const Expr& indices = call->args[1];
  const Expr& axis = call->args[2];
  const Expr& mode = call->args[3];
  return {Call(op_dx, {x, dy, indices, axis, mode})};
}

RAF_OP_GRAD("raf.op.take", TakeGrad);

Array<Expr> EmbeddingGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                          const Expr& dy) {
  static auto op_dx = Op::Get("raf.op.embedding_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_EQ(call->args.size(), 2);
  const Expr& x = call->args[0];
  const Expr& indices = call->args[1];
  return {Call(op_dx, {dy, indices, GetShape(x)})};
}

RAF_OP_GRAD("raf.op.embedding", EmbeddingGrad);

Array<Expr> ScatterGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                        const Expr& dy) {
  static auto op_dx = Op::Get("raf.op.scatter_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_EQ(call->args.size(), 4);
  const Expr& x = call->args[0];
  const Expr& index = call->args[1];
  const Expr& src = call->args[2];
  const Expr& axis = call->args[3];
  return {Call(op_dx, {x, y, dy, index, src, axis})};
}

RAF_OP_GRAD("raf.op.scatter", ScatterGrad);

Array<Expr> CastGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                     const Expr& dy) {
  static auto cast_op = Op::Get("raf.op.cast");
  static auto cast_like_op = Op::Get("raf.op.cast_like");

  const CallNode* call = orig_call.as<CallNode>();
  const Expr& x = call->args[0];
  if (x->checked_type_.defined()) {  // Use cast op if we know the target type.
    auto ttype = x->checked_type().as<TensorTypeNode>();
    CHECK(ttype != nullptr);
    auto dl_dtype = ttype->dtype.operator DLDataType();
    return {Call(cast_op, {dy, MakeConstant(raf::value::StringValue::make(
                                   tvm::runtime::DLDataType2String(dl_dtype)))})};
  }
  return {Call(cast_like_op, {dy, x})};
}

RAF_OP_GRAD("raf.op.cast", CastGrad);

Array<Expr> GatherGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                       const Expr& dy) {
  static auto gather_dx = Op::Get("raf.op.gather_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& data = call->args[0];
  const Expr& axis = call->args[1];
  const Expr& indices = call->args[2];
  return {Call(gather_dx, {data, axis, indices, dy})};
}

RAF_OP_GRAD("raf.op.gather", GatherGrad);

Array<Expr> GatherNdGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                         const Expr& dy) {
  static auto gather_nd_dx = Op::Get("raf.op.gather_nd_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& data = call->args[0];
  const Expr& indices = call->args[1];
  return {Call(gather_nd_dx, {data, indices, dy})};
}

RAF_OP_GRAD("raf.op.gather_nd", GatherNdGrad);

RAF_OP_GRAD("raf.op.full", NoGrads<0>);

RAF_OP_GRAD("raf.op.full_like", NoGrads<1>);

Array<Expr> StridedSliceGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                             const Expr& dy) {
  static auto op_slice_dx = Op::Get("raf.op.strided_slice_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& begin = call->args[1];
  const Expr& end = call->args[2];
  const Expr& strides = call->args[3];
  const Expr& mode = call->args[4];
  return {Call(op_slice_dx, {dy, GetShape(call->args[0]), begin, end, strides, mode})};
}

RAF_OP_GRAD("raf.op.strided_slice", StridedSliceGrad);

Array<Expr> WhereGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                      const Expr& dy) {
  static auto multiply = Op::Get("raf.op.multiply");
  static auto cast = Op::Get("raf.op.cast");
  static auto logical_not = Op::Get("raf.op.logical_not");

  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);

  // Note that sum_like with x1 and x2 should be simplified if their shapes are static.
  const Expr& x1 = call->args[1];
  const Expr& x2 = call->args[2];

  // Cast condition to align x1, x2, dy dtype for multiply.
  const Expr& cond = call->args[0];
  std::string dtype = "float32";
  if (x1->checked_type_.defined()) {
    auto ttype = x1->checked_type().as<TensorTypeNode>();
    CHECK(ttype != nullptr);
    auto dl_dtype = ttype->dtype.operator DLDataType();
    dtype = tvm::runtime::DLDataType2String(dl_dtype);
  }
  auto casted_cond = Call(cast, {cond, MakeConstant(raf::value::StringValue::make(dtype))});

  // Generate not condition. Note that logical_not in TVM only accepts bool dtype,
  // so we have to cast condition to bool if it is not.
  auto bool_cond = cond;
  bool cast_to_bool = true;
  if (cond->checked_type_.defined()) {
    auto ttype = cond->checked_type().as<TensorTypeNode>();
    CHECK(ttype != nullptr);
    auto dl_dtype = ttype->dtype.operator DLDataType();
    if (tvm::runtime::DLDataType2String(dl_dtype) == "bool") {
      cast_to_bool = false;
    }
  }
  if (cast_to_bool) {
    bool_cond = Call(cast, {cond, MakeConstant(raf::value::StringValue::make("bool"))});
  }
  auto not_cond = Call(logical_not, {bool_cond});
  auto casted_not_cond = Call(cast, {not_cond, MakeConstant(raf::value::StringValue::make(dtype))});

  const Expr& dx1 = Call(multiply, {dy, casted_cond});
  const Expr& dx2 = Call(multiply, {dy, casted_not_cond});
  return {NullValue<Expr>(), GetCollapseSumLike(dx1, x1), GetCollapseSumLike(dx2, x2)};
}

RAF_OP_GRAD("raf.op.where", WhereGrad);

RAF_OP_GRAD("raf.op.argwhere", NoGrads<1>);

RAF_OP_GRAD("raf.op.ndarray_size", NoGrads<1>);

Array<Expr> Resize2dGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                         const Expr& dy) {
  static auto resize2d_dx = Op::Get("raf.op.resize2d_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& x = call->args[0];
  const Expr& size = call->args[1];
  const Expr& layout = call->args[2];
  const Expr& method = call->args[3];
  const Expr& coordinate_transformation_mode = call->args[4];
  const Expr& rounding_method = call->args[5];
  const Expr& cubic_alpha = call->args[6];
  const Expr& cubic_exclude = call->args[7];
  const Expr& out_dtype = call->args[8];
  return {Call(resize2d_dx, {x, dy, size, layout, method, coordinate_transformation_mode,
                             rounding_method, cubic_alpha, cubic_exclude, out_dtype})};
}

RAF_OP_GRAD("raf.op.resize2d", Resize2dGrad);

RAF_OP_GRAD("raf.op.arange", NoGrads<0>);
RAF_OP_GRAD("raf.op.zeros", NoGrads<0>);
RAF_OP_GRAD("raf.op.ones", NoGrads<0>);

Array<Expr> CumsumGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                       const Expr& dy) {
  static auto op_cumsum = Op::Get("raf.op.cumsum");
  static auto op_flip = Op::Get("raf.op.reverse");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& axis = call->args[1];
  const Expr& dtype = call->args[2];
  const Expr& exclusive = call->args[3];

  Expr reverse = Call(op_flip, {dy, axis});
  Expr cumsum_reverse = Call(op_cumsum, {reverse, axis, dtype, exclusive});
  Expr result = Call(op_flip, {cumsum_reverse, axis});

  return {result};
}

RAF_OP_GRAD("raf.op.cumsum", CumsumGrad);

RAF_OP_GRAD("raf.op.size", NoGrads<1>);

}  // namespace grad
}  // namespace op
}  // namespace raf
