/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/transform.cc
 * \brief Typing of transform operators
 */
#include "raf/type.h"
#include "raf/op_utils.h"
#include "../schema/ufunc.h"
#include "../schema/nn.h"
#include "../schema/likes.h"
#include "../schema/transform.h"
#include "../declare/declare_utils.h"
#include "../../common/shape_utils.h"
#include "./utils.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;
using namespace schema;
using declare::NormalizeAxis;

Type ArangeInfer(const CallValues& value) {
  const auto* args = value->args.as<ArangeArgs>();
  ICHECK(args != nullptr);
  auto out_type = DataType(ir::String2DLDataType(args->dtype));
  if (args->start->IsInstance<TensorValueObj>() && args->stop->IsInstance<TensorValueObj>() &&
      args->step->IsInstance<TensorValueObj>()) {
    TensorType x = Downcast<TensorType>(GetType(args->start));
    int32_t size;
    if (args->dtype == "float32") {
      size = CalArangeOutputSize<float>(args);
    } else if (args->dtype == "float64") {
      size = CalArangeOutputSize<double>(args);
    } else if (args->dtype == "int32") {
      size = CalArangeOutputSize<int32_t>(args);
    } else if (args->dtype == "int64") {
      size = CalArangeOutputSize<int64_t>(args);
    } else {
      LOG(FATAL) << "Do not support type: " << args->dtype;
    }
    auto out_type = DataType(String2DLDataType(args->dtype));
    return TensorType({PrimExpr(size)}, out_type);
  } else {
    return TensorType({tvm::tir::Any()}, out_type);
  }
}

RAF_OP_TYPE("raf.op.arange", "Arange", ArangeInfer);

Type AdvIndexInfer(const CallValues& value) {
  const auto* args = value->args.as<AdvIndexArgs>();
  CHECK(args != nullptr);

  auto data = Downcast<TensorType>(GetType(args->inputs[0]));

  Array<IndexExpr> oshape;
  Array<IndexExpr> broadcast_shape;

  ICHECK_LE(args->inputs.size() - 1, data->shape.size()) << "too many indices for data!";

  broadcast_shape = Downcast<TensorType>(GetType(args->inputs[1]))->shape;
  for (size_t i = 2; i < args->inputs.size(); ++i) {
    broadcast_shape = BroadcastShape(TensorType(broadcast_shape, data->dtype),
                                     Downcast<TensorType>(GetType(args->inputs[i])));
  }

  for (const auto& dim : broadcast_shape) {
    oshape.push_back(dim);
  }
  for (size_t i = args->inputs.size() - 1; i < data->shape.size(); ++i) {
    oshape.push_back(data->shape[i]);
  }
  return TensorType(oshape, data->dtype);
}

RAF_OP_TYPE("raf.op.adv_index", "AdvIndex", AdvIndexInfer);

Type AdvIndexDxInfer(const CallValues& value) {
  const auto* args = value->args.as<AdvIndexDxArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->inputs[0]));
  return TupleType({x});
}

RAF_OP_TYPE("raf.op.adv_index_dx", "AdvIndexDx", AdvIndexDxInfer);

Type TransposeInfer(const CallValues& value) {
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

RAF_OP_TYPE("raf.op.transpose", "Transpose", TransposeInfer);

Type TransposeDxInfer(const CallValues& value) {
  const auto* args = value->args.as<TransposeArgs>();
  CHECK(args != nullptr);
  std::vector<int64_t> axes(args->axes.size(), -1);
  TensorType dy = Downcast<TensorType>(GetType(args->x));
  size_t ndim = dy->shape.size();

  Array<tvm::PrimExpr> oshape;
  if (axes.size() != 0) {
    for (size_t i = 0; i < ndim; ++i) {
      axes[args->axes[i]] = i;
    }
    CHECK_EQ(axes.size(), ndim);
    for (size_t i = 0; i < ndim; ++i) {
      int64_t axis = axes[i] < 0 ? axes[i] + ndim : axes[i];
      oshape.push_back(dy->shape[axis]);
    }
  } else {
    for (size_t i = 0; i < ndim; ++i) {
      oshape.push_back(dy->shape[ndim - i - 1]);
    }
  }
  return TensorType(oshape, dy->dtype);
}

RAF_OP_TYPE("raf.op.transpose_dx", "TransposeDx", TransposeDxInfer);

Type RepeatDxInfer(const CallValues& value) {
  const auto* args = value->args.as<RepeatDxArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

RAF_OP_TYPE("raf.op.repeat_dx", "RepeatDx", RepeatDxInfer);

Type SwapAxisInfer(const CallValues& value) {
  const auto* args = value->args.as<SwapAxisArgs>();
  TensorType x = Downcast<TensorType>(GetType(args->x));
  size_t ndim = x->shape.size();
  int axis1 = args->axis1;
  int axis2 = args->axis2;
  CHECK_NE(axis1, axis2);
  Array<tvm::PrimExpr> oshape;
  for (int i = 0; i < ndim; i++) {
    if (axis1 == i) {
      oshape.push_back(x->shape[axis2]);
    } else if (axis2 == i) {
      oshape.push_back(x->shape[axis1]);
    } else {
      oshape.push_back(x->shape[i]);
    }
  }
  return TensorType(oshape, x->dtype);
}

RAF_OP_TYPE("raf.op.swap_axis", "SwapAxis", SwapAxisInfer);

Type BatchFlattenInfer(const CallValues& value) {
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

RAF_OP_TYPE("raf.op.batch_flatten", "BatchFlatten", BatchFlattenInfer);

Type ReshapeInfer(const CallValues& value) {
  const auto* args = value->args.as<ReshapeArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  int ndim = x->shape.size();
  Array<PrimExpr> shape = GetShapeExprFromValue(args->shape);
  bool found_any = std::any_of(shape.begin(), shape.end(),
                               [](PrimExpr expr) { return expr->IsInstance<tvm::tir::AnyNode>(); });
  if (found_any) {
    Array<PrimExpr> newshape;
    for (size_t i = 0; i < shape.size(); ++i) {
      auto dim = shape[i];
      if (auto* imm = dim.as<IntImmNode>()) {
        int32_t val = imm->value;
        if (val == -1) {
          newshape.push_back(tvm::tir::Any());
        } else if (val == 0) {
          ICHECK_LT(i, ndim);
          newshape.push_back(x->shape[i]);
        } else {
          ICHECK_GE(val, 1);
          newshape.push_back(tvm::tir::Any());
        }
      } else if (auto* imm = dim.as<tvm::tir::AnyNode>()) {
        newshape.push_back(dim);
      }
    }
    return TensorType(newshape, x->dtype);
  }
  bool reverse = args->reverse;
  PrimExpr size = 1;
  int tbd = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (TypeCheckCompare(shape[i], -1, std::equal_to<int>())) {
      CHECK_EQ(tbd, -1);
      tbd = i;
    } else {
      if (TypeCheckCompare(shape[i], 0, std::equal_to<int>())) {
        if (reverse) {
          CHECK_GE(ndim - (shape.size() - i), 0);
          shape.Set(i, x->shape[ndim - (shape.size() - i)]);
        } else {
          CHECK(i < ndim);
          shape.Set(i, x->shape[i]);
        }
      }
      size = size * shape[i];
    }
  }
  if (tbd >= 0) {
    PrimExpr x_size = 1;
    for (int i = 0; i < ndim; ++i) {
      x_size *= x->shape[i];
    }
    CHECK(TypeCheckCompare(truncmod(x_size, size), 0, std::equal_to<int>()));
    shape.Set(tbd, div(x_size, size));
  }
  // check if reshaped shape is equal to the origin shape
  PrimExpr origin = 1;
  PrimExpr reshaped = 1;
  for (auto s : x->shape) {
    origin *= s;
  }
  for (auto s : shape) {
    reshaped *= s;
  }
  CHECK(TypeCheckCompare(origin, reshaped, std::equal_to<int>()))
      << "ValueError: Number of elements mismatch after reshaping!";
  return TensorType(shape, x->dtype);
}

RAF_OP_TYPE("raf.op.reshape", "Reshape", ReshapeInfer);

Type TakeInfer(const CallValues& value) {
  const auto* args = value->args.as<TakeArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  TensorType indices = Downcast<TensorType>(GetType(args->indices));
  int ndim = x->shape.size();
  std::vector<PrimExpr> shape_vec;
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    int axis = NormalizeAxis(v->value, ndim);
    int i = 0;
    for (; i < axis; ++i) {
      shape_vec.push_back(x->shape[i]);
    }
    for (auto& s : indices->shape) {
      shape_vec.push_back(s);
    }
    for (++i; i < ndim; ++i) {
      shape_vec.push_back(x->shape[i]);
    }
  } else {
    for (auto& s : indices->shape) {
      shape_vec.push_back(s);
    }
  }
  Array<PrimExpr> shape(shape_vec.begin(), shape_vec.end());
  return TensorType(shape, x->dtype);
}

RAF_OP_TYPE("raf.op.take", "Take", TakeInfer);

Type EmbeddingInfer(const CallValues& value) {
  const auto* args = value->args.as<EmbeddingArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  TensorType indices = Downcast<TensorType>(GetType(args->indices));
  int ndim = x->shape.size();
  std::vector<PrimExpr> shape_vec;
  for (auto& s : indices->shape) {
    shape_vec.push_back(s);
  }
  for (int i = 1; i < ndim; ++i) {
    shape_vec.push_back(x->shape[i]);
  }
  Array<PrimExpr> shape(shape_vec.begin(), shape_vec.end());
  return TensorType(shape, x->dtype);
}

RAF_OP_TYPE("raf.op.embedding", "Embedding", EmbeddingInfer);

Type TakeDxInfer(const CallValues& value) {
  const auto* args = value->args.as<TakeDxArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

RAF_OP_TYPE("raf.op.take_dx", "TakeDx", TakeDxInfer);

Type EmbeddingDxInfer(const CallValues& value) {
  const auto* args = value->args.as<EmbeddingDxArgs>();
  CHECK(args != nullptr);
  TensorType dy = Downcast<TensorType>(GetType(args->dy));
  auto shape = GetShapeExprFromValue(args->num_weight);
  return TensorType(shape, dy->dtype);
}

RAF_OP_TYPE("raf.op.embedding_dx", "EmbeddingDx", EmbeddingDxInfer);

Type ConcatenateInfer(const CallValues& value) {
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
        CHECK(TypeCheckCompare(y->shape[k], y0->shape[k], std::equal_to<int>()));
      }
    }
    dimsize += y->shape[axis];
  }
  Array<PrimExpr> shape(y0->shape.begin(), y0->shape.end());
  shape.Set(axis, dimsize);
  return TensorType(shape, y0->dtype);
}

RAF_OP_TYPE("raf.op.concatenate", "Concatenate", ConcatenateInfer);

Type SplitInfer(const CallValues& value) {
  const auto* args = value->args.as<SplitArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  int ndim = x->shape.size();
  int axis = NormalizeAxis(args->axis, ndim);
  Array<Type> ret;
  Value indices_or_sections = args->indices_or_sections;

  if (const auto* scalar = indices_or_sections.as<IntValueObj>()) {
    // Handling first type - integer scalar - sections
    int64_t sections = scalar->value;
    PrimExpr able_divide = truncmod(x->shape[axis], Integer(sections));
    CHECK(TypeCheckCompare(able_divide, 0, std::equal_to<int>()))
        << "indices_or_sections need to be able to divide input.shape[axis]";

    for (size_t i = 0; i < sections; ++i) {
      Array<PrimExpr> oshape;
      for (int j = 0; j < ndim; j++) {
        oshape.push_back(x->shape[j]);
      }
      oshape.Set(axis, div(x->shape[axis], Integer(sections)));
      ret.push_back(TensorType(oshape, x->dtype));
    }
  } else if (const auto* tup = indices_or_sections.as<TupleValueObj>()) {
    // Handling second type - tuple values - indices
    Array<PrimExpr> indices;
    for (auto field : tup->fields) {
      indices.push_back(GetIntExprFromValue(field));
    }
    indices.push_back(x->shape[axis]);
    PrimExpr begin(0);
    for (size_t i = 0; i < indices.size(); ++i) {
      Array<PrimExpr> oshape(x->shape.begin(), x->shape.end());
      oshape.Set(axis, indices[i] - begin);
      begin = indices[i];
      ret.push_back(TensorType(oshape, x->dtype));
    }
  } else {
    // TODO: handle tensor value?
    LOG(FATAL) << "Unsupported value type: " << indices_or_sections->GetTypeKey();
  }
  return TupleType(ret);
}

RAF_OP_TYPE("raf.op.split", "Split", SplitInfer);

Type MeshGridInfer(const CallValues& value) {
  const auto* args = value->args.as<MeshGridArgs>();
  CHECK(args != nullptr);
  CHECK(args->x.size() > 0);
  Array<Type> x;
  std::transform(args->x.begin(), args->x.end(), std::back_inserter(x), GetType);
  TensorType y0 = Downcast<TensorType>(x[0]);
  Array<Type> ret;
  Array<PrimExpr> oshape;

  for (auto& i : x) {
    TensorType y = Downcast<TensorType>(i);
    CHECK(y->shape.size() == 1);
    oshape.push_back(y->shape[0]);
  }
  for (size_t i = 0; i < x.size(); ++i) {
    ret.push_back(TensorType(oshape, y0->dtype));
  }
  return TupleType(ret);
}

RAF_OP_TYPE("raf.op.mesh_grid", "MeshGrid", MeshGridInfer);

Type ClipInfer(const CallValues& value) {
  const auto* args = value->args.as<ClipArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

RAF_OP_TYPE("raf.op.clip", "Clip", ClipInfer);

Type ClipDxInfer(const CallValues& value) {
  const auto* args = value->args.as<ClipDxArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}
RAF_OP_TYPE("raf.op.clip_dx", "ClipDx", ClipDxInfer);

Type ScatterInfer(const CallValues& value) {
  const auto* args = value->args.as<ScatterArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

RAF_OP_TYPE("raf.op.scatter", "Scatter", ScatterInfer);

Type ScatterDxInfer(const CallValues& value) {
  const auto* args = value->args.as<ScatterDxArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

RAF_OP_TYPE("raf.op.scatter_dx", "ScatterDx", ScatterDxInfer);

Type CastInfer(const CallValues& value) {
  const auto* args = value->args.as<CastArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  DataType dtype = DataType(ir::String2DLDataType(args->dtype));
  return TensorType(data->shape, dtype);
}

RAF_OP_TYPE("raf.op.cast", "Cast", CastInfer);

Type CastLikeInfer(const CallValues& value) {
  const auto* args = value->args.as<BinaryLikeArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  TensorType like_type = Downcast<TensorType>(GetType(args->like_type));
  return TensorType(x->shape, like_type->dtype);
}

RAF_OP_TYPE("raf.op.cast_like", "CastLike", CastLikeInfer);

Type GroupCastInfer(const CallValues& value) {
  const auto* args = value->args.as<GroupCastArgs>();
  CHECK(args != nullptr);
  Array<Type> ret;
  DataType dtype = DataType(ir::String2DLDataType(args->dtype));
  for (int i = 0; i < args->tensor_list.size(); ++i) {
    TensorType data = Downcast<TensorType>(GetType(args->tensor_list[i]));
    ret.push_back(TensorType(data->shape, dtype));
  }
  return TupleType(ret);
}

RAF_OP_TYPE("raf.op.group_cast", "GroupCast", GroupCastInfer);

Type ExpandDimsInfer(const CallValues& value) {
  const auto* args = value->args.as<ExpandDimsArgs>();
  CHECK(args);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  int ndim = x->shape.size();
  int axis = args->axis;
  CHECK(-ndim - 1 <= axis && axis <= ndim)
      << "ValueError: invalid axis (expand_dims) = " << axis << " on ndim = " << ndim;
  axis = axis < 0 ? axis + ndim + 1 : axis;
  int num_newaxis = args->num_newaxis;
  Array<PrimExpr> oshape;
  for (int i = 0; i < axis; ++i) {
    oshape.push_back(x->shape[i]);
  }
  for (int i = 0; i < num_newaxis; ++i) {
    oshape.push_back(1);
  }
  for (int i = axis; i < ndim; ++i) {
    oshape.push_back(x->shape[i]);
  }
  return TensorType(oshape, x->dtype);
}

RAF_OP_TYPE("raf.op.expand_dims", "ExpandDims", ExpandDimsInfer);

Type SequenceMaskInfer(const CallValues& value) {
  const auto* args = value->args.as<SequenceMaskArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  int axis = args->axis;
  int ndim = x->shape.size();
  CHECK(-ndim <= axis && axis < ndim)
      << "ValueError: invalid axis = " << axis << " on ndim = " << ndim;
  return x;
}

RAF_OP_TYPE("raf.op.sequence_mask", "SequenceMask", SequenceMaskInfer);

Type ReverseInfer(const CallValues& value) {
  const auto* args = value->args.as<ReverseArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  int axis = args->axis;
  int ndim = x->shape.size();
  CHECK(-ndim <= axis && axis < ndim)
      << "ValueError: invalid axis = " << axis << " on ndim = " << ndim;
  return x;
}

RAF_OP_TYPE("raf.op.reverse", "Reverse", ReverseInfer);

Type ReverseSequenceInfer(const CallValues& value) {
  const auto* args = value->args.as<ReverseSequenceArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  TensorType sequence_length = Downcast<TensorType>(GetType(args->sequence_length));
  int batch_axis = args->batch_axis;
  int s_ndim = sequence_length->shape.size();
  CHECK(TypeCheckCompare(s_ndim, 1, std::equal_to<int>()));
  CHECK(TypeCheckCompare(sequence_length->shape[0], x->shape[batch_axis], std::equal_to<int>()));
  return x;
}

RAF_OP_TYPE("raf.op.reverse_sequence", "ReverseSequence", ReverseSequenceInfer);

Type BroadcastToInfer(const CallValues& value) {
  const auto* args = value->args.as<BinaryToArgs>();
  CHECK(args != nullptr);
  std::vector<int64_t> shape = args->shape;
  Array<PrimExpr> oshape;
  for (auto& s : shape) {
    oshape.push_back(Integer(s));
  }
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return TensorType(oshape, x->dtype);
}

RAF_OP_TYPE("raf.op.broadcast_to", "BroadcastTo", BroadcastToInfer);

Type BinaryShapeLikeInfer(const CallValues& value) {
  const auto* args = value->args.as<BinaryLikeArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  TensorType like_type = Downcast<TensorType>(GetType(args->like_type));
  return TensorType(like_type->shape, x->dtype);
}

RAF_OP_TYPE("raf.op.broadcast_to_like", "BroadcastToLike", BinaryShapeLikeInfer);

RAF_OP_TYPE("raf.op.collapse_sum_like", "CollapseSumLike", BinaryShapeLikeInfer);

RAF_OP_TYPE("raf.op.reshape_like", "ReshapeLike", BinaryShapeLikeInfer);

Type RepeatInfer(const CallValues& value) {
  const auto* args = value->args.as<RepeatArgs>();
  CHECK(args != nullptr);
  CHECK(args->axis.defined());
  TensorType x = Downcast<TensorType>(GetType(args->x));
  int repeats = args->repeats;
  const auto* v = args->axis.as<IntValueObj>();
  int ndim = x->shape.size();
  Array<PrimExpr> shape(x->shape.begin(), x->shape.end());
  int axis = NormalizeAxis(v->value, ndim);
  shape.Set(axis, x->shape[axis] * repeats);
  return TensorType(shape, x->dtype);
}

RAF_OP_TYPE("raf.op.repeat", "Repeat", RepeatInfer);

Type StackInfer(const CallValues& value) {
  const auto* args = value->args.as<StackArgs>();
  CHECK(args != nullptr);
  CHECK(args->x.size() > 0);
  Array<Type> x;
  std::transform(args->x.begin(), args->x.end(), std::back_inserter(x), GetType);
  TensorType y0 = Downcast<TensorType>(x[0]);
  int ndim = y0->shape.size();
  int axis = args->axis;
  axis = axis < 0 ? axis + ndim + 1 : axis;
  PrimExpr stack_dim = 0;
  for (auto& i : x) {
    TensorType y = Downcast<TensorType>(i);
    CHECK(y->shape.size() == y0->shape.size());
    for (int k = 0; k < y0->shape.size(); k++) {
      CHECK(TypeCheckCompare(y->shape[k], y0->shape[k], std::equal_to<int>()));
    }
    stack_dim += 1;
  }
  Array<PrimExpr> shape;
  shape.reserve(y0->shape.size() + 1);

  for (int i = 0; i < axis; i++) {
    shape.push_back(y0->shape[i]);
  }
  shape.push_back(stack_dim);
  for (int i = axis; i < ndim; i++) {
    shape.push_back(y0->shape[i]);
  }
  return TensorType(shape, y0->dtype);
}

RAF_OP_TYPE("raf.op.stack", "Stack", StackInfer);

Type GatherInfer(const CallValues& value) {
  const auto* args = value->args.as<GatherArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  int axis = args->axis;
  TensorType indices = Downcast<TensorType>(GetType(args->indices));

  size_t idim = indices->shape.size();

  Array<PrimExpr> oshape;
  for (size_t i = 0; i < idim; i++) {
    oshape.push_back(indices->shape[i]);
  }
  return TensorType(oshape, data->dtype);
}

RAF_OP_TYPE("raf.op.gather", "Gather", GatherInfer);

Type GatherDxInfer(const CallValues& value) {
  const auto* args = value->args.as<GatherDxArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  return data;
}

RAF_OP_TYPE("raf.op.gather_dx", "GatherDx", GatherDxInfer);

Type GatherNdInfer(const CallValues& value) {
  const auto* args = value->args.as<GatherNdArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  TensorType indices = Downcast<TensorType>(GetType(args->indices));
  int indices_dim0 = indices->shape[0].as<IntImmNode>()->value;
  int ddim = data->shape.size();
  int idim = indices->shape.size();
  int odim = idim - 1 + ddim - indices_dim0;
  CHECK_LE(indices_dim0, ddim);

  Array<PrimExpr> oshape;
  for (int i = 0; i < odim; ++i) {
    if (i + 1 < idim) {
      oshape.push_back(indices->shape[i + 1]);
    } else {
      oshape.push_back(data->shape[i + 1 - idim + indices_dim0]);
    }
  }
  return TensorType(oshape, data->dtype);
}

RAF_OP_TYPE("raf.op.gather_nd", "GatherNd", GatherNdInfer);

Type GatherNdDxInfer(const CallValues& value) {
  const auto* args = value->args.as<GatherNdDxArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  return data;
}

RAF_OP_TYPE("raf.op.gather_nd_dx", "GatherNdDx", GatherNdDxInfer);

Type StridedSliceInfer(const CallValues& value) {
  const auto* args = value->args.as<StridedSliceArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->x));

  auto dshape = data->shape;
  int64_t num_axis = dshape.size();
  Array<PrimExpr> begin = GetShapeExprFromValue(args->begin);
  Array<PrimExpr> end = GetShapeExprFromValue(args->end);
  auto is_any = [](PrimExpr expr) { return expr->IsInstance<tvm::tir::AnyNode>(); };
  bool found_any_begin = std::any_of(begin.begin(), begin.end(), is_any);
  bool found_any_end = std::any_of(end.begin(), end.end(), is_any);
  if (found_any_begin || found_any_end) {
    Array<PrimExpr> oshape;
    for (int64_t i = 0; i < num_axis; ++i) {
      oshape.push_back(tvm::tir::Any());
    }
    return TensorType(oshape, data->dtype);
  }

  CHECK(!begin.empty()) << "strided_slice received invalid begin";
  CHECK(!end.empty()) << "strided_slice received invalid end";
  CHECK_EQ(begin.size(), end.size()) << "begin.size() != end.size()";

  // calculate output shape
  std::vector<PrimExpr> oshape(num_axis);
  // stride will be set as 1 if slice mode is enabled
  std::vector<int64_t> stride_vec(num_axis, 1);
  if (args->slice_mode == "end") {
    CHECK(!args->strides.empty()) << "strided_slice received invalid strides";
    CHECK_EQ(begin.size(), args->strides.size()) << "begin.size() != strides.size()";
    for (size_t i = 0; i < args->strides.size(); ++i) {
      stride_vec[i] = args->strides[i];
    }
  }
  const int64_t max_range = std::numeric_limits<int64_t>::max();
  std::vector<int64_t> begin_vec;
  for (size_t i = 0; i < begin.size(); ++i) {
    auto* imm = begin[i].as<IntImmNode>();
    begin_vec.push_back(imm->value);
  }
  for (int64_t i = begin_vec.size(); i < num_axis; ++i) {
    begin_vec.push_back(stride_vec[i] > 0 ? 0 : max_range);
  }

  std::vector<int64_t> end_vec;
  for (size_t i = 0; i < end.size(); ++i) {
    auto* imm = end[i].as<IntImmNode>();
    if (args->slice_mode == "size") {
      if (imm->value < 0) {
        end_vec.push_back(max_range);
      } else {
        end_vec.push_back(begin_vec[i] + imm->value);
      }
    } else if (args->slice_mode == "end") {
      end_vec.push_back(imm->value);
    } else {
      LOG(FATAL) << "Unsupported slice mode: " << args->slice_mode;
    }
  }
  for (int64_t i = end_vec.size(); i < num_axis; ++i) {
    end_vec.push_back(stride_vec[i] < 0 ? 0 : max_range);
  }

  for (int64_t i = 0; i < num_axis; ++i) {
    int64_t stride_v = stride_vec[i];
    int64_t begin_v = begin_vec[i];
    int64_t end_v = end_vec[i];

    if ((stride_v == 1 && begin_v == 0 && end_v == max_range) ||
        (stride_v == -1 && begin_v == max_range && end_v == 0)) {
      // Quick path, do not slice this dimension.
      oshape[i] = dshape[i];
      continue;
    }
    // Normal path, require the shape to be concrete integer.
    // Require concrete integer as symbolic inference of min/max
    // can get complicated and not very helpful.
    auto dim_size_expr = dshape[i];
    const auto* dim_size_int = dim_size_expr.as<IntImmNode>();
    if (dim_size_int == nullptr) {
      // TODO: try infer it
      oshape[i] = tvm::tir::Any();
      continue;
    }
    int64_t dim_size = dim_size_int->value;

    begin_v = (begin_v < 0) ? dim_size + begin_v : begin_v;
    end_v = (end_v < 0) ? dim_size + end_v : end_v;

    int64_t slice_range;
    int64_t step;
    if (stride_v < 0) {
      if (end_v < -1) end_v = -1;
      CHECK_LE(end_v, begin_v) << "strided_slice get empty slice at axis " << i;
      begin_v = std::min(dim_size - 1, begin_v);
      slice_range = begin_v - end_v;
      step = -stride_v;
    } else {
      if (begin_v < 0) begin_v = 0;
      CHECK_GE(stride_v, 0);
      CHECK_LE(begin_v, end_v) << "strided_slice get invalid slice at axis " << i;
      end_v = std::min(dim_size, end_v);
      slice_range = end_v - begin_v;
      step = stride_v;
    }
    CHECK_NE(step, 0) << "step can not be zero ";
    oshape[i] = Integer((slice_range + step - 1) / step);
  }

  return TensorType(oshape, data->dtype);
}

Type StridedSliceDxInfer(const CallValues& value) {
  const auto* args = value->args.as<StridedSliceDxArgs>();
  CHECK(args != nullptr);
  TensorType dy = Downcast<TensorType>(GetType(args->dy));
  Array<tvm::PrimExpr> oshape = GetShapeExprFromValue(args->shape);
  return TensorType(oshape, dy->dtype);
}

RAF_OP_TYPE("raf.op.strided_slice", "StridedSlice", StridedSliceInfer);
RAF_OP_TYPE("raf.op.strided_slice_dx", "StridedSliceDx", StridedSliceDxInfer);

Type StridedSetInfer(const CallValues& value) {
  const auto* args = value->args.as<StridedSetArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  return data;
}

RAF_OP_TYPE("raf.op.strided_set", "StridedSet", StridedSetInfer);

Type SqueezeInfer(const CallValues& value) {
  const auto* args = value->args.as<SqueezeArgs>();
  CHECK(args != nullptr);
  const std::vector<int64_t>& axis = args->axis;
  TensorType xtype = Downcast<TensorType>(GetType(args->x));
  auto ishape = xtype->shape;
  int ndim = ishape.size();
  std::vector<int64_t> normalized_axis;

  for (int i = 0; i < axis.size(); i++) {
    normalized_axis.push_back(axis[i] >= 0 ? axis[i] : axis[i] + ndim);
  }

  Array<PrimExpr> oshape;
  if (normalized_axis.size() != 0) {
    for (int axis_dim = 0; axis_dim < ndim; ++axis_dim) {
      if (std::find(normalized_axis.begin(), normalized_axis.end(), axis_dim) ==
          normalized_axis.end()) {
        oshape.push_back(ishape[axis_dim]);
      } else {
        CHECK(TypeCheckCompare(ishape[axis_dim], 1, std::equal_to<int>()))
            << "Axis to be squeezed is not of size 1";
      }
    }
  } else {
    for (int axis_dim = 0; axis_dim < ndim; ++axis_dim) {
      if (!TypeCheckCompare(ishape[axis_dim], 1, std::equal_to<int>())) {
        oshape.push_back(ishape[axis_dim]);
      }
    }
  }
  return TensorType(oshape, xtype->dtype);
}

RAF_OP_TYPE("raf.op.squeeze", "Squeeze", SqueezeInfer);

Type FullInfer(const CallValues& value) {
  const auto* args = value->args.as<FullArgs>();
  CHECK(args != nullptr);
  Array<tvm::PrimExpr> shape = GetShapeExprFromValue(args->shape);
  return TensorType(shape, DataType(ir::String2DLDataType(args->dtype)));
}

RAF_OP_TYPE("raf.op.full", "Full", FullInfer);

Type FullLikeInfer(const CallValues& value) {
  const auto* args = value->args.as<FullLikeArgs>();
  CHECK(args != nullptr);
  return GetType(args->data);
}

RAF_OP_TYPE("raf.op.full_like", "FullLike", FullLikeInfer);

Type WhereInfer(const CallValues& value) {
  const auto* args = value->args.as<WhereArgs>();
  CHECK(args != nullptr);

  TensorType x = Downcast<TensorType>(GetType(args->x));
  TensorType y = Downcast<TensorType>(GetType(args->y));
  if (x->shape.size() >= y->shape.size()) {
    return x;
  } else {
    return y;
  }
}

RAF_OP_TYPE("raf.op.where", "Where", WhereInfer);

Type Resize2DInfer(const CallValues& value) {
  const auto* args = value->args.as<Resize2DArgs>();
  CHECK(args != nullptr);

  TensorType x = Downcast<TensorType>(GetType(args->x));
  Array<PrimExpr> size = GetShapeExprFromValue(args->size);
  Array<PrimExpr> shape(x->shape);

  CHECK(size.size() > 0);
  if (size.size() == 1) size.push_back(size[0]);

  // setup the output tensor shape
  if (args->layout == "NCHW") {
    shape.Set(2, size[0]);
    shape.Set(3, size[1]);
  } else if (args->layout == "NHWC") {
    shape.Set(1, size[0]);
    shape.Set(2, size[1]);
  } else {
    LOG(FATAL) << "NotImplementedError: we only support NCHW and NHWC layout.";
    throw;
  }

  DataType out_dtype(String2DLDataType(args->out_dtype));
  if (args->out_dtype.size() == 0) out_dtype = x->dtype;

  return TensorType(shape, out_dtype);
}

RAF_OP_TYPE("raf.op.resize2d", "Resize2D", Resize2DInfer);

Type Resize2DDxInfer(const CallValues& value) {
  const auto* args = value->args.as<Resize2DDxArgs>();
  CHECK(args != nullptr);

  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

RAF_OP_TYPE("raf.op.resize2d_dx", "Resize2DDx", Resize2DDxInfer);

Type ArgwhereInfer(const CallValues& value) {
  const auto* args = value->args.as<ArgwhereArgs>();
  CHECK(args != nullptr);
  auto cond = args->condition;
  TensorType ty = Downcast<TensorType>(GetType(cond));
  Array<tvm::PrimExpr> shape;
  shape.push_back(Any());
  shape.push_back(int32_t(ty->shape.size()));
  return TensorType(shape, DataType::Int(32));
}

RAF_OP_TYPE("raf.op.argwhere", "Argwhere", ArgwhereInfer);

Type UpperBoundArgwhereInfer(const CallValues& value) {
  const auto* args = value->args.as<ArgwhereArgs>();
  CHECK(args != nullptr);
  auto cond = args->condition;
  TensorType ty = Downcast<TensorType>(GetType(cond));
  int32_t ndim = ty->shape.size();
  PrimExpr size = 1;
  for (size_t i = 0; i < ndim; ++i) {
    size *= ty->shape[i];
  }
  TensorType data_type = TensorType({size, ndim}, DataType::Int(32));
  TensorType shape_type = TensorType({2}, DataType::Int(64));
  return TupleType({data_type, shape_type});
}

RAF_OP_TYPE("raf.op.upper_bound.argwhere", "UpperBoundArgwhere", UpperBoundArgwhereInfer);

Type CumsumInfer(const CallValues& value) {
  const auto* args = value->args.as<CumsumArgs>();
  CHECK(args != nullptr);
  return GetType(args->x);
}

RAF_OP_TYPE("raf.op.cumsum", "Cumsum", CumsumInfer);

Type SizeInfer(const CallValues& value) {
  const auto* args = value->args.as<SizeArgs>();
  TensorType x = Downcast<TensorType>(GetType(args->x));
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    return TensorType({}, tvm::runtime::DataType::Int(32));
  }
  Array<Type> out_types;
  for (int i = 0; i < x->shape.size(); ++i) {
    out_types.push_back(TensorType({}, tvm::runtime::DataType::Int(32)));
  }
  return TupleType(out_types);
}

RAF_OP_TYPE("raf.op.size", "Size", SizeInfer);

}  // namespace op
}  // namespace raf
