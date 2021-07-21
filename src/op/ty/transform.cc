/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/transform.cc
 * \brief Typing of transform operators
 */
#include "mnm/type.h"
#include "../schema/ufunc.h"
#include "../schema/nn.h"
#include "../schema/likes.h"
#include "../schema/transform.h"
#include "../declare/declare_utils.h"
#include "../../common/shape_utils.h"
#include "./utils.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace mnm::value;
using namespace schema;
using declare::NormalizeAxis;

Type ArangeInfer(const CallValues& value) {
  const auto* args = value->args.as<ArangeArgs>();
  if (args->start->IsInstance<TensorValueObj>() && args->stop->IsInstance<TensorValueObj>() &&
      args->step->IsInstance<TensorValueObj>()) {
    TensorType x = Downcast<TensorType>(GetType(args->start));
    int32_t size;
    if (args->dtype == "float32") {
      size = CalArangeOutputSize<float>(args);
    } else if (args->dtype == "float64") {
      size = CalArangeOutputSize<double>(args);
    } else if (args->dtype == "int64") {
      size = CalArangeOutputSize<int64_t>(args);
    } else {
      LOG(FATAL) << "Do not support type: " << args->dtype;
    }
    auto out_type = DataType(String2DLDataType(args->dtype));
    return TensorType({PrimExpr(size)}, out_type);
  } else {
    return IncompleteType(tvm::kType);
  }
}

MNM_OP_TYPE("mnm.op.arange", "Arange", ArangeInfer);

Type AdvIndexInfer(const CallValues& value) {
  const auto* args = value->args.as<AdvIndexArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->inputs[0]));
  auto x_shape = x->shape;
  auto x_dim = x->shape.size();
  TensorType index0 = Downcast<TensorType>(GetType(args->inputs[1]));
  std::vector<int64_t> shape;
  for (int j = 0; j < index0->shape.size(); ++j) {
    shape.push_back(index0->shape[j].as<IntImmNode>()->value);
  }
  if (args->inputs.size() - 1 < x_dim) {
    for (int j = args->inputs.size() - 1; j < x_dim; ++j) {
      shape.push_back(x_shape[j].as<IntImmNode>()->value);
    }

  } else {
    for (int i = 2; i < args->inputs.size(); ++i) {
      TensorType index = Downcast<TensorType>(GetType(args->inputs[i]));
      for (int j = 0; j < index->shape.size(); ++j) {
        shape[j] = shape[j] == 1 ? index->shape[j].as<IntImmNode>()->value : shape[j];
      }
    }
  }
  Array<tvm::PrimExpr> oshape;
  for (int i = 0; i < shape.size(); ++i) {
    int32_t s = shape[i];
    oshape.push_back(PrimExpr(s));
  }
  return TensorType(oshape, x->dtype);
}

MNM_OP_TYPE("mnm.op.adv_index", "AdvIndex", AdvIndexInfer);

Type AdvIndexDxInfer(const CallValues& value) {
  const auto* args = value->args.as<AdvIndexDxArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->inputs[0]));
  return TupleType({x});
}

MNM_OP_TYPE("mnm.op.adv_index_dx", "AdvIndexDx", AdvIndexDxInfer);

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

MNM_OP_TYPE("mnm.op.transpose", "Transpose", TransposeInfer);

Type TransposeDxInfer(const CallValues& value) {
  const auto* args = value->args.as<TransposeDxArgs>();
  CHECK(args != nullptr);
  TensorType dy = Downcast<TensorType>(GetType(args->dy));
  Array<tvm::PrimExpr> oshape;
  for (auto dim : args->primal_shape) {
    oshape.push_back(IntImm(DataType::Int(32), dim));
  }
  return TensorType(oshape, dy->dtype);
}

MNM_OP_TYPE("mnm.op.transpose_dx", "TransposeDx", TransposeDxInfer);

Type RepeatDxInfer(const CallValues& value) {
  const auto* args = value->args.as<RepeatDxArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

MNM_OP_TYPE("mnm.op.repeat_dx", "RepeatDx", RepeatDxInfer);

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

MNM_OP_TYPE("mnm.op.swap_axis", "SwapAxis", SwapAxisInfer);

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

MNM_OP_TYPE("mnm.op.batch_flatten", "BatchFlatten", BatchFlattenInfer);

Type ReshapeInfer(const CallValues& value) {
  const auto* args = value->args.as<ReshapeArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  int ndim = x->shape.size();
  Array<PrimExpr> shape;
  for (auto& s : args->shape) {
    shape.push_back(Integer(s));
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

MNM_OP_TYPE("mnm.op.reshape", "Reshape", ReshapeInfer);

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

MNM_OP_TYPE("mnm.op.take", "Take", TakeInfer);

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

MNM_OP_TYPE("mnm.op.embedding", "Embedding", EmbeddingInfer);

Type TakeDxInfer(const CallValues& value) {
  const auto* args = value->args.as<TakeDxArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

MNM_OP_TYPE("mnm.op.take_dx", "TakeDx", TakeDxInfer);

Type EmbeddingDxInfer(const CallValues& value) {
  const auto* args = value->args.as<EmbeddingDxArgs>();
  CHECK(args != nullptr);
  TensorType dy = Downcast<TensorType>(GetType(args->dy));
  std::vector<PrimExpr> shape;
  for (auto val : args->num_weight) {
    shape.push_back(Integer(val));
  }
  return TensorType(shape, dy->dtype);
}

MNM_OP_TYPE("mnm.op.embedding_dx", "EmbeddingDx", EmbeddingDxInfer);

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

MNM_OP_TYPE("mnm.op.concatenate", "Concatenate", ConcatenateInfer);

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
      auto int_value = field.as<IntValueObj>();
      indices.push_back(Integer(int_value->value));
    }
    indices.push_back(x->shape[axis]);
    PrimExpr begin(0);
    for (size_t i = 0; i < indices.size(); ++i) {
      Array<PrimExpr> oshape(x->shape.begin(), x->shape.end());
      oshape.Set(axis, indices[i] - begin);
      begin = indices[i];
      ret.push_back(TensorType(oshape, x->dtype));
    }
  }
  return TupleType(ret);
}

MNM_OP_TYPE("mnm.op.split", "Split", SplitInfer);

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

MNM_OP_TYPE("mnm.op.mesh_grid", "MeshGrid", MeshGridInfer);

Type ClipInfer(const CallValues& value) {
  const auto* args = value->args.as<ClipArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

MNM_OP_TYPE("mnm.op.clip", "Clip", ClipInfer);

Type ClipDxInfer(const CallValues& value) {
  const auto* args = value->args.as<ClipDxArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}
MNM_OP_TYPE("mnm.op.clip_dx", "ClipDx", ClipDxInfer);

Type ScatterInfer(const CallValues& value) {
  const auto* args = value->args.as<ScatterArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

MNM_OP_TYPE("mnm.op.scatter", "Scatter", ScatterInfer);

Type ScatterDxInfer(const CallValues& value) {
  const auto* args = value->args.as<ScatterDxArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

MNM_OP_TYPE("mnm.op.scatter_dx", "ScatterDx", ScatterDxInfer);

Type CastInfer(const CallValues& value) {
  const auto* args = value->args.as<CastArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  DataType dtype = DataType(ir::String2DLDataType(args->dtype));
  return TensorType(data->shape, dtype);
}

MNM_OP_TYPE("mnm.op.cast", "Cast", CastInfer);

Type CastLikeInfer(const CallValues& value) {
  const auto* args = value->args.as<CastLikeArgs>();
  CHECK(args != nullptr);
  TensorType dtype_like = Downcast<TensorType>(GetType(args->dtype_like));
  return dtype_like;
}

MNM_OP_TYPE("mnm.op.cast_like", "CastLike", CastLikeInfer);

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

MNM_OP_TYPE("mnm.op.expand_dims", "ExpandDims", ExpandDimsInfer);

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

MNM_OP_TYPE("mnm.op.sequence_mask", "SequenceMask", SequenceMaskInfer);

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

MNM_OP_TYPE("mnm.op.reverse", "Reverse", ReverseInfer);

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

MNM_OP_TYPE("mnm.op.reverse_sequence", "ReverseSequence", ReverseSequenceInfer);

Type BroadcastToInfer(const CallValues& value) {
  const auto* args = value->args.as<BroadcastToArgs>();
  CHECK(args != nullptr);
  std::vector<int64_t> shape = args->shape;
  Array<PrimExpr> oshape;
  for (auto& s : shape) {
    oshape.push_back(Integer(s));
  }
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return TensorType(oshape, x->dtype);
}

MNM_OP_TYPE("mnm.op.broadcast_to", "BroadcastTo", BroadcastToInfer);

Type BroadcastToLikeInfer(const CallValues& value) {
  const auto* args = value->args.as<BroadcastToLikeArgs>();
  CHECK(args != nullptr);
  TensorType broadcast_type = Downcast<TensorType>(GetType(args->broadcast_type));
  return broadcast_type;
}

MNM_OP_TYPE("mnm.op.broadcast_to_like", "BroadcastToLike", BroadcastToLikeInfer);

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

MNM_OP_TYPE("mnm.op.repeat", "Repeat", RepeatInfer);

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

MNM_OP_TYPE("mnm.op.stack", "Stack", StackInfer);

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

MNM_OP_TYPE("mnm.op.gather", "Gather", GatherInfer);

Type GatherDxInfer(const CallValues& value) {
  const auto* args = value->args.as<GatherDxArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  return data;
}

MNM_OP_TYPE("mnm.op.gather_dx", "GatherDx", GatherDxInfer);

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

MNM_OP_TYPE("mnm.op.gather_nd", "GatherNd", GatherNdInfer);

Type GatherNdDxInfer(const CallValues& value) {
  const auto* args = value->args.as<GatherNdDxArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  return data;
}

MNM_OP_TYPE("mnm.op.gather_nd_dx", "GatherNdDx", GatherNdDxInfer);

Type StridedSliceInfer(const CallValues& value) {
  const auto* args = value->args.as<StridedSliceArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->x));

  auto dshape = data->shape;
  int64_t num_axis = dshape.size();

  CHECK(!args->begin.empty()) << "strided_slice received invalid begin";
  CHECK(!args->end.empty()) << "strided_slice received invalid end";
  CHECK_EQ(args->begin.size(), args->end.size()) << "begin.size() != end.size()";

  // calculate output shape
  std::vector<PrimExpr> oshape(num_axis);
  // stride will be set as 1 if slice mode is enabled
  std::vector<int64_t> stride_vec(num_axis, 1);
  if (args->slice_mode == "end") {
    CHECK(!args->strides.empty()) << "strided_slice received invalid strides";
    CHECK_EQ(args->begin.size(), args->strides.size()) << "begin.size() != strides.size()";
    for (size_t i = 0; i < args->strides.size(); ++i) {
      stride_vec[i] = args->strides[i];
    }
  }
  const int64_t max_range = std::numeric_limits<int64_t>::max();
  std::vector<int64_t> begin_vec;
  for (size_t i = 0; i < args->begin.size(); ++i) {
    begin_vec.push_back(args->begin[i]);
  }
  for (int64_t i = begin_vec.size(); i < num_axis; ++i) {
    begin_vec.push_back(stride_vec[i] > 0 ? 0 : max_range);
  }

  std::vector<int64_t> end_vec;
  for (size_t i = 0; i < args->end.size(); ++i) {
    if (args->slice_mode == "size") {
      if (args->end[i] < 0) {
        end_vec.push_back(max_range);
      } else {
        end_vec.push_back(begin_vec[i] + args->end[i]);
      }
    } else if (args->slice_mode == "end") {
      end_vec.push_back(args->end[i]);
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
    CHECK(dim_size_int) << "Symbolic data shape is not supported yet.";
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
    oshape[i] = Integer((slice_range + step - 1) / step);
  }

  return TensorType(oshape, data->dtype);
}

Type StridedSliceDxInfer(const CallValues& value) {
  const auto* args = value->args.as<StridedSliceDxArgs>();
  CHECK(args != nullptr);
  TensorType dy = Downcast<TensorType>(GetType(args->dy));
  Array<tvm::PrimExpr> oshape;
  for (auto dim : args->primal_shape) {
    oshape.push_back(IntImm(DataType::Int(32), dim));
  }
  return TensorType(oshape, dy->dtype);
}

MNM_OP_TYPE("mnm.op.strided_slice", "StridedSlice", StridedSliceInfer);
MNM_OP_TYPE("mnm.op.strided_slice_dx", "StridedSliceDx", StridedSliceDxInfer);

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

MNM_OP_TYPE("mnm.op.squeeze", "Squeeze", SqueezeInfer);

Type FullInfer(const CallValues& value) {
  const auto* args = value->args.as<FullArgs>();
  CHECK(args != nullptr);
  Array<tvm::PrimExpr> shape;
  for (int i = 0; i < args->shape.size(); ++i) {
    CHECK_GE(args->shape[i], 1);
    shape.push_back((int32_t)args->shape[i]);
  }

  return TensorType(shape, DataType(ir::String2DLDataType(args->dtype)));
}

MNM_OP_TYPE("mnm.op.full", "Full", FullInfer);

Type FullLikeInfer(const CallValues& value) {
  const auto* args = value->args.as<FullLikeArgs>();
  CHECK(args != nullptr);
  return GetType(args->data);
}

MNM_OP_TYPE("mnm.op.full_like", "FullLike", FullLikeInfer);

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

MNM_OP_TYPE("mnm.op.where", "Where", WhereInfer);

Type Resize2DInfer(const CallValues& value) {
  const auto* args = value->args.as<Resize2DArgs>();
  CHECK(args != nullptr);

  TensorType x = Downcast<TensorType>(GetType(args->x));
  std::vector<int64_t> size(args->size);
  Array<PrimExpr> shape(x->shape);

  CHECK(size.size() > 0);
  if (size.size() == 1) size.push_back(size[0]);

  // setup the output tensor shape
  if (args->layout == "NCHW") {
    shape.Set(2, Integer(size[0]));
    shape.Set(3, Integer(size[1]));
  } else if (args->layout == "NHWC") {
    shape.Set(1, Integer(size[0]));
    shape.Set(2, Integer(size[1]));
  } else {
    LOG(FATAL) << "NotImplementedError: we only support NCHW and NHWC layout.";
    throw;
  }

  DataType out_dtype(String2DLDataType(args->out_dtype));
  if (args->out_dtype.size() == 0) out_dtype = x->dtype;

  return TensorType(shape, out_dtype);
}

MNM_OP_TYPE("mnm.op.resize2d", "Resize2D", Resize2DInfer);

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

MNM_OP_TYPE("mnm.op.argwhere", "Argwhere", ArgwhereInfer);

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

MNM_OP_TYPE("mnm.op.upper_bound.argwhere", "UpperBoundArgwhere", UpperBoundArgwhereInfer);

}  // namespace op
}  // namespace mnm
