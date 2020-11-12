/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/transform.cc
 * \brief Typing of transform operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/ufunc.h"
#include "../schema/nn.h"
#include "../schema/likes.h"
#include "../schema/transform.h"
#include "../declare/declare_utils.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using namespace mnm::value;
using namespace schema;
using declare::NormalizeAxis;
using tvm::relay::Type;
using namespace tvm;
using namespace relay;

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
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.transpose_dx", "TransposeDx", TransposeDxInfer);

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
    int axis = NormalizeAxis(v->data, ndim);
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

Type TakeDxInfer(const CallValues& value) {
  const auto* args = value->args.as<TakeDxArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

MNM_OP_TYPE("mnm.op.take_dx", "TakeDx", TakeDxInfer);

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
    int64_t sections = scalar->data;
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
      indices.push_back(Integer(int_value->data));
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

MNM_OP_TYPE("mnm.op.sequence_mask", "SequenceMask", SequenceMaskInfer)

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
  int axis = NormalizeAxis(v->data, ndim);
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

Type GatherNdInfer(const CallValues& value) {
  const auto* args = value->args.as<GatherNdArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  TensorType indices = Downcast<TensorType>(GetType(args->indices));
  PrimExpr ddim = static_cast<int>(data->shape.size());
  PrimExpr idim = static_cast<int>(indices->shape.size());
  PrimExpr odim = idim - 1 + ddim - indices->shape[0];
  CHECK(TypeCheckCompare(indices->shape[0], ddim, std::less_equal<int>()));
  Array<PrimExpr> oshape = {odim, -1};
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

}  // namespace type
}  // namespace op
}  // namespace mnm
