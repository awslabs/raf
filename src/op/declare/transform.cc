/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/transform.cc
 * \brief Declaration of transform operators
 */

#include <functional>
#include <numeric>

#include "mnm/op.h"
#include "mnm/tensor.h"
#include "./declare_utils.h"
#include "../schema/ufunc.h"
#include "../schema/likes.h"
#include "../schema/nn.h"
#include "../schema/transform.h"
#include "../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;
using common::shape_utils::IsCompact;
using tensor::Tensor;

MNM_OP_DECLARE("mnm.op.batch_flatten", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  const int ndim = x->ndim;
  CHECK_GE(ndim, 2) << "ValueError: batch_flatten only works with ndim >= 2";

  if (IsCompact(*x)) {
    const int64_t* dshape = x->shape;
    int64_t flat{1};
    for (int i = 1; i < ndim; ++i) {
      flat = flat * int64_t{dshape[i]};
    }
    call->callee = ir::NullValue<OpValue>();
    call->out = TensorValue::make(Tensor(args->x).CreateView({dshape[0], flat}, {}, nullptr));
    call->out->op_env = args->x->op_env;
    return;
  }
  LOG(FATAL) << "NotImplementedError: for now we only support batch_flatten on contiguous tensor.";
  throw;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

MNM_OP_DECLARE("mnm.op.reshape", [](const CallValues& call) {
  const auto* args = call->args.as<ReshapeArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  bool reverse = args->reverse;
  std::vector<int64_t> shape = args->shape;
  int64_t size = 1;
  int tbd = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == -1) {
      CHECK_EQ(tbd, -1);
      tbd = i;
    } else {
      if (shape[i] == 0) {
        if (reverse) {
          CHECK_GE(x->ndim - (shape.size() - i), 0);
          shape[i] = x->shape[x->ndim - (shape.size() - i)];
        } else {
          CHECK(i < x->ndim);
          shape[i] = x->shape[i];
        }
      }
      size = size * shape[i];
    }
  }
  if (tbd >= 0) {
    int64_t x_size = 1;
    for (int i = 0; i < x->ndim; ++i) {
      x_size *= x->shape[i];
    }
    CHECK_EQ(x_size % size, 0);
    shape[tbd] = x_size / size;
  }
  call->ctx = x->ctx;
  call->callee = ir::NullValue<OpValue>();
  if (IsCompact(*x)) {
    int64_t origin = std::accumulate(x->shape, x->shape + x->ndim, 1LL, std::multiplies<int64_t>());
    int64_t reshaped = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    CHECK_EQ(origin, reshaped) << "ValueError: Number of elements mismatch after reshaping!";
    call->out = TensorValue::make(Tensor(args->x).CreateView(shape));
    call->out->op_env = args->x->op_env;
    return;
  }
  LOG(FATAL) << "NotImplementedError: for now we only support reshape on contiguous tensor.";
  throw;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

MNM_OP_DECLARE("mnm.op.take", [](const CallValues& call) {
  const auto* args = call->args.as<TakeArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  DLTensor* indices = args->indices;
  std::vector<int64_t> shape;
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    int axis = NormalizeAxis(v->data, x->ndim);
    shape.insert(shape.end(), x->shape, x->shape + axis);
    shape.insert(shape.end(), indices->shape, indices->shape + indices->ndim);
    shape.insert(shape.end(), x->shape + axis + 1, x->shape + x->ndim);
  } else {
    shape.insert(shape.end(), indices->shape, indices->shape + indices->ndim);
  }
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

MNM_OP_DECLARE("mnm.op.take_dx", [](const CallValues& call) {
  const auto* args = call->args.as<TakeDxArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  ;
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

MNM_OP_DECLARE("mnm.op.expand_dims", [](const CallValues& call) {
  const auto* args = call->args.as<ExpandDimsArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  int axis = args->axis;
  int ndim = x->ndim;
  CHECK(-ndim - 1 <= axis && axis <= ndim)
      << "ValueError: invalid axis (expand_dims) = " << axis << " on ndim = " << ndim;
  axis = axis < 0 ? axis + ndim + 1 : axis;
  int num_newaxis = args->num_newaxis;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  shape.insert(shape.begin() + axis, num_newaxis, 1);
  if (IsCompact(*x)) {
    call->callee = ir::NullValue<OpValue>();
    call->out = TensorValue::make(Tensor(args->x).CreateView(shape));
    call->out->op_env = args->x->op_env;
    return;
  }
  LOG(FATAL) << "NotImplementedError: for now we only support expand_dims on contiguous tensor.";
  throw;
}).set_attr<TOpPattern>("TOpPattern", kBroadcast);

MNM_OP_DECLARE("mnm.op.sequence_mask", [](const CallValues& call) {
  const auto* args = call->args.as<SequenceMaskArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  // TODO(@hzfan): checks x.shape and sequence_length.shape
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

MNM_OP_DECLARE("mnm.op.reverse", [](const CallValues& call) {
  const auto* args = call->args.as<ReverseArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

MNM_OP_DECLARE("mnm.op.reverse_sequence", [](const CallValues& call) {
  const auto* args = call->args.as<ReverseSequenceArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  DLTensor* sequence_length = args->sequence_length;
  int batch_axis = args->batch_axis;
  int64_t* ishape = x->shape;
  int64_t* sshape = sequence_length->shape;
  int s_ndim = sequence_length->ndim;
  CHECK_EQ(s_ndim, 1);
  CHECK_EQ(sshape[0], ishape[batch_axis]);
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

MNM_OP_DECLARE("mnm.op.broadcast_to", [](const CallValues& call) {
  const auto* args = call->args.as<BroadcastToArgs>();
  DLTensor* x = args->x;
  std::vector<int64_t> shape = args->shape;
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kBroadcast);

MNM_OP_DECLARE("mnm.op.repeat", [](const CallValues& call) {
  const auto* args = call->args.as<RepeatArgs>();
  CHECK(args != nullptr);
  CHECK(args->axis.defined());
  DLTensor* x = args->x;
  const int64_t* ishape = x->shape;
  int repeat = args->repeats;
  int ndim = x->ndim;
  std::vector<int64_t> shape;

  shape.resize(x->ndim);
  int axis = args->axis.as<IntValueObj>()->data;
  CHECK(axis >= -ndim && axis < ndim)
      << "repeat only accepts `axis` in [-data.ndim, data.ndim - 1]"
      << ", but got axis = " << axis << ", and data.ndim = " << ndim;
  axis = axis >= 0 ? axis : axis + ndim;

  for (int i = 0; i < x->ndim; i++) {
    if (axis == i) {
      shape[i] = ishape[i] * repeat;
    } else {
      shape[i] = ishape[i];
    }
  }

  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kBroadcast);

MNM_OP_DECLARE("mnm.op.transpose", [](const CallValues& call) {
  const auto* args = call->args.as<TransposeArgs>();
  CHECK(args != nullptr);
  const std::vector<int64_t>& axes = args->axes;
  const DLTensor* x = args->x;
  int64_t* ishape = x->shape;
  int ndim = x->ndim;

  std::vector<int64_t> oshape(ndim, -1);
  if (axes.size() != 0) {
    CHECK_EQ(ndim, axes.size());
    for (int i = 0; i < ndim; ++i) {
      int axis = axes[i] >= 0 ? axes[i] : axes[i] + ndim;
      oshape[i] = ishape[axis];
    }
  } else {
    for (int i = 0; i < ndim; ++i) {
      oshape[i] = ishape[ndim - i - 1];
    }
  }
  call->out = TensorValue::Assemble(x->ctx, x->dtype, oshape);
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

MNM_OP_DECLARE("mnm.op.transpose_dx", [](const CallValues& call) {
  const auto* args = call->args.as<TransposeDxArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

MNM_OP_DECLARE("mnm.op.broadcast_to_like", [](const CallValues& call) {
  const auto* args = call->args.as<BroadcastToLikeArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  DLTensor* broadcast_type = args->broadcast_type;
  std::vector<int64_t> shape(broadcast_type->shape, broadcast_type->shape + broadcast_type->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/broadcast_type->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kBroadcast);

MNM_OP_DECLARE("mnm.op.stack", [](const CallValues& call) {
  const auto* args = call->args.as<StackArgs>();
  CHECK(args != nullptr);
  const std::vector<BaseTensorValue>& x = args->x;
  CHECK_GE(x.size(), 1U);
  DLTensor* y0 = x[0];

  int axis = args->axis;
  CHECK(-y0->ndim <= axis && axis < y0->ndim)
      << "ValueError: invalid axis = " << axis << " on ndim = " << y0->ndim;
  axis = axis < 0 ? axis + y0->ndim + 1 : axis;

  int64_t stack_dim = 0;
  for (auto i : x) {
    DLTensor* y = i;
    CHECK(y->ndim == y0->ndim);
    for (int k = 0; k < y0->ndim; ++k) {
      CHECK(y->shape[k] == y0->shape[k]);
    }
    stack_dim += 1;
  }
  std::vector<int64_t> shape;
  shape.reserve(y0->ndim + 1);

  for (int i = 0; i < axis; i++) {
    shape.emplace_back(y0->shape[i]);
  }
  shape.emplace_back(stack_dim);
  for (int i = axis; i < y0->ndim; i++) {
    shape.emplace_back(y0->shape[i]);
  }
  call->out = TensorValue::Assemble(/*ctx=*/y0->ctx,
                                    /*dtype=*/y0->dtype,
                                    /*shape=*/shape);
  call->ctx = y0->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

void StackDx(const CallValues& call) {
  const auto* args = call->args.as<StackArgs>();
  CHECK(args != nullptr);
  const std::vector<BaseTensorValue>& x = args->x;
  CHECK_GE(x.size(), 1U);
  DLTensor* y0 = x[0];
  int axis = NormalizeAxis(args->axis, y0->ndim);
  call->callee = ir::NullValue<OpValue>();
  ScalarValue sections_v = ScalarValue::make((int)x.size());
  ScalarValue axis_v = ScalarValue::make(axis);
  call->out = TupleValue::make({sections_v, axis_v});
}

MNM_OP_DECLARE("mnm.op.stack_dx", StackDx).set_attr<TOpPattern>("TOpPattern", kInjective);

MNM_OP_DECLARE("mnm.op.concatenate", [](const CallValues& call) {
  const auto* args = call->args.as<ConcatenateArgs>();
  CHECK(args != nullptr);
  const std::vector<BaseTensorValue>& x = args->x;
  CHECK_GE(x.size(), 1U);
  DLTensor* y0 = x[0];
  int axis = NormalizeAxis(args->axis, y0->ndim);
  int64_t dimsize = 0;
  for (auto i : x) {
    DLTensor* y = i;
    CHECK(y->ndim == y0->ndim);
    for (int k = 0; k < y0->ndim; ++k) {
      if (k != axis) {
        CHECK(y->shape[k] == y0->shape[k]);
      }
    }
    dimsize += y->shape[axis];
  }
  std::vector<int64_t> shape(y0->shape, y0->shape + y0->ndim);
  shape[axis] = dimsize;
  call->out = TensorValue::Assemble(/*ctx=*/y0->ctx,
                                    /*dtype=*/y0->dtype,
                                    /*shape=*/shape);
  call->ctx = y0->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

void ConcatenateDx(const CallValues& call) {
  const auto* args = call->args.as<ConcatenateArgs>();
  CHECK(args != nullptr);
  const std::vector<BaseTensorValue>& x = args->x;
  int axis = args->axis;
  ir::Array<Value> res;
  if (x.size() > 0U) {
    DLTensor* y0 = x[0];
    axis = NormalizeAxis(axis, y0->ndim);
    int64_t acc = 0;
    for (size_t i = 0; i + 1 < x.size(); ++i) {
      DLTensor* x0 = x[i];
      acc += x0->shape[axis];
      res.push_back(ScalarValue::make(acc));
    }
  }
  call->callee = ir::NullValue<OpValue>();
  call->out = TupleValue::make(res);
}

MNM_OP_DECLARE("mnm.op.concatenate_dx", ConcatenateDx)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

MNM_OP_DECLARE("mnm.op.split", [](const CallValues& call) {
  const auto* args = call->args.as<SplitArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  int axis = NormalizeAxis(args->axis, x->ndim);
  std::vector<TensorValue> ret;

  value::Value indices_or_sections = args->indices_or_sections;

  // indices_or_sections can be of 2 types - Integer or a tuple. The 2 types are
  // handled differently.
  if (const auto* scalar = indices_or_sections.as<IntValueObj>()) {
    // Handling first type - integer scalar - sections
    int64_t sections = scalar->data;
    CHECK_EQ(x->shape[axis] % sections, 0)
        << "indices_or_sections need to be able to divide input.shape[axis]";

    for (size_t i = 0; i < sections; ++i) {
      std::vector<int64_t> oshape(x->shape, x->shape + x->ndim);
      oshape[axis] = oshape[axis] / sections;
      ret.push_back(TensorValue::Assemble(/*ctx=*/x->ctx,
                                          /*dtype=*/x->dtype,
                                          /*shape=*/oshape));
    }
  } else if (const auto* tup = indices_or_sections.as<TupleValueObj>()) {
    // Handling second type - tuple values - indices
    std::vector<int64_t> indices;
    for (auto field : tup->fields) {
      auto int_value = field.as<IntValueObj>();
      indices.push_back(int_value->data);
    }
    indices.push_back(x->shape[axis]);
    int64_t begin = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      std::vector<int64_t> oshape(x->shape, x->shape + x->ndim);
      oshape[axis] = indices[i] - begin;
      begin = indices[i];
      ret.push_back(TensorValue::Assemble(/*ctx=*/x->ctx,
                                          /*dtype=*/x->dtype,
                                          /*shape=*/oshape));
    }
  }
  call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

MNM_OP_DECLARE("mnm.op.clip", [](const CallValues& call) {
  const auto* args = call->args.as<ClipArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kElemWise);

MNM_OP_DECLARE("mnm.op.clip_dx", [](const CallValues& call) {
  const auto* args = call->args.as<ClipDxArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kElemWise);

MNM_OP_DECLARE("mnm.op.cast", [](const CallValues& call) {
  const auto* args = call->args.as<CastArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  std::string dtype = args->dtype;
  std::vector<int64_t> shape(data->shape, data->shape + data->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/data->ctx,
                                    /*dtype=*/ir::String2DLDataType(dtype),
                                    /*shape=*/shape);
  call->ctx = data->ctx;
}).set_attr<TOpPattern>("TOpPattern", kElemWise);

MNM_OP_DECLARE("mnm.op.cast_like", [](const CallValues& call) {
  const auto* args = call->args.as<CastLikeArgs>();
  CHECK(args != nullptr);
  DLTensor* dtype_like = args->dtype_like;
  std::vector<int64_t> shape(dtype_like->shape, dtype_like->shape + dtype_like->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/dtype_like->ctx,
                                    /*dtype=*/dtype_like->dtype,
                                    /*shape=*/shape);
  call->ctx = dtype_like->ctx;
}).set_attr<TOpPattern>("TOpPattern", kElemWise);

MNM_OP_DECLARE("mnm.op.gather_nd", [](const CallValues& call) {
  const auto* args = call->args.as<GatherNdArgs>();
  CHECK(args != nullptr);
  const DLTensor* data = args->data;
  const DLTensor* indices = args->indices;
  int64_t* dshape = data->shape;
  int64_t* ishape = indices->shape;
  int ddim = data->ndim;
  int idim = indices->ndim;
  int odim = idim - 1 + ddim - ishape[0];
  CHECK_LE(ishape[0], ddim);

  std::vector<int64_t> oshape(odim, -1);
  for (int i = 0; i < odim; ++i) {
    if (i + 1 < idim) {
      oshape[i] = ishape[i + 1];
    } else {
      oshape[i] = dshape[i + 1 - idim + ishape[0]];
    }
  }
  call->out = TensorValue::Assemble(data->ctx, data->dtype, oshape);
  call->ctx = data->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

MNM_OP_DECLARE("mnm.op.gather_nd_dx", [](const CallValues& call) {
  const auto* args = call->args.as<GatherNdDxArgs>();
  CHECK(args != nullptr);
  const DLTensor* data = args->data;
  std::vector<int64_t> shape(data->shape, data->shape + data->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/data->ctx,
                                    /*dtype=*/data->dtype,
                                    /*shape=*/shape);
  call->ctx = data->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

MNM_OP_DECLARE("mnm.op.squeeze", [](const CallValues& call) {
  const auto* args = call->args.as<SqueezeArgs>();
  CHECK(args != nullptr);
  const std::vector<int64_t>& axis = args->axis;
  const DLTensor* x = args->x;
  int64_t* ishape = x->shape;
  int ndim = x->ndim;
  std::vector<int64_t> normalized_axis;

  for (int i = 0; i < axis.size(); i++) {
    normalized_axis.push_back(axis[i] >= 0 ? axis[i] : axis[i] + ndim);
  }

  std::vector<int64_t> oshape;
  if (normalized_axis.size() != 0) {
    for (int axis_dim = 0; axis_dim < ndim; ++axis_dim) {
      if (std::find(normalized_axis.begin(), normalized_axis.end(), axis_dim) ==
          normalized_axis.end()) {
        oshape.push_back(ishape[axis_dim]);
      } else {
        CHECK_EQ(ishape[axis_dim], 1) << "Axis to be squeezed is not of size 1";
      }
    }
  } else {
    for (int axis_dim = 0; axis_dim < ndim; ++axis_dim) {
      if (ishape[axis_dim] != 1) {
        oshape.push_back(ishape[axis_dim]);
      }
    }
  }
  call->out = TensorValue::Assemble(x->ctx, x->dtype, oshape);
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace declare
}  // namespace op
}  // namespace mnm
