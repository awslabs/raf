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
});

MNM_OP_DECLARE("mnm.op.reshape", [](const CallValues &call) {
  const auto* args = call->args.as<ReshapeArgs>();
  CHECK(args != nullptr);
  DLTensor *x = args->x;
  const std::vector<int64_t> &shape = args->shape;
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
});

void ReshapeDx(const CallValues &call) {
  const auto* args = call->args.as<ReshapeArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  const int ndim = x->ndim;
  ir::Array<Value> res;
  for (int i = 0; i < ndim; i++) {
    res.push_back(ScalarValue::make(x->shape[i]));
  }
  call->callee = ir::NullValue<OpValue>();
  call->out = TupleValue::make(res);
}

MNM_OP_DECLARE("mnm.op.reshape_dx", ReshapeDx);

MNM_OP_DECLARE("mnm.op.take", [](const CallValues &call) {
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
});

MNM_OP_DECLARE("mnm.op.expand_dims", [](const CallValues& call) {
  const auto* args = call->args.as<ExpandDimsArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  int axis = NormalizeAxis(args->axis, x->ndim);
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
});

MNM_OP_DECLARE("mnm.op.sequence_mask", [](const CallValues &call) {
  const auto* args = call->args.as<SequenceMaskArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  // TODO(@hzfan): checks x.shape and sequence_length.shape
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
});

MNM_OP_DECLARE("mnm.op.broadcast_to", [](const CallValues &call) {
  const auto* args = call->args.as<BroadcastToArgs>();
  DLTensor* x = args->x;
  std::vector<int64_t> shape = args->shape;
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
});

MNM_OP_DECLARE("mnm.op.transpose", [](const CallValues &call) {
  const auto* args = call->args.as<TransposeArgs>();
  CHECK(args != nullptr);
  const std::vector<int64_t> &axes = args->axes;
  const DLTensor *x = args->x;
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
});

MNM_OP_DECLARE("mnm.op.transpose_dx", [](const CallValues &call) {
  const auto* args = call->args.as<TransposeDxArgs>();
  CHECK(args != nullptr);
  const DLTensor *x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
});

MNM_OP_DECLARE("mnm.op.broadcast_to_like", [](const CallValues &call) {
  const auto* args = call->args.as<BroadcastToLikeArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  DLTensor* broadcast_type = args->broadcast_type;
  std::vector<int64_t> shape(broadcast_type->shape,
                             broadcast_type->shape + broadcast_type->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/broadcast_type->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
});

MNM_OP_DECLARE("mnm.op.concatenate", [](const CallValues &call) {
  const auto* args = call->args.as<ConcatenateArgs>();
  CHECK(args != nullptr);
  const std::vector<TensorValue>& x = args->x;
  CHECK_GE(x.size(), 1U);
  DLTensor *y0 = x[0];
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
});


MNM_OP_DECLARE("mnm.op.split", [](const CallValues &call){
  const auto* args = call->args.as<SplitArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  std::vector<int64_t> indices_or_sections = args->indices_or_sections;
  int axis = NormalizeAxis(args->axis, x->ndim);
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  int64_t start = 0;
  int64_t end = 0;
  std::vector<TensorValue> ret;
  indices_or_sections.push_back(x->shape[axis]);
  for (size_t i = 0; i < indices_or_sections.size(); ++i) {
    start = end;
    end = indices_or_sections[i];
    shape[axis] = end - start;
    ret.push_back(TensorValue::Assemble(/*ctx=*/x->ctx,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape));
  }
  call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  call->ctx = x->ctx;
});

void ConcatenateDx(const CallValues &call) {
  const auto* args = call->args.as<ConcatenateArgs>();
  CHECK(args != nullptr);
  const std::vector<TensorValue>& x = args->x;
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

MNM_OP_DECLARE("mnm.op.concatenate_dx", ConcatenateDx);

MNM_OP_DECLARE("mnm.op.clip", [](const CallValues &call) {
  const auto* args = call->args.as<ClipArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
});

MNM_OP_DECLARE("mnm.op.clip_dx", [](const CallValues &call) {
  const auto* args = call->args.as<ClipDxArgs>();
  CHECK(args != nullptr);
  const DLTensor *x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
});

}  // namespace declare
}  // namespace op
}  // namespace mnm

