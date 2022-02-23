/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/unary.cc
 * \brief Declaration of unary operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/ufunc.h"
#include "../ty/utils.h"
#include <cmath>
#include "math.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;
using tvm::Array;
using tvm::Downcast;

#define MNM_SWITCH_SCALAR(var, value, body)                     \
  do {                                                          \
    if (const auto* var = (value).as<IntValueObj>()) {          \
      body;                                                     \
    } else if (const auto* var = (value).as<FloatValueObj>()) { \
      body;                                                     \
    } else if (const auto* var = (value).as<BoolValueObj>()) {  \
      body;                                                     \
    }                                                           \
  } while (0)

#define MNM_UNARY_SCALAR(op, x)                 \
  MNM_SWITCH_SCALAR(v, x, {                     \
    call->callee = ir::NullValue<OpValue>();    \
    call->out = ScalarValue::make(op v->value); \
    return;                                     \
  })

#define MNM_UNARY_TENSOR(x)                     \
  if (x->IsInstance<TensorValueObj>()) {        \
    const TensorValue& tv = MakeUnaryTensor(x); \
    call->out = tv;                             \
    call->device = tv->tensor->device;          \
    return;                                     \
  }

TensorValue MakeUnaryTensor(DLTensor* x) {
  int ndim = x->ndim;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  return TensorValue::Assemble(x->device, x->dtype, shape);
}

MNM_OP_DECLARE("mnm.op.negative", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  MNM_UNARY_SCALAR(-, args->x);
  MNM_UNARY_TENSOR(args->x)
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.rsqrt", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  MNM_SWITCH_SCALAR(v, args->x, {
    call->callee = ir::NullValue<OpValue>();
    double a = v->value;
    double result = 1.0 / sqrt(a);
    call->out = ScalarValue::make(result);
    return;
  });
  MNM_UNARY_TENSOR(args->x);
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.logical_not", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  MNM_UNARY_SCALAR(!, args->x);
  LOG(FATAL) << "NotImplementedError";
  throw;
});

void Unary(const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*dev=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->device = x->device;
  for (int i = 0, e = shape.size(); i < e; ++i) {
    if (shape[i] == 0) {
      call->callee = ir::NullValue<OpValue>();
      break;
    }
  }
}

MNM_OP_DECLARE("mnm.op.relu", Unary);
MNM_OP_DECLARE("mnm.op.gelu", Unary);
MNM_OP_DECLARE("mnm.op.tanh", Unary);
MNM_OP_DECLARE("mnm.op.sigmoid", Unary);
MNM_OP_DECLARE("mnm.op.copy", Unary);
MNM_OP_DECLARE("mnm.op.abs", Unary);
MNM_OP_DECLARE("mnm.op.ceil", Unary);
MNM_OP_DECLARE("mnm.op.floor", Unary);
MNM_OP_DECLARE("mnm.op.log", Unary);
MNM_OP_DECLARE("mnm.op.log2", Unary);
MNM_OP_DECLARE("mnm.op.exp", Unary);
MNM_OP_DECLARE("mnm.op.cos", Unary);
MNM_OP_DECLARE("mnm.op.sin", Unary);
MNM_OP_DECLARE("mnm.op.sign", Unary);
MNM_OP_DECLARE("mnm.op.round", Unary);
MNM_OP_DECLARE("mnm.op.erf", Unary);
MNM_OP_DECLARE("mnm.op.sqrt", Unary);
MNM_OP_DECLARE("mnm.op.atan", Unary);
MNM_OP_DECLARE("mnm.op.zeros_like", Unary);
MNM_OP_DECLARE("mnm.op.ones_like", Unary);

MNM_OP_DECLARE("mnm.op.trunc", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  MNM_SWITCH_SCALAR(v, args->x, {
    call->callee = ir::NullValue<OpValue>();
    double a = v->value;
    double result = trunc(a);
    call->out = ScalarValue::make(result);
    return;
  });
  MNM_UNARY_TENSOR(args->x);
  LOG(FATAL) << "NotImplementedError";
  throw;
});

void UnaryDx(const CallValues& call) {
  // TODO(@junrushao1994): sanity check
  const auto* args = call->args.as<UnaryDxArgs>();
  CHECK(args != nullptr);
  CHECK(args->x.defined() || args->y.defined());
  DLTensor* source;
  if (args->x.defined()) {
    source = args->x.value();
  } else {
    source = args->y.value();
  }
  std::vector<int64_t> shape(source->shape, source->shape + source->ndim);
  call->out = TensorValue::Assemble(/*dev=*/source->device,
                                    /*dtype=*/source->dtype,
                                    /*shape=*/shape);
  call->device = source->device;
}

MNM_OP_DECLARE("mnm.op.relu_dx", UnaryDx);
MNM_OP_DECLARE("mnm.op.gelu_dx", UnaryDx);
MNM_OP_DECLARE("mnm.op.tanh_dx", UnaryDx);
// TODO(@yzhliu, @icemelon9): We don't have tvm impl for sigmoid_dx. So currently don't fuse it.
MNM_OP_DECLARE("mnm.op.sigmoid_dx", UnaryDx).set_attr<TOpPattern>("TOpPattern", kOpaque);
MNM_OP_DECLARE("mnm.op.erf_dx", UnaryDx);
MNM_OP_DECLARE("mnm.op.sqrt_dx", UnaryDx);

void Shape(const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  auto x_type = op::GetType(args->x);
  std::vector<Value> shape;
  if (auto* t_type = x_type.as<ir::TensorTypeNode>()) {
    for (auto ty : t_type->shape) {
      shape.push_back(ScalarValue::make(ty.as<ir::IntImmNode>()->value));
    }
    call->out = TupleValue::make(shape);
  } else {
    call->out = ir::NullValue<Value>();
  }
  call->callee = ir::NullValue<OpValue>();
}

// TODO(@icemelon9): Currently use opaque for shape related op.
MNM_OP_DECLARE("mnm.op.shape", Shape).set_attr<TOpPattern>("TOpPattern", kOpaque);

MNM_OP_DECLARE("mnm.op.ndarray_size", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);

  DLTensor* x = args->x;
  CHECK(x != nullptr);

  int64_t result = 1;
  int ndim = x->ndim;
  for (int i = 0; i < ndim; i++) {
    result *= x->shape[i];
  }

  result = ndim > 0 ? result : 0;
  call->out = ScalarValue::make(result);
  call->callee = ir::NullValue<OpValue>();
});

MNM_OP_DECLARE("mnm.op.numel", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);

  DLTensor* x = args->x;
  CHECK(x != nullptr);
  call->out = TensorValue::Assemble(/*dev=*/Device(DevType::kCPU(), 0),
                                    /*dtype=*/DType(DTypeCode::kInt(), 32),
                                    /*shape=*/std::vector<int64_t>());
  call->device = x->device;
});

MNM_OP_DECLARE("mnm.op.shape_as_tensor", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);

  DLTensor* x = args->x;
  CHECK(x != nullptr);
  call->out = TensorValue::Assemble(/*dev=*/Device(DevType::kCPU(), 0),
                                    /*dtype=*/DType(DTypeCode::kInt(), 32),
                                    /*shape=*/{x->ndim});
  call->device = x->device;
});

}  // namespace declare
}  // namespace op
}  // namespace mnm
