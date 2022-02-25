/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/unary.cc
 * \brief Declaration of unary operators
 */
#include "raf/op.h"
#include "raf/tensor.h"
#include "../schema/ufunc.h"
#include "../ty/utils.h"
#include <cmath>
#include "math.h"

namespace raf {
namespace op {
namespace declare {

using namespace raf::op::schema;
using namespace raf::value;
using tvm::Array;
using tvm::Downcast;

#define RAF_SWITCH_SCALAR(var, value, body)                     \
  do {                                                          \
    if (const auto* var = (value).as<IntValueObj>()) {          \
      body;                                                     \
    } else if (const auto* var = (value).as<FloatValueObj>()) { \
      body;                                                     \
    } else if (const auto* var = (value).as<BoolValueObj>()) {  \
      body;                                                     \
    }                                                           \
  } while (0)

#define RAF_UNARY_SCALAR(op, x)                 \
  RAF_SWITCH_SCALAR(v, x, {                     \
    call->callee = ir::NullValue<OpValue>();    \
    call->out = ScalarValue::make(op v->value); \
    return;                                     \
  })

#define RAF_UNARY_TENSOR(x)                     \
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

RAF_OP_DECLARE("raf.op.negative", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  RAF_UNARY_SCALAR(-, args->x);
  RAF_UNARY_TENSOR(args->x)
  LOG(FATAL) << "NotImplementedError";
  throw;
});

RAF_OP_DECLARE("raf.op.rsqrt", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  RAF_SWITCH_SCALAR(v, args->x, {
    call->callee = ir::NullValue<OpValue>();
    double a = v->value;
    double result = 1.0 / sqrt(a);
    call->out = ScalarValue::make(result);
    return;
  });
  RAF_UNARY_TENSOR(args->x);
  LOG(FATAL) << "NotImplementedError";
  throw;
});

RAF_OP_DECLARE("raf.op.logical_not", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  RAF_UNARY_SCALAR(!, args->x);
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

RAF_OP_DECLARE("raf.op.relu", Unary);
RAF_OP_DECLARE("raf.op.gelu", Unary);
RAF_OP_DECLARE("raf.op.tanh", Unary);
RAF_OP_DECLARE("raf.op.sigmoid", Unary);
RAF_OP_DECLARE("raf.op.copy", Unary);
RAF_OP_DECLARE("raf.op.abs", Unary);
RAF_OP_DECLARE("raf.op.ceil", Unary);
RAF_OP_DECLARE("raf.op.floor", Unary);
RAF_OP_DECLARE("raf.op.log", Unary);
RAF_OP_DECLARE("raf.op.log2", Unary);
RAF_OP_DECLARE("raf.op.exp", Unary);
RAF_OP_DECLARE("raf.op.cos", Unary);
RAF_OP_DECLARE("raf.op.sin", Unary);
RAF_OP_DECLARE("raf.op.sign", Unary);
RAF_OP_DECLARE("raf.op.round", Unary);
RAF_OP_DECLARE("raf.op.erf", Unary);
RAF_OP_DECLARE("raf.op.sqrt", Unary);
RAF_OP_DECLARE("raf.op.atan", Unary);
RAF_OP_DECLARE("raf.op.zeros_like", Unary);
RAF_OP_DECLARE("raf.op.ones_like", Unary);

RAF_OP_DECLARE("raf.op.trunc", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  RAF_SWITCH_SCALAR(v, args->x, {
    call->callee = ir::NullValue<OpValue>();
    double a = v->value;
    double result = trunc(a);
    call->out = ScalarValue::make(result);
    return;
  });
  RAF_UNARY_TENSOR(args->x);
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

RAF_OP_DECLARE("raf.op.relu_dx", UnaryDx);
RAF_OP_DECLARE("raf.op.gelu_dx", UnaryDx);
RAF_OP_DECLARE("raf.op.tanh_dx", UnaryDx);
// TODO(@yzhliu, @icemelon9): We don't have tvm impl for sigmoid_dx. So currently don't fuse it.
RAF_OP_DECLARE("raf.op.sigmoid_dx", UnaryDx).set_attr<TOpPattern>("TOpPattern", kOpaque);
RAF_OP_DECLARE("raf.op.erf_dx", UnaryDx);
RAF_OP_DECLARE("raf.op.sqrt_dx", UnaryDx);

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
RAF_OP_DECLARE("raf.op.shape", Shape).set_attr<TOpPattern>("TOpPattern", kOpaque);

RAF_OP_DECLARE("raf.op.ndarray_size", [](const CallValues& call) {
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

RAF_OP_DECLARE("raf.op.numel", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);

  DLTensor* x = args->x;
  CHECK(x != nullptr);
  call->out = TensorValue::Assemble(/*dev=*/Device(DevType::kCPU(), 0),
                                    /*dtype=*/DType(DTypeCode::kInt(), 32),
                                    /*shape=*/std::vector<int64_t>());
  call->device = x->device;
});

RAF_OP_DECLARE("raf.op.shape_as_tensor", [](const CallValues& call) {
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
}  // namespace raf
