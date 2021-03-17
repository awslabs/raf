/*!
 * Copyright (c) 2019 by Contributors
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

#define MNM_UNARY_SCALAR(op, x)                \
  MNM_SWITCH_SCALAR(v, x, {                    \
    call->callee = ir::NullValue<OpValue>();   \
    call->out = ScalarValue::make(op v->data); \
    return;                                    \
  })

#define MNM_UNARY_TENSOR(x)                     \
  if (x->IsInstance<TensorValueObj>()) {        \
    const TensorValue& tv = MakeUnaryTensor(x); \
    call->out = tv;                             \
    call->device = tv->tensor->ctx;             \
    return;                                     \
  }

#define MNM_DECLARE_UNARY_OP(op_name, body) \
  MNM_OP_DECLARE(op_name, body).set_attr<TOpPattern>("TOpPattern", kElemWise)

TensorValue MakeUnaryTensor(DLTensor* x) {
  int ndim = x->ndim;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  return TensorValue::Assemble(x->ctx, x->dtype, shape);
}

MNM_DECLARE_UNARY_OP("mnm.op.negative", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  MNM_UNARY_SCALAR(-, args->x);
  MNM_UNARY_TENSOR(args->x)
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_DECLARE_UNARY_OP("mnm.op.rsqrt", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  MNM_SWITCH_SCALAR(v, args->x, {
    call->callee = ir::NullValue<OpValue>();
    double a = v->data;
    double result = 1.0 / sqrt(a);
    call->out = ScalarValue::make(result);
    return;
  });
  MNM_UNARY_TENSOR(args->x);
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_DECLARE_UNARY_OP("mnm.op.logical_not", [](const CallValues& call) {
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
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->device = x->ctx;
  for (int i = 0, e = shape.size(); i < e; ++i) {
    if (shape[i] == 0) {
      call->callee = ir::NullValue<OpValue>();
      break;
    }
  }
}

MNM_DECLARE_UNARY_OP("mnm.op.relu", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.tanh", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.sigmoid", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.copy", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.abs", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.ceil", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.floor", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.log", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.exp", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.cos", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.sin", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.sign", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.round", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.erf", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.sqrt", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.atan", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.zeros_like", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.ones_like", Unary);

MNM_DECLARE_UNARY_OP("mnm.op.trunc", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  MNM_SWITCH_SCALAR(v, args->x, {
    call->callee = ir::NullValue<OpValue>();
    double a = v->data;
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
  call->out = TensorValue::Assemble(/*ctx=*/source->ctx,
                                    /*dtype=*/source->dtype,
                                    /*shape=*/shape);
  call->device = source->ctx;
}

MNM_DECLARE_UNARY_OP("mnm.op.relu_dx", UnaryDx);
MNM_DECLARE_UNARY_OP("mnm.op.tanh_dx", UnaryDx);
MNM_DECLARE_UNARY_OP("mnm.op.sigmoid_dx", UnaryDx);
MNM_DECLARE_UNARY_OP("mnm.op.erf_dx", UnaryDx);
MNM_DECLARE_UNARY_OP("mnm.op.sqrt_dx", UnaryDx);

void Shape(const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  auto x_type = op::type::GetType(args->x);
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

}  // namespace declare
}  // namespace op
}  // namespace mnm
