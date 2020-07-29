/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/unary.cc
 * \brief Declaration of unary operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/ufunc.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

#define MNM_SWITCH_SCALAR(var, value, body)                     \
  do {                                                          \
    if (const auto* var = (value).as<IntValueObj>()) {          \
      body;                                                     \
    } else if (const auto* var = (value).as<FloatValueObj>()) { \
      body;                                                     \
    } else if (const auto* var = (value).as<BoolValueObj>()) {  \
      body;                                                     \
    }                                                           \
  } while (0);

#define MNM_UNARY_SCALAR(op, x)                \
  MNM_SWITCH_SCALAR(v, x, {                    \
    call->callee = ir::NullValue<OpValue>();   \
    call->out = ScalarValue::make(op v->data); \
    return;                                    \
  })

#define MNM_DECLARE_UNARY_OP(op_name, body) \
  MNM_OP_DECLARE(op_name, body) \
    .set_attr<TOpPattern>("TOpPattern", kElemWise)

MNM_DECLARE_UNARY_OP("mnm.op.negative", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_UNARY_SCALAR(-, args->x);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_DECLARE_UNARY_OP("mnm.op.logical_not", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_UNARY_SCALAR(!, args->x);
  }
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
  call->ctx = x->ctx;
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
MNM_DECLARE_UNARY_OP("mnm.op.cos", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.erf", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.sqrt", Unary);
MNM_DECLARE_UNARY_OP("mnm.op.atan", Unary);

void UnaryDx(const CallValues& call) {
  // TODO(@junrushao1994): sanity check
  const auto* args = call->args.as<UnaryDxArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}

MNM_DECLARE_UNARY_OP("mnm.op.relu_dx", UnaryDx);
MNM_DECLARE_UNARY_OP("mnm.op.tanh_dx", UnaryDx);
MNM_DECLARE_UNARY_OP("mnm.op.sigmoid_dx", UnaryDx);
MNM_DECLARE_UNARY_OP("mnm.op.erf_dx", UnaryDx);
MNM_DECLARE_UNARY_OP("mnm.op.sqrt_dx", UnaryDx);

void Shape(const CallValues &call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<Value> shape;
  std::for_each(x->shape, x->shape + x->ndim, [&shape](int64_t x) {
    shape.push_back(ScalarValue::make(x));
  });
  call->out = TupleValue::make(shape);
  call->callee = ir::NullValue<OpValue>();
}

// TODO(@icemelon9): Currently use opaque for shape related op.
MNM_OP_DECLARE("mnm.op.shape", Shape).set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace declare
}  // namespace op
}  // namespace mnm
